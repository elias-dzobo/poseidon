"""
End-to-end flood detection pipeline.

Orchestrates the full chain for a given AOI and date range:

  1. Download Sentinel-1 + Sentinel-2 scenes (ingestion)
  2. Preprocess SAR and optical data
  3. Build feature stacks
  4. Run U-Net flood segmentation → flood mask + probability map
  5. Generate SAR false-color composite with mask overlay
  6. Run LFM2.5 VL → structured flood report
  7. Write GeoTIFF, GeoJSON, JSON report, and text report

Usage
-----
    python -m src.pipeline.run_detection \
        --start 2024-06-10 \
        --end   2024-06-15 \
        --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.ingestion.sentinel1 import Sentinel1Downloader
from src.ingestion.sentinel2 import Sentinel2Downloader
from src.ingestion.auxiliary import DEMLoader, JRCWaterLoader
from src.preprocessing.sar import SARPreprocessor
from src.preprocessing.optical import OpticalPreprocessor
from src.preprocessing.features import FeatureBuilder
from src.preprocessing.composite import SARCompositeGenerator
from src.models.unet.predict import FloodPredictor
from src.models.vlm.model import LFMFloodReporter
from src.models.vlm.generate import build_metadata, report_to_json, report_to_text
from src.output.maps import FloodMapWriter
from src.output.reports import ReportWriter

logger = logging.getLogger(__name__)


class FloodDetectionPipeline:
    """
    Orchestrates the full flood detection and reporting pipeline.

    Parameters
    ----------
    settings_path : str | Path
        Path to config/settings.yaml.
    unet_weights : str | Path
        Path to trained U-Net checkpoint (.pt).
    vlm_lora_path : str | Path, optional
        Path to fine-tuned LoRA adapter directory.
    output_dir : str | Path
        Root directory for pipeline outputs.
    """

    def __init__(
        self,
        settings_path: str | Path = "config/settings.yaml",
        unet_weights: str | Path = "models/weights/unet/best.pt",
        vlm_lora_path: Optional[str | Path] = None,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.settings = self._load_settings(settings_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg = self.settings
        bbox = cfg["region"]["aoi_bbox"]

        # Ingestion
        self.s1_downloader = Sentinel1Downloader(
            output_dir=cfg["paths"]["raw_data"] + "/sentinel1"
        )
        self.s2_downloader = Sentinel2Downloader(
            output_dir=cfg["paths"]["raw_data"] + "/sentinel2"
        )
        self.dem_loader = DEMLoader(
            output_dir=cfg["paths"]["raw_data"] + "/dem"
        )
        self.jrc_loader = JRCWaterLoader(
            output_dir=cfg["paths"]["raw_data"] + "/jrc"
        )

        # Preprocessing
        self.sar_preprocessor = SARPreprocessor(
            output_dir=cfg["paths"]["processed_data"] + "/sentinel1"
        )
        self.optical_preprocessor = OpticalPreprocessor(
            output_dir=cfg["paths"]["processed_data"] + "/sentinel2"
        )
        self.feature_builder = FeatureBuilder(
            output_dir=cfg["paths"]["features"]
        )
        self.composite_generator = SARCompositeGenerator()

        # Models
        self.predictor = FloodPredictor(model_path=unet_weights)
        self.reporter = LFMFloodReporter(lora_path=vlm_lora_path)

        # Output writers
        self.map_writer = FloodMapWriter(output_dir=self.output_dir / "maps")
        self.report_writer = ReportWriter(output_dir=self.output_dir / "reports")

    def run(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> list[dict]:
        """
        Execute the full pipeline for a date range.

        Parameters
        ----------
        start_date : str | datetime
            Start of the search window.
        end_date : str | datetime
            End of the search window.

        Returns
        -------
        list[dict]
            Summary of each processed event:
            {event_id, date, severity, flood_area_km2, outputs}
        """
        cfg = self.settings
        bbox = cfg["region"]["aoi_bbox"]

        logger.info(f"Pipeline start: {start_date} → {end_date}")

        # --- Step 1: Data ingestion ---
        logger.info("Step 1/7: Downloading Sentinel-1 scenes...")
        s1_paths = self.s1_downloader.download_range(bbox, start_date, end_date)

        logger.info("Step 1/7: Downloading Sentinel-2 scenes...")
        s2_paths = self.s2_downloader.download_range(
            bbox, start_date, end_date,
            cloud_cover_max=cfg["sentinel2"]["cloud_cover_max"]
        )

        logger.info("Step 1/7: Loading auxiliary data...")
        dem_path = self.dem_loader.download(bbox)
        slope_path = self.dem_loader.compute_slope(dem_path)
        hand_path = self.dem_loader.compute_hand(dem_path, drainage_mask_path=None)
        jrc_path = self.jrc_loader.download(bbox)
        permanent_water_path = self.jrc_loader.get_permanent_water_mask(jrc_path)

        results = []

        # Process each S1 scene (paired with nearest S2 where available)
        for s1_path in s1_paths:
            event_result = self._process_scene(
                s1_path=s1_path,
                s2_paths=s2_paths,
                dem_path=dem_path,
                slope_path=slope_path,
                hand_path=hand_path,
                permanent_water_path=permanent_water_path,
            )
            results.append(event_result)

        logger.info(f"Pipeline complete. Processed {len(results)} events.")
        return results

    def _process_scene(
        self,
        s1_path: Path,
        s2_paths: list[Path],
        dem_path: Path,
        slope_path: Path,
        hand_path: Path,
        permanent_water_path: Path,
    ) -> dict:
        """Process a single S1 scene through the full detection + reporting chain."""
        cfg = self.settings
        bbox = cfg["region"]["aoi_bbox"]

        # --- Step 2: Preprocessing ---
        logger.info("Step 2/7: Preprocessing SAR...")
        sar_path = self.sar_preprocessor.process(s1_path, bbox=bbox)

        s2_match = self._find_nearest_s2(s1_path, s2_paths)
        optical_path = None
        if s2_match:
            logger.info("Step 2/7: Preprocessing optical...")
            optical_path = self.optical_preprocessor.process(s2_match, bbox=bbox)

        # --- Step 3: Feature stacking ---
        logger.info("Step 3/7: Building feature stack...")
        feature_path = self.feature_builder.build(
            sar_path=sar_path,
            optical_path=optical_path,
            hand_path=hand_path,
            slope_path=slope_path,
            permanent_water_path=permanent_water_path,
            output_name=s1_path.stem,
        )

        # --- Step 4: Flood segmentation ---
        logger.info("Step 4/7: Running U-Net flood detection...")
        prob_path, mask_path = self.predictor.predict(feature_path)

        # --- Step 5: Generate composite for VLM ---
        logger.info("Step 5/7: Generating SAR composite...")
        composite_path = self.composite_generator.generate(
            sar_path=sar_path,
            flood_mask_path=mask_path,
            output_name=s1_path.stem,
        )

        # --- Step 6: Generate flood report ---
        logger.info("Step 6/7: Generating flood report with LFM2.5 VL...")
        acquisition_date = self._extract_acquisition_date(s1_path)
        metadata = build_metadata(
            acquisition_date=acquisition_date,
            region=cfg["region"],
            data_sources=["Sentinel-1 IW GRD"]
            + (["Sentinel-2 L2A"] if optical_path else []),
        )
        report = self.reporter.generate(composite_path, metadata)

        # --- Step 7: Write outputs ---
        logger.info("Step 7/7: Writing outputs...")
        geotiff_path = self.map_writer.write_geotiff(mask_path, report)
        geojson_path = self.predictor.vectorize_mask(mask_path)
        json_report_path = self.report_writer.write_json(report)
        text_report_path = self.report_writer.write_text(report)

        return {
            "event_id": report.event_id,
            "date": report.acquisition_date,
            "severity": report.severity,
            "flood_area_km2": report.flood_extent.total_area_km2,
            "outputs": {
                "geotiff": str(geotiff_path),
                "geojson": str(geojson_path),
                "json_report": str(json_report_path),
                "text_report": str(text_report_path),
            },
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _load_settings(path: str | Path) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _find_nearest_s2(s1_path: Path, s2_paths: list[Path]) -> Optional[Path]:
        """
        Match the S1 scene to the temporally nearest S2 scene.
        Returns None if no S2 scenes are available.
        """
        # TODO: parse acquisition dates from filenames, find minimum time delta
        if not s2_paths:
            return None
        return s2_paths[0]  # placeholder

    @staticmethod
    def _extract_acquisition_date(scene_path: Path) -> str:
        """
        Extract acquisition datetime from Sentinel-1 filename.

        Sentinel-1 filenames encode the date as: S1A_IW_GRDH_..._YYYYMMDDTHHMMSS_...
        """
        # TODO: parse date from filename using regex
        return datetime.utcnow().isoformat()  # placeholder


def main():
    parser = argparse.ArgumentParser(description="Run Poseidon flood detection pipeline")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--unet-weights", default="models/weights/unet/best.pt")
    parser.add_argument("--vlm-lora", default=None)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pipeline = FloodDetectionPipeline(
        settings_path=args.config,
        unet_weights=args.unet_weights,
        vlm_lora_path=args.vlm_lora,
        output_dir=args.output_dir,
    )
    results = pipeline.run(args.start, args.end)

    for r in results:
        print(f"\n[{r['event_id']}] {r['date']} — Severity: {r['severity'].upper()}")
        print(f"  Flooded area: {r['flood_area_km2']:.1f} km²")
        for k, v in r["outputs"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
