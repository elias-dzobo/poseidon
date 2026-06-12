"""
Auxiliary data loaders.

DEMLoader
---------
Downloads Copernicus DEM GLO-30 tiles (30m) from the public AWS S3 bucket.
No credentials required.

Tile naming convention:
  s3://copernicus-dem-30m/
    Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM/
      Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM.tif

Derives slope (GDAL) and HAND (pysheds) from the merged DEM.

JRCWaterLoader
--------------
Downloads JRC Global Surface Water (GSW) seasonality tiles from Google Cloud:
  https://storage.googleapis.com/global-surface-water/downloads2021/
    seasonality/seasonality_{lon}{hem}_{lat}{hem}_v1_4.tif

Tile grid: 10° × 10°, named by SW corner (e.g. 0E_0N covers 0–10°E, 0–10°N).
Accra (~5.6°N, 0.2°W) falls in tile: 10W_0N.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import rasterio
import rasterio.merge
import rasterio.transform
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CHUNK = 4 * 1024 * 1024  # 4 MB download chunks


# ---------------------------------------------------------------------------
# DEM Loader
# ---------------------------------------------------------------------------

class DEMLoader:
    """
    Downloads Copernicus DEM GLO-30 tiles and derives slope + HAND.

    Parameters
    ----------
    output_dir : str | Path
        Root directory for DEM products.
        Subdirectories tiles/, merged/, derived/ are created automatically.
    """

    # Public AWS bucket — no auth required
    _S3_BASE = (
        "https://copernicus-dem-30m.s3.amazonaws.com/"
        "Copernicus_DSM_COG_10_{lat_code}_00_{lon_code}_00_DEM/"
        "Copernicus_DSM_COG_10_{lat_code}_00_{lon_code}_00_DEM.tif"
    )
    # pysheds stream accumulation threshold (pixels above this = stream network)
    _STREAM_ACC_THRESHOLD = 1000

    def __init__(self, output_dir: str | Path = "data/raw/dem") -> None:
        self.output_dir = Path(output_dir)
        self.tiles_dir = self.output_dir / "tiles"
        self.merged_dir = self.output_dir / "merged"
        self.derived_dir = self.output_dir / "derived"
        for d in (self.tiles_dir, self.merged_dir, self.derived_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, bbox: list[float]) -> Path:
        """
        Download all Copernicus DEM 30m tiles covering the bounding box,
        merge them, and clip to the AOI.

        Parameters
        ----------
        bbox : list[float]
            [min_lat, min_lon, max_lat, max_lon]

        Returns
        -------
        Path
            Path to the clipped, merged DEM GeoTIFF (EPSG:4326, 30m).
        """
        tiles = self._tiles_for_bbox(bbox)
        local_tiles: list[Path] = []

        for lat_code, lon_code in tiles:
            tile_path = self._download_tile(lat_code, lon_code)
            if tile_path:
                local_tiles.append(tile_path)

        if not local_tiles:
            raise RuntimeError("No DEM tiles downloaded — check bbox coordinates")

        merged_path = self._merge_and_clip(local_tiles, bbox)
        logger.info(f"DEM ready: {merged_path}")
        return merged_path

    def compute_slope(self, dem_path: Path) -> Path:
        """
        Derive a slope raster (degrees) from a DEM GeoTIFF using GDAL.

        Parameters
        ----------
        dem_path : Path
            Input DEM GeoTIFF.

        Returns
        -------
        Path
            Slope GeoTIFF (float32, degrees).
        """
        slope_path = self.derived_dir / (dem_path.stem + "_slope.tif")
        if slope_path.exists():
            logger.info(f"Slope already computed: {slope_path.name}")
            return slope_path

        logger.info(f"Computing slope from {dem_path.name}")
        # gdaldem is the most reliable way to compute slope
        result = subprocess.run(
            [
                "gdaldem", "slope",
                str(dem_path),
                str(slope_path),
                "-of", "GTiff",
                "-b", "1",
                "-s", "111120",   # scale factor for geographic CRS (degrees → metres)
                "-co", "COMPRESS=LZW",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gdaldem slope failed: {result.stderr}"
            )
        logger.info(f"Slope written: {slope_path}")
        return slope_path

    def compute_hand(
        self,
        dem_path: Path,
        drainage_mask_path: Optional[Path] = None,
    ) -> Path:
        """
        Compute Height Above Nearest Drainage (HAND) using pysheds.

        HAND measures how many metres each pixel sits above the nearest
        stream cell in the flow network. Values < 5m are highly flood-prone.

        Algorithm:
          1. Fill pits in DEM (removes single-cell sinks)
          2. Fill depressions (removes larger sinks that trap flow)
          3. Resolve flats (directs flow across flat areas)
          4. Compute D8 flow direction
          5. Compute flow accumulation
          6. Threshold accumulation → stream network
          7. Compute HAND = elevation - nearest upstream stream elevation

        Parameters
        ----------
        dem_path : Path
            DEM GeoTIFF (EPSG:4326 or projected).
        drainage_mask_path : Path, optional
            Pre-computed stream network raster. If provided, skips steps 4–6.

        Returns
        -------
        Path
            HAND GeoTIFF (float32, metres). Same grid as dem_path.
        """
        try:
            from pysheds.grid import Grid
        except ImportError:
            raise ImportError(
                "pysheds is required for HAND computation. "
                "Install with: pip install pysheds"
            )

        hand_path = self.derived_dir / (dem_path.stem + "_hand.tif")
        if hand_path.exists():
            logger.info(f"HAND already computed: {hand_path.name}")
            return hand_path

        logger.info(f"Computing HAND from {dem_path.name}")

        grid = Grid.from_raster(str(dem_path))
        dem = grid.read_raster(str(dem_path))

        # --- Hydrological conditioning ---
        logger.debug("Filling pits...")
        pit_filled = grid.fill_pits(dem)

        logger.debug("Filling depressions...")
        flooded = grid.fill_depressions(pit_filled)

        logger.debug("Resolving flats...")
        inflated = grid.resolve_flats(flooded)

        # --- Flow network ---
        logger.debug("Computing flow direction (D8)...")
        # ESRI D8 direction map
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        fdir = grid.flowdir(inflated, dirmap=dirmap)

        if drainage_mask_path is None:
            logger.debug("Computing flow accumulation...")
            acc = grid.accumulation(fdir, dirmap=dirmap)
            streams = acc > self._STREAM_ACC_THRESHOLD
        else:
            logger.debug(f"Loading drainage mask from {drainage_mask_path}")
            streams = grid.read_raster(str(drainage_mask_path)).astype(bool)

        # --- HAND ---
        logger.debug("Computing HAND...")
        hand = grid.compute_hand(fdir, dem, streams, inplace=False)

        # Clip negative values (artefacts from pit-filling) and extreme outliers
        hand_arr = np.array(hand)
        hand_arr = np.clip(hand_arr, 0, 100).astype(np.float32)

        # Write output preserving DEM CRS + transform
        with rasterio.open(dem_path) as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", count=1, compress="lzw", nodata=-9999.0)

        with rasterio.open(hand_path, "w", **profile) as dst:
            dst.write(hand_arr[np.newaxis, :, :])

        logger.info(f"HAND written: {hand_path}")
        return hand_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tiles_for_bbox(bbox: list[float]) -> list[tuple[str, str]]:
        """
        Return the list of (lat_code, lon_code) tile identifiers needed
        to cover the bounding box.

        Copernicus DEM GLO-30 tiles are 1° × 1°, named by their SW corner.
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        tiles = []
        for lat in range(math.floor(min_lat), math.ceil(max_lat)):
            for lon in range(math.floor(min_lon), math.ceil(max_lon)):
                lat_hem = "N" if lat >= 0 else "S"
                lon_hem = "E" if lon >= 0 else "W"
                lat_code = f"{lat_hem}{abs(lat):02d}"
                lon_code = f"{lon_hem}{abs(lon):03d}"
                tiles.append((lat_code, lon_code))
        return tiles

    def _download_tile(self, lat_code: str, lon_code: str) -> Optional[Path]:
        """
        Download a single Copernicus DEM tile.

        Returns None if the tile does not exist (ocean / no data).
        """
        url = self._S3_BASE.format(lat_code=lat_code, lon_code=lon_code)
        tile_name = f"DEM_{lat_code}_{lon_code}.tif"
        tile_path = self.tiles_dir / tile_name

        if tile_path.exists() and tile_path.stat().st_size > 0:
            logger.debug(f"DEM tile cached: {tile_name}")
            return tile_path

        logger.info(f"Downloading DEM tile: {tile_name}")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 404:
                logger.debug(f"DEM tile not found (ocean?): {tile_name}")
                return None
            resp.raise_for_status()

            total = int(resp.headers.get("Content-Length", 0))
            with open(tile_path, "wb") as fh, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=tile_name,
                leave=False,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        fh.write(chunk)
                        pbar.update(len(chunk))
            return tile_path

        except requests.RequestException as exc:
            logger.warning(f"Failed to download DEM tile {tile_name}: {exc}")
            return None

    def _merge_and_clip(self, tile_paths: list[Path], bbox: list[float]) -> Path:
        """
        Mosaic multiple DEM tiles into one GeoTIFF, clipped to the AOI bbox.
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        merged_name = (
            f"dem_merged_"
            f"{min_lat:.2f}N_{min_lon:.2f}E_"
            f"{max_lat:.2f}N_{max_lon:.2f}E.tif"
        ).replace("-", "S")

        merged_path = self.merged_dir / merged_name
        if merged_path.exists():
            logger.info(f"Merged DEM already exists: {merged_path.name}")
            return merged_path

        datasets = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic, transform = rasterio.merge.merge(
                datasets,
                bounds=(min_lon, min_lat, max_lon, max_lat),
                resampling=Resampling.bilinear,
            )
            profile = datasets[0].profile.copy()
            profile.update(
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                compress="lzw",
                dtype="float32",
            )
            with rasterio.open(merged_path, "w", **profile) as dst:
                dst.write(mosaic.astype(np.float32))
        finally:
            for ds in datasets:
                ds.close()

        return merged_path


# ---------------------------------------------------------------------------
# JRC Global Surface Water Loader
# ---------------------------------------------------------------------------

class JRCWaterLoader:
    """
    Downloads JRC Global Surface Water seasonality tiles from Google Cloud.

    Tiles are 10° × 10° GeoTIFFs, publicly accessible, no credentials required.

    URL pattern:
      https://storage.googleapis.com/global-surface-water/downloads2021/
        seasonality/seasonality_{lon_code}_{lat_code}_v1_4.tif

    Where lon_code and lat_code are the SW corner of the 10° tile,
    e.g. '10W' and '0N' for the tile covering 10°W–0°W, 0°N–10°N.

    Parameters
    ----------
    output_dir : str | Path
        Directory for JRC tile downloads.
    """

    _GCS_BASE = (
        "https://storage.googleapis.com/global-surface-water/downloads2021"
        "/seasonality/seasonality_{lon_code}_{lat_code}_v1_4.tif"
    )

    def __init__(self, output_dir: str | Path = "data/raw/jrc") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, bbox: list[float], year: int = 2021) -> Path:
        """
        Download and clip the JRC seasonality raster for the given AOI.

        The seasonality band encodes how many months per year (0–12) a pixel
        was observed as surface water over the 1984–2021 Landsat record.

        Parameters
        ----------
        bbox : list[float]
            [min_lat, min_lon, max_lat, max_lon]
        year : int
            Dataset version year. 2021 is the latest available.

        Returns
        -------
        Path
            Clipped JRC seasonality GeoTIFF (uint8, months 0–12).
        """
        tiles = self._tiles_for_bbox(bbox)
        tile_paths: list[Path] = []

        for lat_code, lon_code in tiles:
            tile_path = self._download_tile(lat_code, lon_code)
            if tile_path:
                tile_paths.append(tile_path)

        if not tile_paths:
            raise RuntimeError(
                "No JRC tiles downloaded — check bbox or network connectivity"
            )

        output_path = self._merge_and_clip(tile_paths, bbox)
        logger.info(f"JRC seasonality ready: {output_path}")
        return output_path

    def get_permanent_water_mask(
        self, jrc_path: Path, threshold: int = 10
    ) -> Path:
        """
        Binarize the JRC seasonality raster into a permanent water mask.

        Pixels with water presence ≥ threshold months/year are classified
        as permanent water. These pixels should be excluded from flood
        change-detection (they are baseline-wet, not event-flooded).

        Parameters
        ----------
        jrc_path : Path
            JRC seasonality GeoTIFF (values 0–12).
        threshold : int
            Minimum months/year to classify as permanent water.
            Default 10 means water present ≥10 months/year.

        Returns
        -------
        Path
            Binary mask GeoTIFF (uint8): 1 = permanent water, 0 = not.
        """
        mask_path = jrc_path.parent / (jrc_path.stem + f"_permanent_water_t{threshold}.tif")
        if mask_path.exists():
            logger.info(f"Permanent water mask cached: {mask_path.name}")
            return mask_path

        logger.info(
            f"Creating permanent water mask (threshold={threshold} months) "
            f"from {jrc_path.name}"
        )

        with rasterio.open(jrc_path) as src:
            seasonality = src.read(1)
            profile = src.profile.copy()

        mask = (seasonality >= threshold).astype(np.uint8)
        water_pct = mask.mean() * 100

        profile.update(dtype="uint8", count=1, compress="lzw", nodata=0)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask[np.newaxis, :, :])

        logger.info(
            f"Permanent water mask written: {mask_path.name} "
            f"({water_pct:.2f}% permanent water)"
        )
        return mask_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tiles_for_bbox(bbox: list[float]) -> list[tuple[str, str]]:
        """
        Return (lat_code, lon_code) for all 10° JRC tiles covering the bbox.

        JRC tiles are 10° × 10°, named by SW corner rounded to the
        nearest 10° boundary.
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        tiles = []

        # Floor to nearest 10° grid
        lat_start = int(math.floor(min_lat / 10) * 10)
        lon_start = int(math.floor(min_lon / 10) * 10)
        lat_end = int(math.ceil(max_lat / 10) * 10)
        lon_end = int(math.ceil(max_lon / 10) * 10)

        for lat in range(lat_start, lat_end, 10):
            for lon in range(lon_start, lon_end, 10):
                lat_code = f"{abs(lat)}{'N' if lat >= 0 else 'S'}"
                lon_code = f"{abs(lon)}{'E' if lon >= 0 else 'W'}"
                tiles.append((lat_code, lon_code))
        return tiles

    def _download_tile(self, lat_code: str, lon_code: str) -> Optional[Path]:
        """
        Download a single JRC seasonality tile.

        Returns None if the tile does not exist (e.g. open ocean tile).
        """
        url = self._GCS_BASE.format(lat_code=lat_code, lon_code=lon_code)
        tile_name = f"jrc_seasonality_{lon_code}_{lat_code}.tif"
        tile_path = self.output_dir / tile_name

        if tile_path.exists() and tile_path.stat().st_size > 0:
            logger.debug(f"JRC tile cached: {tile_name}")
            return tile_path

        logger.info(f"Downloading JRC tile: {tile_name}")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 404:
                logger.debug(f"JRC tile not found: {tile_name}")
                return None
            resp.raise_for_status()

            total = int(resp.headers.get("Content-Length", 0))
            with open(tile_path, "wb") as fh, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=tile_name,
                leave=False,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        fh.write(chunk)
                        pbar.update(len(chunk))
            return tile_path

        except requests.RequestException as exc:
            logger.warning(f"Failed to download JRC tile {tile_name}: {exc}")
            return None

    def _merge_and_clip(self, tile_paths: list[Path], bbox: list[float]) -> Path:
        """
        Mosaic JRC tiles and clip to the AOI bounding box.
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        out_name = (
            f"jrc_seasonality_"
            f"{min_lat:.2f}N_{min_lon:.2f}E_"
            f"{max_lat:.2f}N_{max_lon:.2f}E.tif"
        ).replace("-", "S")
        out_path = self.output_dir / out_name

        if out_path.exists():
            logger.info(f"Merged JRC tile already exists: {out_path.name}")
            return out_path

        datasets = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic, transform = rasterio.merge.merge(
                datasets,
                bounds=(min_lon, min_lat, max_lon, max_lat),
                resampling=Resampling.nearest,  # categorical data
            )
            profile = datasets[0].profile.copy()
            profile.update(
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                compress="lzw",
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic)
        finally:
            for ds in datasets:
                ds.close()

        return out_path
