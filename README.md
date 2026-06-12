# Poseidon — Satellite Flood Detection System

End-to-end satellite-based flood detection for Ghana (starting with Accra). Poseidon ingests
Sentinel-1 SAR + Sentinel-2 optical data, runs a U-Net segmentation model to produce
pixel-level flood masks, then passes the result to a fine-tuned **LFM2.5 VL-450M**
vision-language model to generate structured flood situation reports.

## Pipeline

```
Sentinel-1 + Sentinel-2 + DEM → Preprocessing → Feature Stack
→ U-Net (flood mask) → LFM2.5 VL (flood report)
→ GeoTIFF + GeoJSON + JSON/PDF report
```

## Project structure

```
config/           — settings.yaml (AOI, thresholds), model_config.yaml
data/             — raw/, processed/, features/, labels/, training_pairs/
src/
  ingestion/      — Sentinel-1/2 downloaders + auxiliary data (DEM, JRC)
  preprocessing/  — SAR chain, optical cloud mask, feature stacking, composites
  models/
    unet/         — architecture, dataset, train, predict
    vlm/          — LFM2.5 wrapper, dataset, LoRA finetune, report generation
  pipeline/       — end-to-end orchestrator (run_detection.py)
  output/         — map writer (GeoTIFF/GeoJSON), report writer (JSON/text/PDF)
notebooks/        — exploration and training notebooks
tests/
```

## Key design decisions

- **U-Net first** — pixel-wise flood segmentation on 7-channel feature stacks
  (VV, VH, VV/VH ratio, NDWI, MNDWI, HAND, slope)
- **LFM2.5 VL second** — scene-level reasoning → structured JSON flood report
- **SAR false-color composite** — VV→R, VH→G, VV/VH→B, bridging SAR to the VLM vision encoder
- **LoRA fine-tuning** — rank-16 LoRA on q/k/v/o projections with 4-bit QLoRA
- **Ghana-specific** — Odaw River basin, Korle Lagoon, seasonal calendar (Apr–Jun / Sep–Nov)

## Quickstart

```bash
pip install -r requirements.txt
python -m src.pipeline.run_detection --config config/settings.yaml
```

> Raw/processed satellite data and model weights are excluded from the repo (see `.gitignore`).
