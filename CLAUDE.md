# Poseidon — Flood Detection System

## What this project does
End-to-end satellite-based flood detection for Ghana (starting with Accra).
Ingests Sentinel-1 SAR + Sentinel-2 optical data, runs a U-Net segmentation
model to produce pixel-level flood masks, then passes the result to a
fine-tuned LFM2.5 VL-450M vision-language model to generate structured
flood situation reports.

## Architecture
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
- **U-Net first**: pixel-wise flood segmentation on 7-channel feature stacks
  (VV, VH, VV/VH ratio, NDWI, MNDWI, HAND, slope)
- **LFM2.5 VL second**: scene-level reasoning → structured JSON flood report
- **SAR false-color composite**: VV→R, VH→G, VV/VH→B bridges SAR to VLM vision encoder
- **LoRA fine-tuning**: rank-16 LoRA on q/k/v/o projections with 4-bit QLoRA
- **Ghana-specific**: Odaw River basin, Korle Lagoon, seasonal calendar (Apr–Jun / Sep–Nov)

## Environment setup
```bash
cp .env.example .env
# fill in CDSE_USERNAME and CDSE_PASSWORD

pip install -r requirements.txt
# GDAL and SNAP must be installed separately at the system level
```

## Running the pipeline
```bash
python -m src.pipeline.run_detection \
    --start 2024-06-10 \
    --end   2024-06-15 \
    --config config/settings.yaml \
    --unet-weights models/weights/unet/best.pt \
    --vlm-lora models/weights/vlm/lora-adapter
```

## Training
- U-Net: `notebooks/03_unet_training.ipynb` — uses Sen1Floods11 + Ghana labels
- VLM:   `notebooks/04_vlm_finetuning.ipynb` — LoRA on flood report pairs

## TODO (stubs to implement)
All `raise NotImplementedError` blocks are documented stubs with
implementation guidance in their docstrings. Build order:
1. `src/ingestion/` — CDSE API auth + download
2. `src/preprocessing/sar.py` — SNAP/pyroSAR chain
3. `src/preprocessing/optical.py` — rasterio band extraction + cloud mask
4. `src/preprocessing/features.py` — alignment + stacking + normalisation
5. `src/models/unet/dataset.py` — Sen1Floods11 loader
6. `src/models/unet/predict.py` — tiled inference
7. `src/models/vlm/model.py` — LFM2.5 VL load + inference
8. `src/models/vlm/finetune.py` — QLoRA training
