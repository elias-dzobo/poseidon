"""
Poseidon data ingestion layer.

Exports the three main ingestion classes for use by the pipeline.
"""

from src.ingestion.sentinel1 import Sentinel1Downloader
from src.ingestion.sentinel2 import Sentinel2Downloader
from src.ingestion.auxiliary import DEMLoader, JRCWaterLoader

__all__ = [
    "Sentinel1Downloader",
    "Sentinel2Downloader",
    "DEMLoader",
    "JRCWaterLoader",
]
