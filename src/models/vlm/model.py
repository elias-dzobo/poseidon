"""
LFM2.5 VL-450M wrapper for flood report generation.

Wraps Liquid AI's LFM2.5 VL-450M vision-language model with:
  - 4-bit quantisation (BitsAndBytes) for memory-efficient inference
  - LoRA adapter loading for fine-tuned weights
  - Structured prompt construction with SAR composite + metadata
  - JSON-mode output parsing into typed FloodReport objects
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Report schema (typed output from VLM)
# ------------------------------------------------------------------

@dataclass
class AffectedArea:
    name: str
    severity: str           # low | medium | high | critical
    area_km2: float


@dataclass
class HydrologicalIndicators:
    odaw_river_status: str
    korle_lagoon_status: str
    permanent_water_expansion_pct: float


@dataclass
class FloodExtent:
    total_area_km2: float
    pct_of_aoi_flooded: float
    change_from_baseline_km2: float


@dataclass
class FloodReport:
    event_id: str
    acquisition_date: str
    region: dict
    severity: str                               # low | medium | high | critical
    confidence_score: float
    flood_extent: FloodExtent
    affected_areas: list[AffectedArea]
    hydrological_indicators: HydrologicalIndicators
    recommendations: list[str]
    data_sources: list[str]
    model_versions: dict
    raw_json: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Model wrapper
# ------------------------------------------------------------------

class LFMFloodReporter:
    """
    Wraps LFM2.5 VL-450M for flood situation report generation.

    Accepts a SAR false-color composite (with optional flood mask overlay)
    and structured metadata, returns a typed FloodReport.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path. Default: 'liquid/lfm2.5-vl-450m'.
    lora_path : str | Path, optional
        Path to LoRA adapter weights directory. If None, runs base model.
    load_in_4bit : bool
        Enable 4-bit quantisation via BitsAndBytes. Default: True.
    device_map : str
        HuggingFace device_map. Default: 'auto'.
    system_prompt : str, optional
        System instruction injected into every prompt.
    """

    DEFAULT_MODEL = "liquid/lfm2.5-vl-450m"

    DEFAULT_SYSTEM_PROMPT = (
        "You are a flood analysis expert system. You will be given a satellite "
        "image composite of an area in Accra, Ghana, overlaid with a flood "
        "detection mask. Analyze the image and generate a structured JSON flood "
        "assessment report. Respond ONLY with valid JSON matching the schema. "
        "Do not include any prose outside the JSON object."
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lora_path: Optional[str | Path] = None,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.lora_path = Path(lora_path) if lora_path else None
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """
        Load model and processor into memory.

        Lazy — call explicitly before first inference or call generate()
        which will auto-load.
        """
        # TODO:
        # 1. Build BitsAndBytesConfig if load_in_4bit
        # 2. Load AutoModelForVision2Seq + AutoProcessor from model_name
        # 3. If lora_path, load with PeftModel.from_pretrained
        raise NotImplementedError

    def generate(
        self,
        composite_path: Path,
        metadata: dict,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> FloodReport:
        """
        Generate a flood report for a given SAR composite image.

        Parameters
        ----------
        composite_path : Path
            Path to the false-color composite PNG (with flood mask overlay).
        metadata : dict
            Contextual metadata injected into the prompt:
              - acquisition_date: str
              - region: dict (country, city, bbox)
              - season: str
              - data_sources: list[str]
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature. Low values → deterministic structured output.

        Returns
        -------
        FloodReport
            Parsed and validated flood report.
        """
        if not self._loaded:
            self.load()

        prompt = self._build_prompt(metadata)
        raw_output = self._run_inference(composite_path, prompt, max_new_tokens, temperature)
        report = self._parse_output(raw_output, metadata)
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, metadata: dict) -> str:
        """
        Construct the user-turn prompt with injected metadata.

        The system prompt handles general instructions; the user prompt
        provides event-specific context.
        """
        return (
            f"Acquisition date: {metadata.get('acquisition_date', 'unknown')}\n"
            f"Region: {metadata.get('region', {})}\n"
            f"Season: {metadata.get('season', 'unknown')}\n"
            f"Data sources: {', '.join(metadata.get('data_sources', []))}\n\n"
            "Analyze the satellite image and generate the flood assessment JSON report."
        )

    def _run_inference(
        self,
        composite_path: Path,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """
        Run model inference with the image + text prompt.

        Returns the raw generated string.
        """
        # TODO:
        # 1. Load composite_path as PIL Image
        # 2. Build chat-format messages: [system, user(text+image)]
        # 3. Apply chat template via processor
        # 4. Run model.generate(), decode output tokens
        raise NotImplementedError

    def _parse_output(self, raw_output: str, metadata: dict) -> FloodReport:
        """
        Parse the VLM's JSON string output into a typed FloodReport.

        Falls back gracefully if JSON is malformed — logs a warning and
        returns a partial report with raw_json populated for inspection.
        """
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("VLM output was not valid JSON — attempting extraction")
            data = self._extract_json_from_text(raw_output)

        return self._dict_to_report(data, metadata)

    @staticmethod
    def _extract_json_from_text(text: str) -> dict:
        """
        Best-effort JSON extraction from text that may contain surrounding prose.
        Looks for the outermost { } block.
        """
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.error("Could not extract JSON from VLM output")
        return {}

    @staticmethod
    def _dict_to_report(data: dict, metadata: dict) -> FloodReport:
        """Construct a FloodReport dataclass from a parsed JSON dict."""
        # TODO: map JSON keys to dataclass fields, handle missing keys with defaults
        raise NotImplementedError
