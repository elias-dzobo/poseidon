"""
LoRA fine-tuning script for LFM2.5 VL-450M.

Uses HuggingFace PEFT + TRL SFTTrainer for supervised fine-tuning on
flood report generation. 4-bit quantisation (QLoRA) keeps GPU memory
under 16 GB, making this runnable on a single A100 or RTX 4090.

Training recipe:
  - Base model: liquid/lfm2.5-vl-450m
  - Quantisation: 4-bit NF4 (BitsAndBytes)
  - LoRA: rank 16, alpha 32, targets: q_proj, k_proj, v_proj, o_proj
  - Optimizer: AdamW (paged, for memory efficiency)
  - Scheduler: cosine with warmup
  - Loss: causal LM (cross-entropy on assistant response tokens only)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_bnb_config(config: dict):
    """
    Build a BitsAndBytesConfig for 4-bit QLoRA.

    Parameters
    ----------
    config : dict
        vlm.quantization section from model_config.yaml.

    Returns
    -------
    BitsAndBytesConfig
    """
    # TODO: import BitsAndBytesConfig from transformers
    # return BitsAndBytesConfig(
    #     load_in_4bit=config["load_in_4bit"],
    #     bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
    #     bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
    #     bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
    # )
    raise NotImplementedError


def build_lora_config(config: dict):
    """
    Build a LoraConfig for PEFT.

    Parameters
    ----------
    config : dict
        vlm.lora section from model_config.yaml.

    Returns
    -------
    LoraConfig
    """
    # TODO: import LoraConfig from peft
    # return LoraConfig(
    #     r=config["r"],
    #     lora_alpha=config["alpha"],
    #     lora_dropout=config["dropout"],
    #     target_modules=config["target_modules"],
    #     bias=config["bias"],
    #     task_type="CAUSAL_LM",
    # )
    raise NotImplementedError


def finetune(
    config: dict,
    data_dir: str | Path,
    output_dir: str | Path = "models/weights/vlm",
    resume_from: Optional[str | Path] = None,
) -> Path:
    """
    Fine-tune LFM2.5 VL-450M with LoRA on the flood report dataset.

    Parameters
    ----------
    config : dict
        Full vlm section from model_config.yaml.
    data_dir : str | Path
        Path to training_pairs directory.
    output_dir : str | Path
        Directory to save LoRA adapter weights and tokenizer.
    resume_from : str | Path, optional
        Resume training from a checkpoint directory.

    Returns
    -------
    Path
        Path to the saved LoRA adapter directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model with 4-bit quantisation...")
    # TODO: 1. build_bnb_config, load AutoModelForVision2Seq + AutoProcessor
    # TODO: 2. enable_input_require_grads on model (required for gradient checkpointing)
    # TODO: 3. build_lora_config, wrap with get_peft_model
    # TODO: 4. build FloodReportDataset for train and val splits
    # TODO: 5. configure TrainingArguments (from model_config training section)
    # TODO: 6. instantiate SFTTrainer, call .train()
    # TODO: 7. save_pretrained to output_dir

    raise NotImplementedError
