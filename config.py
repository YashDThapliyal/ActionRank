"""Typed, immutable configuration loaded from config.yaml."""
from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    hf_repo: str
    subset: str
    raw_dir: str
    processed_dir: str
    eval_fraction: float
    split_seed: int
    require_win: bool
    max_files: int | None


@dataclass(frozen=True)
class VerbalizeConfig:
    full_history_steps: int
    observation_chars: int
    args_chars: int
    description_chars: int


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    device: str
    dtype: str
    pooling: str
    max_prompt_tokens: int
    head_hidden: int
    score_temperature: float
    tool_init: str  # text | random


@dataclass(frozen=True)
class Tier1Config:
    head: str  # table | span
    cache_dir: str
    cache_batch_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    checkpoint: str
    span_checkpoint: str


@dataclass(frozen=True)
class Tier2Config:
    head: str  # table | span
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lr: float
    epochs: int
    batch_size: int
    grad_accum: int
    max_train_examples: int
    checkpoint_dir: str
    train_seed: int | None = None  # shuffle order + LoRA init; None falls back to data.split_seed


@dataclass(frozen=True)
class BaselineConfig:
    max_new_tokens: int
    num_beams: int
    sft_epochs: int
    sft_batch_size: int
    sft_grad_accum: int
    sft_checkpoint_dir: str


@dataclass(frozen=True)
class EvalConfig:
    max_eval_examples: int
    results_dir: str
    latency_warmup: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    verbalize: VerbalizeConfig
    model: ModelConfig
    tier1: Tier1Config
    tier2: Tier2Config
    baseline: BaselineConfig
    eval: EvalConfig


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "verbalize": VerbalizeConfig,
    "model": ModelConfig,
    "tier1": Tier1Config,
    "tier2": Tier2Config,
    "baseline": BaselineConfig,
    "eval": EvalConfig,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _build_section(cls: type, raw: Any, section: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"config section '{section}' must be a mapping")
    expected = {f.name for f in fields(cls)}
    required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
    missing = required - raw.keys()
    extra = raw.keys() - expected
    if missing or extra:
        raise ValueError(
            f"config section '{section}': missing={sorted(missing)} extra={sorted(extra)}"
        )
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate config.yaml into a frozen Config tree.

    The path defaults to config.yaml next to this file; ACTIONRANK_CONFIG overrides it (used to evaluate
    variant runs, e.g. last-token pooling, without touching the main config)."""
    if path is None:
        path = os.environ.get("ACTIONRANK_CONFIG") or DEFAULT_CONFIG_PATH
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    missing = _SECTIONS.keys() - raw.keys()
    extra = raw.keys() - _SECTIONS.keys()
    if missing or extra:
        raise ValueError(f"config sections: missing={sorted(missing)} extra={sorted(extra)}")
    cfg = Config(**{name: _build_section(cls, raw[name], name) for name, cls in _SECTIONS.items()})
    device = os.environ.get("ACTIONRANK_DEVICE")  # e.g. cuda on Colab, without editing config.yaml
    if device:
        cfg = replace(cfg, model=replace(cfg.model, device=device))
    return cfg
