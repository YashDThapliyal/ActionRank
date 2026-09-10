import dataclasses
from pathlib import Path

import pytest

from config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config_has_expected_sections():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.model.backbone == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.data.eval_fraction == 0.15
    assert cfg.verbalize.full_history_steps in (2, 3)
    assert 8 <= cfg.tier2.lora_rank <= 16


def test_config_is_immutable():
    cfg = load_config(ROOT / "config.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model.backbone = "x"  # type: ignore[misc]


def test_unknown_key_is_rejected(tmp_path):
    good = (ROOT / "config.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(good.replace("split_seed:", "typo_seed:"))
    with pytest.raises(ValueError, match="data"):
        load_config(bad)


def test_device_env_override(monkeypatch):
    monkeypatch.setenv("ACTIONRANK_DEVICE", "cuda")
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.model.device == "cuda"


def test_config_path_env_override(monkeypatch, tmp_path):
    alt = tmp_path / "alt.yaml"
    alt.write_text((ROOT / "config.yaml").read_text().replace("pooling: mean", "pooling: last"))
    monkeypatch.setenv("ACTIONRANK_CONFIG", str(alt))
    assert load_config().model.pooling == "last"
