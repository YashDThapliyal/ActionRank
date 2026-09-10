"""Generation baseline: same backbone, chat-prompted to emit the next tool name autoregressively."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from tqdm import tqdm

from config import Config
from data import Catalog, Example
from model import load_backbone, synchronize
from verbalize import build_baseline_messages

log = logging.getLogger(__name__)

_PREFIX_RE = re.compile(r"^(?:action|tool|next tool|answer)\s*:\s*", re.IGNORECASE)
_STRIP_CHARS = "`'\"*.,:;!? \t"


@dataclass(frozen=True)
class BaselinePrediction:
    top1: str
    topk: tuple[str, ...]
    latency_s: float
    in_candidates: bool
    in_catalog: bool
    raw: str


def normalize_tool_name(text: str) -> str:
    """Reduce free-form generation to a single tool-name token."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = _PREFIX_RE.sub("", lines[0]).strip(_STRIP_CHARS)
    token = re.split(r"[\s(]", first, maxsplit=1)[0]
    return token.strip(_STRIP_CHARS)


def dedupe_keep_order(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    return tuple(x for x in items if not (x in seen or seen.add(x)))


def _chat_inputs(tokenizer, messages: list[dict[str, str]], device: torch.device, max_tokens: int) -> dict[str, torch.Tensor]:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    return {k: v.to(device) for k, v in batch.items()}


def _decode_new(tokenizer, sequences: torch.Tensor, prompt_len: int) -> list[str]:
    return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in sequences]


@torch.no_grad()
def generate_top1(tokenizer, model, inputs: dict[str, torch.Tensor], cfg: Config) -> tuple[str, float]:
    """Greedy decode; returns (raw text, wall-clock seconds incl. prefill + decode)."""
    device = inputs["input_ids"].device
    synchronize(device)
    started = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=cfg.baseline.max_new_tokens, do_sample=False,
                         num_beams=1, pad_token_id=tokenizer.pad_token_id)
    synchronize(device)
    return _decode_new(tokenizer, out, inputs["input_ids"].shape[1])[0], time.perf_counter() - started


@torch.no_grad()
def generate_topk(tokenizer, model, inputs: dict[str, torch.Tensor], cfg: Config) -> tuple[str, ...]:
    """Beam search with num_beams return sequences, normalized and de-duplicated."""
    beams = cfg.baseline.num_beams
    out = model.generate(**inputs, max_new_tokens=cfg.baseline.max_new_tokens, do_sample=False,
                         num_beams=beams, num_return_sequences=beams, pad_token_id=tokenizer.pad_token_id,
                         early_stopping=True)
    return dedupe_keep_order(normalize_tool_name(t) for t in _decode_new(tokenizer, out, inputs["input_ids"].shape[1]))


def predict_one(tokenizer, model, example: Example, catalog: Catalog, cfg: Config) -> BaselinePrediction:
    device = next(model.parameters()).device
    inputs = _chat_inputs(tokenizer, build_baseline_messages(example, catalog, cfg.verbalize), device,
                          cfg.model.max_prompt_tokens)
    raw, latency = generate_top1(tokenizer, model, inputs, cfg)
    top1 = normalize_tool_name(raw)
    topk = dedupe_keep_order((top1, *generate_topk(tokenizer, model, inputs, cfg))) if cfg.baseline.num_beams > 1 else (top1,)
    return BaselinePrediction(top1=top1, topk=topk, latency_s=latency, in_candidates=top1 in example.candidates,
                              in_catalog=top1 in catalog, raw=raw)


def run_baseline(examples: Sequence[Example], catalog: Catalog, cfg: Config, tokenizer=None, model=None,
                 warmup: int = 0) -> tuple[BaselinePrediction, ...]:
    """Run the generation baseline over examples; the first `warmup` decisions are re-run untimed."""
    if tokenizer is None or model is None:
        tokenizer, model = load_backbone(cfg.model)
    for ex in examples[:warmup]:
        predict_one(tokenizer, model, ex, catalog, cfg)
    return tuple(predict_one(tokenizer, model, ex, catalog, cfg) for ex in tqdm(examples, desc="baseline", leave=False))
