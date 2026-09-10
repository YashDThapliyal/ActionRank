"""Evaluate ActionRank vs. the generation baseline: accuracy, hallucination rate, latency."""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm

from config import Config, load_config
from data import FINISH, Catalog, Example, load_dataset
from model import ActionRankModel, build_candidate_mask, load_backbone, load_head, synchronize
from verbalize import build_prompt

log = logging.getLogger(__name__)
TOPK = 5


@dataclass(frozen=True)
class Metrics:
    name: str
    n: int
    top1: float
    top5: float
    hallucination_rate: float
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    top1_no_finish: float
    n_no_finish: int


def _rate(hits: Sequence[bool]) -> float:
    return sum(hits) / len(hits) if hits else 0.0


def compute_metrics(name: str, labels: Sequence[str], top1: Sequence[str], topk: Sequence[tuple[str, ...]],
                    valid: Sequence[bool], latencies_s: Sequence[float] | None, finish_name: str = FINISH) -> Metrics:
    n = len(labels)
    lengths = {len(top1), len(topk), len(valid), n} | ({len(latencies_s)} if latencies_s is not None else set())
    if lengths != {n}:
        raise ValueError(f"length mismatch: labels={n} top1={len(top1)} topk={len(topk)} valid={len(valid)}")
    top1_hits = [p == y for p, y in zip(top1, labels)]
    no_finish = [hit for hit, y in zip(top1_hits, labels) if y != finish_name]
    ms = [1000.0 * s for s in latencies_s] if latencies_s is not None else None
    return Metrics(
        name=name, n=n, top1=_rate(top1_hits),
        top5=_rate([y in ks[:TOPK] for ks, y in zip(topk, labels)]),
        hallucination_rate=_rate([not ok for ok in valid]),
        latency_mean_ms=statistics.fmean(ms) if ms else None,
        latency_p50_ms=statistics.median(ms) if ms else None,
        top1_no_finish=_rate(no_finish), n_no_finish=len(no_finish),
    )


@torch.no_grad()
def evaluate_actionrank(model: ActionRankModel, examples: Sequence[Example], catalog: Catalog, cfg: Config,
                        name: str, warmup: int = 0) -> Metrics:
    """One decision at a time (batch size 1) so latency is comparable with the baseline."""
    model.eval()
    names = catalog.names
    top1, topk, valid, latencies = [], [], [], []
    for ex in tqdm(list(examples[:warmup]) + list(examples), desc=name, leave=False):
        synchronize(model.device)
        started = time.perf_counter()
        prompt = build_prompt(ex, catalog, cfg.verbalize)
        mask = build_candidate_mask([ex], catalog).to(model.device)
        ranked = model.rank([prompt], mask, TOPK)[0]
        synchronize(model.device)
        elapsed = time.perf_counter() - started
        if warmup > 0:
            warmup -= 1
            continue
        picks = tuple(names[i] for i in ranked)
        if not picks:
            raise RuntimeError(f"no candidate received a finite score for example {ex.query_id}")
        top1.append(picks[0]); topk.append(picks); valid.append(picks[0] in ex.candidates); latencies.append(elapsed)
    return compute_metrics(name, [ex.label for ex in examples], top1, topk, valid, latencies)


def evaluate_baseline(examples: Sequence[Example], catalog: Catalog, cfg: Config, tokenizer, backbone,
                      warmup: int = 0) -> Metrics:
    from baseline import run_baseline

    preds = run_baseline(examples, catalog, cfg, tokenizer, backbone, warmup=warmup)
    return compute_metrics("baseline-generation", [ex.label for ex in examples], [p.top1 for p in preds],
                           [p.topk for p in preds], [p.in_candidates for p in preds], [p.latency_s for p in preds])


def reference_rows(train: Sequence[Example], evaluation: Sequence[Example], seed: int = 0) -> list[Metrics]:
    """Cheap reference points: uniform random candidate, and the globally most frequent label among candidates."""
    rng = random.Random(seed)
    labels = [ex.label for ex in evaluation]
    rand = [rng.choice(ex.candidates) for ex in evaluation]
    rand_k = [tuple(rng.sample(ex.candidates, min(TOPK, len(ex.candidates)))) for ex in evaluation]
    freq = Counter(ex.label for ex in train)
    ordered = [tuple(sorted(ex.candidates, key=lambda c: -freq[c])) for ex in evaluation]
    return [
        compute_metrics("random-candidate", labels, rand, rand_k, [True] * len(labels), None),
        compute_metrics("most-frequent-candidate", labels, [o[0] for o in ordered], ordered, [True] * len(labels), None),
    ]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def render_table(rows: Sequence[Metrics]) -> str:
    header = "| system | n | top-1 | top-5 | hallucination | latency mean (ms) | latency p50 (ms) | top-1 excl. Finish |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    body = [f"| {m.name} | {m.n} | {_pct(m.top1)} | {_pct(m.top5)} | {_pct(m.hallucination_rate)} | "
            f"{_ms(m.latency_mean_ms)} | {_ms(m.latency_p50_ms)} | {_pct(m.top1_no_finish)} (n={m.n_no_finish}) |"
            for m in rows]
    return "\n".join([header, sep, *body]) + "\n"


def write_results(rows: Sequence[Metrics], results_dir: Path, note: str = "") -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps([asdict(m) for m in rows], indent=1))
    md = results_dir / "results.md"
    md.write_text(f"# Results\n\n{note}\n\n{render_table(rows)}")
    return md


def _load_tier1(cfg: Config, tokenizer, backbone) -> ActionRankModel:
    head = load_head(Path(cfg.tier1.checkpoint)).to(next(backbone.parameters()).device)
    return ActionRankModel(tokenizer, backbone, head, cfg.model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ActionRank systems and the generation baseline")
    parser.add_argument("--systems", default="tier1,baseline", help="comma list of: tier1, tier2, baseline")
    parser.add_argument("--limit", type=int, default=None, help="override eval.max_eval_examples")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - {"tier1", "tier2", "baseline"}
    if unknown:
        raise SystemExit(f"unknown systems: {sorted(unknown)}")
    ds = load_dataset(cfg)
    limit = args.limit or cfg.eval.max_eval_examples
    evaluation = ds.eval[:limit]
    rows = reference_rows(ds.train, evaluation, seed=cfg.data.split_seed)
    tokenizer, backbone = load_backbone(cfg.model)
    warm = cfg.eval.latency_warmup
    if "tier1" in systems:
        rows.append(evaluate_actionrank(_load_tier1(cfg, tokenizer, backbone), evaluation, ds.catalog, cfg, "actionrank-tier1", warm))
    if "baseline" in systems:
        rows.append(evaluate_baseline(evaluation, ds.catalog, cfg, tokenizer, backbone, warm))
    if "tier2" in systems:
        from train_tier2 import load_tier2

        rows.append(evaluate_actionrank(load_tier2(cfg, ds.catalog), evaluation, ds.catalog, cfg, "actionrank-tier2", warm))
    note = (f"Eval subset: first {len(evaluation)} of {len(ds.eval)} held-out step examples "
            f"({cfg.data.eval_fraction:.0%} of trajectories). Catalog size {len(ds.catalog)}. "
            f"Backbone {cfg.model.backbone} ({cfg.model.dtype}) on {cfg.model.device}.")
    path = write_results(rows, Path(cfg.eval.results_dir), note)
    print(render_table(rows))
    print(f"written {path}")


if __name__ == "__main__":
    main()
