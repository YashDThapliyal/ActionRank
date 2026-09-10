"""Overnight driver for the span-pooling head: latency probe -> full re-encode (with early bail-out) ->
head training -> three-system eval -> prediction analysis. Everything is logged; any guard failure raises."""
from __future__ import annotations

import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("overnight")

REFERENCE_MS = 232.0        # measured by scripts/prototypes/span_breakdown.py (CPU-offload span path)
MAX_LATENCY_RATIO = 1.5     # abort if a probe's mean latency exceeds 1.5x the reference
PROBE_EXAMPLES = 30
PROBE_WARMUP = 3
EXPECTED_ENCODE_S_PER_EXAMPLE = 0.45   # batch-16 throughput measured for the table-head cache (same prefill)
MAX_ENCODE_RATIO = 3.0                 # abort if sustained throughput is 3x slower than expected
ENCODE_CHECK_AFTER = 320               # examples (20 batches) before the first throughput check


def check_latency(samples_ms: Sequence[float], reference_ms: float, max_ratio: float, label: str) -> float:
    """Mean of the samples; raises RuntimeError when it exceeds reference_ms * max_ratio."""
    mean = statistics.mean(samples_ms)
    if mean > reference_ms * max_ratio:
        raise RuntimeError(f"{label}: mean latency {mean:.0f} ms/decision exceeds {max_ratio}x the "
                           f"{reference_ms:.0f} ms reference; aborting before burning the run")
    log.info("%s: mean latency %.0f ms/decision (reference %.0f ms, limit %.0f ms) OK", label, mean, reference_ms, reference_ms * max_ratio)
    return mean


def probe_latency(model, examples, catalog, cfg) -> list[float]:
    """Single-decision latencies (ms) through the exact eval path (rank_example, CPU-offload span pooling)."""
    from model import synchronize

    if getattr(model, "span_pool_device", None) != "cpu":
        raise RuntimeError("span model is not using the CPU-offload pooling path")
    samples = []
    for i, ex in enumerate(list(examples[:PROBE_WARMUP]) + list(examples[:PROBE_EXAMPLES])):
        synchronize(model.device)
        t0 = time.perf_counter()
        model.rank_example(ex, catalog, 5, cfg.verbalize)
        synchronize(model.device)
        if i >= PROBE_WARMUP:
            samples.append(1000 * (time.perf_counter() - t0))
    return samples


def make_progress_guard():
    def guard(n_done: int, seconds: float) -> None:
        if n_done >= ENCODE_CHECK_AFTER:
            per_example = seconds / n_done
            if per_example > EXPECTED_ENCODE_S_PER_EXAMPLE * MAX_ENCODE_RATIO:
                raise RuntimeError(f"encode throughput {per_example:.2f} s/example after {n_done} examples exceeds "
                                   f"{MAX_ENCODE_RATIO}x the expected {EXPECTED_ENCODE_S_PER_EXAMPLE} s/example; aborting")
            if n_done % 1600 < 16:
                log.info("encode: %d examples, %.2f s/example", n_done, per_example)
    return guard


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from config import load_config
    from data import load_dataset
    from model import SpanActionRankModel, SpanScoringHead, load_backbone, load_span_head
    from train_tier1 import run_span_tier1

    cfg = load_config()
    if cfg.tier1.head != "span":
        raise RuntimeError("config.yaml tier1.head must be 'span' for this driver")
    ds = load_dataset(cfg)
    tokenizer, backbone = load_backbone(cfg.model)
    hidden = backbone.config.hidden_size
    probe_set = ds.eval[:cfg.eval.max_eval_examples]

    # 1. latency probe with an untrained span head through the real eval path
    untrained = SpanActionRankModel(tokenizer, backbone, SpanScoringHead(hidden, cfg.model.head_hidden, cfg.model.score_temperature), cfg.model, ds.catalog)
    check_latency(probe_latency(untrained, probe_set, ds.catalog, cfg), REFERENCE_MS, MAX_LATENCY_RATIO, "pre-encode probe")

    # 2. full re-encode with throughput guard, then head training + held-out eval on the cache
    summary = run_span_tier1(cfg, ds, tokenizer, backbone, refresh=False, progress=make_progress_guard())
    log.info("span tier1 summary: %s", json.dumps({k: v for k, v in summary.items() if k != "history"}))

    # 3. latency probe again with the trained head (the artifact the eval will use)
    trained = SpanActionRankModel(tokenizer, backbone, load_span_head(Path(cfg.tier1.span_checkpoint), ds.catalog), cfg.model, ds.catalog)
    check_latency(probe_latency(trained, probe_set, ds.catalog, cfg), REFERENCE_MS, MAX_LATENCY_RATIO, "post-train probe")
    del untrained, trained, backbone

    # 4. three-system eval on the same 500 held-out steps, then prediction analysis (fresh process: clean memory)
    subprocess.run([sys.executable, str(ROOT / "eval.py"), "--systems", "tier1,span,baseline"], check=True, cwd=ROOT)
    analysis = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyse_predictions.py")], check=True, cwd=ROOT,
                              capture_output=True, text=True).stdout
    (ROOT / "results" / "analysis.md").write_text("# Prediction analysis (first 500 held-out steps)\n\n" + analysis)
    log.info("analysis:\n%s", analysis)
    log.info("overnight span run complete")


if __name__ == "__main__":
    main()
