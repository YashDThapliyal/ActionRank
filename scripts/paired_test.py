"""Paired top-1 analysis of two systems' prediction files on the same steps.

Primary: scorer-minus-generator top-1 difference with a trajectory-clustered bootstrap 95% CI and a
predeclared practical-equivalence margin. Secondary: step-level exact McNemar test on the discordant pairs.

    .venv/bin/python scripts/paired_test.py results/06-unmonitored-holdout/predictions_tier2.jsonl \
        results/06-unmonitored-holdout/predictions_baseline_sft.jsonl --margin 3
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Sequence


def load_predictions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _hits(preds: Sequence[dict]) -> list[bool]:
    return [p["top1"] == p["label"] for p in preds]


def _check_aligned(a: Sequence[dict], b: Sequence[dict]) -> None:
    if len(a) != len(b):
        raise ValueError(f"prediction files differ in length: {len(a)} vs {len(b)}")
    for x, y in zip(a, b):
        if x["query_id"] != y["query_id"] or x["label"] != y["label"]:
            raise ValueError("prediction files are not over the same steps in the same order")


def paired_counts(a: Sequence[dict], b: Sequence[dict]) -> tuple[int, int, int, int]:
    """(both right, only a right, only b right, neither)."""
    _check_aligned(a, b)
    ha, hb = _hits(a), _hits(b)
    both = sum(x and y for x, y in zip(ha, hb))
    only_a = sum(x and not y for x, y in zip(ha, hb))
    only_b = sum(y and not x for x, y in zip(ha, hb))
    return both, only_a, only_b, len(ha) - both - only_a - only_b


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_bootstrap_ci(a: Sequence[dict], b: Sequence[dict], n_boot: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float]:
    """95% percentile CI, in percentage points, for mean(top-1 a) - mean(top-1 b), resampling trajectories."""
    _check_aligned(a, b)
    by_traj: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(a):
        by_traj[p["query_id"]].append(i)
    trajs = list(by_traj)
    ha, hb = _hits(a), _hits(b)
    diff_by_traj = {t: sum(ha[i] - hb[i] for i in idx) for t, idx in by_traj.items()}
    size_by_traj = {t: len(idx) for t, idx in by_traj.items()}
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [trajs[rng.randrange(len(trajs))] for _ in trajs]
        n = sum(size_by_traj[t] for t in sample)
        draws.append(100.0 * sum(diff_by_traj[t] for t in sample) / n)
    draws.sort()
    lo = draws[int(alpha / 2 * n_boot)]
    hi = draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def verdict(ci: tuple[float, float], margin: float) -> str:
    lo, hi = ci
    if -margin < lo and hi < margin:
        return "equivalent"
    if lo > 0 and lo >= margin:
        return "first system better"
    if hi < 0 and hi <= -margin:
        return "second system better"
    return "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first", help="predictions_*.jsonl for the first system (reported as first minus second)")
    ap.add_argument("second")
    ap.add_argument("--margin", type=float, default=3.0, help="practical-equivalence margin in percentage points")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    a, b = load_predictions(args.first), load_predictions(args.second)
    both, only_a, only_b, neither = paired_counts(a, b)
    n = len(a)
    acc_a, acc_b = 100 * (both + only_a) / n, 100 * (both + only_b) / n
    lo, hi = cluster_bootstrap_ci(a, b, n_boot=args.boot, seed=args.seed)
    print(f"n = {n} steps, {len({p['query_id'] for p in a})} trajectories")
    print(f"top-1: first {acc_a:.1f}%  second {acc_b:.1f}%  difference {acc_a - acc_b:+.1f} pt")
    print(f"paired: both {both}, only first {only_a}, only second {only_b}, neither {neither}")
    print(f"trajectory-clustered bootstrap 95% CI for the difference: [{lo:+.1f}, {hi:+.1f}] pt")
    print(f"equivalence margin ±{args.margin:g} pt -> {verdict((lo, hi), args.margin)}")
    print(f"secondary, step-level exact McNemar p = {mcnemar_exact(only_a, only_b):.3f}")


if __name__ == "__main__":
    main()
