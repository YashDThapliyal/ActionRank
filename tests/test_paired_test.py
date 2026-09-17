import json

from scripts.paired_test import cluster_bootstrap_ci, load_predictions, mcnemar_exact, paired_counts, verdict


def _preds(rows):
    return [{"query_id": q, "label": y, "top1": t} for q, y, t in rows]


def test_paired_counts():
    a = _preds([("1", "x", "x"), ("1", "x", "x"), ("2", "x", "y"), ("2", "x", "y")])
    b = _preds([("1", "x", "x"), ("1", "x", "y"), ("2", "x", "x"), ("2", "x", "y")])
    both, only_a, only_b, neither = paired_counts(a, b)
    assert (both, only_a, only_b, neither) == (1, 1, 1, 1)


def test_mcnemar_exact_symmetric_discordants_is_not_significant():
    assert mcnemar_exact(10, 10) > 0.9


def test_mcnemar_exact_lopsided_discordants_is_significant():
    assert mcnemar_exact(30, 5) < 0.001


def test_cluster_bootstrap_ci_is_tight_when_systems_agree_everywhere():
    a = _preds([(str(i // 3), "x", "x") for i in range(30)])
    lo, hi = cluster_bootstrap_ci(a, a, n_boot=200, seed=0)
    assert lo == 0.0 and hi == 0.0


def test_cluster_bootstrap_resamples_trajectories_not_steps():
    # one trajectory carries every disagreement; a step-level bootstrap would give a narrow CI, a
    # trajectory-level one must let the whole cluster drop out, so the lower bound must reach 0
    a = _preds([("big", "x", "x")] * 10 + [(str(i), "x", "x") for i in range(20)])
    b = _preds([("big", "x", "y")] * 10 + [(str(i), "x", "x") for i in range(20)])
    lo, hi = cluster_bootstrap_ci(a, b, n_boot=500, seed=1)
    assert lo <= 0.0 < hi


def test_verdict_equivalent_inside_margin():
    assert verdict((-1.0, 2.0), margin=3.0) == "equivalent"


def test_verdict_inconclusive_when_interval_crosses_margin():
    assert verdict((-1.0, 4.0), margin=3.0) == "inconclusive"


def test_verdict_difference_when_interval_excludes_zero_and_margin():
    assert verdict((3.5, 7.0), margin=3.0) == "first system better"
    assert verdict((-7.0, -3.5), margin=3.0) == "second system better"


def test_load_predictions_reads_jsonl(tmp_path):
    p = tmp_path / "predictions_x.jsonl"
    p.write_text(json.dumps({"query_id": "1", "label": "a", "top1": "a", "topk": ["a"], "valid": True}) + "\n")
    assert load_predictions(p)[0]["top1"] == "a"
