from eval import Metrics, compute_metrics, reference_rows, render_table
from data import Example


def _rows():
    labels = ["a", "b", "Finish", "c"]
    top1 = ["a", "x", "Finish", "b"]
    topk = [("a", "b"), ("x", "b", "c"), ("Finish",), ("b", "a")]
    valid = [True, False, True, True]
    latencies = [0.010, 0.020, 0.030, 0.040]
    return labels, top1, topk, valid, latencies


def test_compute_metrics_counts():
    m = compute_metrics("sys", *_rows())
    assert isinstance(m, Metrics)
    assert m.n == 4
    assert m.top1 == 2 / 4
    assert m.top5 == 3 / 4
    assert m.hallucination_rate == 1 / 4
    assert m.latency_mean_ms == 25.0
    assert m.latency_p50_ms == 25.0


def test_compute_metrics_excludes_finish_subset():
    m = compute_metrics("sys", *_rows())
    assert m.n_no_finish == 3
    assert m.top1_no_finish == 1 / 3


def test_compute_metrics_without_latency():
    labels, top1, topk, valid, _ = _rows()
    m = compute_metrics("ref", labels, top1, topk, valid, None)
    assert m.latency_mean_ms is None and m.latency_p50_ms is None


def test_compute_metrics_rejects_length_mismatch():
    import pytest

    labels, top1, topk, valid, lat = _rows()
    with pytest.raises(ValueError):
        compute_metrics("sys", labels, top1[:-1], topk, valid, lat)


def test_render_table_has_header_and_rows():
    m = compute_metrics("sys", *_rows())
    text = render_table([m, m])
    lines = text.strip().splitlines()
    assert lines[0].startswith("| system |")
    assert len(lines) == 4  # header, separator, two rows
    assert "sys" in lines[2] and "50.0%" in lines[2]


def test_reference_rows_random_and_frequent():
    train = [Example("1", "q", (), ("a", "b", "Finish"), "a"), Example("2", "q", (), ("a", "b", "Finish"), "a"),
             Example("3", "q", (), ("a", "b", "Finish"), "Finish")]
    evaluation = [Example("4", "q", (), ("a", "b", "Finish"), "a"), Example("5", "q", (), ("b", "Finish"), "b")]
    rows = reference_rows(train, evaluation, seed=0)
    names = [r.name for r in rows]
    assert names == ["random-candidate", "most-frequent-candidate"]
    frequent = rows[1]
    assert frequent.top1 == 0.5  # predicts 'a' for ex4 (hit), 'Finish' for ex5 (miss: 'a' not a candidate)
    assert frequent.hallucination_rate == 0.0
    assert rows[0].hallucination_rate == 0.0


def test_top5_counts_top1_hit_even_if_absent_from_topk_list():
    m = compute_metrics("s", ["a"], ["a"], [("b", "c")], [True], None)
    assert m.top1 == 1.0 and m.top5 == 1.0


def test_top5_uses_at_most_five_entries():
    m = compute_metrics("s", ["z"], ["a"], [("b", "c", "d", "e", "f", "z")], [True], None)
    assert m.top5 == 0.0


def test_evaluate_actionrank_caps_warmup_to_available_examples():
    import torch

    from config import load_config
    from data import Catalog, ToolSpec
    from eval import evaluate_actionrank

    class FakeModel:
        device = torch.device("cpu")

        def eval(self):
            return self

        def rank(self, prompts, mask, k):
            return [(0,)]

    cat = Catalog((ToolSpec("a", "A"), ToolSpec("Finish", "F")))
    ex = [Example("1", "q", (), ("a", "Finish"), "a")]
    m, _ = evaluate_actionrank(FakeModel(), ex, cat, load_config(), "fake", warmup=3)
    assert m.n == 1 and m.top1 == 1.0


def test_evaluate_actionrank_records_per_example_predictions():
    import torch

    from config import load_config
    from data import Catalog, ToolSpec
    from eval import evaluate_actionrank, predictions_to_jsonl

    class FakeModel:
        device = torch.device("cpu")

        def eval(self):
            return self

        def rank(self, prompts, mask, k):
            return [(1, 0)]

    cat = Catalog((ToolSpec("a", "A"), ToolSpec("Finish", "F")))
    ex = [Example("1", "q", (), ("a", "Finish"), "a")]
    m, preds = evaluate_actionrank(FakeModel(), ex, cat, load_config(), "fake")
    assert m.top1 == 0.0 and m.top5 == 1.0
    assert preds == [{"query_id": "1", "label": "a", "top1": "Finish", "topk": ["Finish", "a"], "valid": True}]
    assert predictions_to_jsonl(preds).strip().startswith('{"query_id": "1"')


def test_write_results_removes_stale_prediction_files(tmp_path):
    from eval import write_results

    stale = tmp_path / "predictions_tier2.jsonl"
    stale.write_text("{}\n")
    m = compute_metrics("sys", *_rows())
    write_results([m], tmp_path, predictions={"tier1": [{"query_id": "1"}]})
    assert (tmp_path / "predictions_tier1.jsonl").exists()
    assert not stale.exists()
