from pathlib import Path

import torch

from config import load_config
from data import Catalog, Example, ToolSpec
from train_tier1 import CachedSpanSplit, cache_span_split, split_cached_spans, train_span_head

ROOT = Path(__file__).resolve().parents[1]


def _clustered(n: int, dim: int, seed: int) -> CachedSpanSplit:
    """Each example: query vector near the vector of its labelled candidate; 3 candidates per example."""
    g = torch.Generator().manual_seed(seed)
    tools = torch.randn(n, 3, dim, generator=g)
    labels_local = torch.randint(0, 3, (n,), generator=g)
    q = tools[torch.arange(n), labels_local] + 0.1 * torch.randn(n, dim, generator=g)
    tool_idx = torch.stack([torch.randperm(10, generator=g)[:3] for _ in range(n)])
    labels = tool_idx[torch.arange(n), labels_local]
    return CachedSpanSplit(q=q, tools=tools, tool_mask=torch.ones(n, 3, dtype=torch.bool), tool_idx=tool_idx,
                           labels=labels, query_ids=tuple(str(i // 2) for i in range(n)))


def test_split_cached_spans_by_query_id():
    split = _clustered(40, 8, seed=0)
    train, val = split_cached_spans(split, 0.25, seed=1)
    assert len(train) + len(val) == 40 and len(val) == 10
    assert not set(train.query_ids) & set(val.query_ids)


def test_train_span_head_learns():
    cfg = load_config(ROOT / "config.yaml")
    head, history = train_span_head(_clustered(200, 16, 1), _clustered(60, 16, 2), cfg, num_tools=10, hidden_dim=16)
    assert history["eval_top1"][-1] > 0.9
    assert "eval_top1_before_training" in history


def test_cache_span_split_shapes_and_reuse(tmp_path, monkeypatch):
    import train_tier1

    cat = Catalog((ToolSpec("a", "Alpha tool"), ToolSpec("b", "Beta tool"), ToolSpec("Finish", "Stop")))
    examples = (Example("1", "q", (), ("a", "Finish"), "a"), Example("2", "q", (), ("a", "b", "Finish"), "b"))
    calls = []

    def fake_encode(tokenizer, backbone, prompts, spans, cfg, grad=False):
        calls.append(len(prompts))
        return torch.zeros(len(prompts), 4), [torch.ones(len(s), 4) for s in spans]

    monkeypatch.setattr(train_tier1, "encode_with_spans", fake_encode)
    cfg = load_config(ROOT / "config.yaml")
    split = cache_span_split(examples, cat, None, None, cfg, tmp_path / "s.pt", batch_size=8)
    assert split.q.shape == (2, 4) and split.tools.shape == (2, 3, 4)
    assert split.tool_mask.tolist() == [[True, True, False], [True, True, True]]
    assert split.tool_idx[0].tolist()[:2] == [0, 2] and split.labels.tolist() == [0, 1]
    cache_span_split(examples, cat, None, None, cfg, tmp_path / "s.pt", batch_size=8)
    assert calls == [2]
