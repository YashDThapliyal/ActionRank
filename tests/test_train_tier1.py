from pathlib import Path

import torch

from config import load_config
from data import Catalog, Example, ToolSpec
from train_tier1 import CachedSplit, cache_split, load_cached, topk_accuracy, train_head

ROOT = Path(__file__).resolve().parents[1]


def _clustered_split(n: int, num_tools: int, dim: int, seed: int) -> CachedSplit:
    centers = torch.randn(num_tools, dim, generator=torch.Generator().manual_seed(0)) * 3
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_tools, (n,), generator=g)
    h = centers[labels] + 0.3 * torch.randn(n, dim, generator=g)
    mask = torch.ones(n, num_tools, dtype=torch.bool)
    return CachedSplit(h=h, labels=labels, candidate_mask=mask, query_ids=tuple(str(i) for i in range(n)))


def test_topk_accuracy():
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.5, 0.2, 0.3], [0.0, 0.5, 1.0]])
    labels = torch.tensor([1, 2, 0])
    assert topk_accuracy(logits, labels, 1) == 1 / 3
    assert topk_accuracy(logits, labels, 2) == 2 / 3
    assert topk_accuracy(logits, labels, 3) == 1.0


def test_train_head_learns_clustered_vectors():
    cfg = load_config(ROOT / "config.yaml")
    train = _clustered_split(300, 6, 16, seed=1)
    evaluation = _clustered_split(60, 6, 16, seed=2)
    head, history = train_head(train, evaluation, cfg, num_tools=6, hidden_dim=16)
    assert history["eval_top1"][-1] > 0.9
    assert len(history["train_loss"]) == cfg.tier1.epochs
    assert head.tool_embedding.num_embeddings == 6


def test_cache_split_writes_and_reloads(tmp_path, monkeypatch):
    import train_tier1

    cat = Catalog((ToolSpec("a", "A"), ToolSpec("b", "B"), ToolSpec("Finish", "F")))
    examples = tuple(Example(str(i), "q", (), ("a", "Finish"), "a" if i % 2 else "Finish") for i in range(5))
    calls = []

    def fake_encode(tokenizer, model, prompts, cfg, grad=False):
        calls.append(len(prompts))
        return torch.arange(len(prompts), dtype=torch.float32).unsqueeze(1).repeat(1, 4)

    monkeypatch.setattr(train_tier1, "encode_prompts", fake_encode)
    cfg = load_config(ROOT / "config.yaml")
    path = tmp_path / "train.pt"
    split = cache_split(examples, cat, tokenizer=None, backbone=None, cfg=cfg, path=path, batch_size=2)
    assert split.h.shape == (5, 4) and calls == [2, 2, 1]
    assert split.labels.tolist() == [2, 0, 2, 0, 2]
    assert split.candidate_mask.tolist()[0] == [True, False, True]
    assert split.query_ids == ("0", "1", "2", "3", "4")
    reloaded = load_cached(path)
    assert torch.equal(reloaded.h, split.h) and reloaded.query_ids == split.query_ids
    # existing cache is reused without re-encoding
    again = cache_split(examples, cat, None, None, cfg, path, batch_size=2)
    assert calls == [2, 2, 1] and torch.equal(again.h, split.h)


def test_split_cached_by_query_id_is_disjoint_and_deterministic():
    from train_tier1 import split_cached

    qids = tuple(str(i // 3) for i in range(30))  # 10 trajectories x 3 steps
    split = CachedSplit(h=torch.arange(30, dtype=torch.float32).unsqueeze(1), labels=torch.zeros(30, dtype=torch.long),
                        candidate_mask=torch.ones(30, 1, dtype=torch.bool), query_ids=qids)
    train, val = split_cached(split, fraction=0.2, seed=1)
    assert len(train) + len(val) == 30 and len(val) == 6
    assert not set(train.query_ids) & set(val.query_ids)
    assert torch.equal(split_cached(split, 0.2, 1)[1].h, val.h)


def test_cache_is_reencoded_when_prompt_inputs_change(tmp_path, monkeypatch):
    import dataclasses
    import train_tier1

    cat = Catalog((ToolSpec("a", "Alpha tool description"), ToolSpec("Finish", "Finish the task")))
    examples = tuple(Example(str(i), "q", (), ("a", "Finish"), "a") for i in range(3))
    calls = []
    monkeypatch.setattr(train_tier1, "encode_prompts", lambda tok, m, prompts, cfg, grad=False: (calls.append(1), torch.zeros(len(prompts), 2))[1])
    cfg = load_config(ROOT / "config.yaml")
    path = tmp_path / "c.pt"
    cache_split(examples, cat, None, None, cfg, path, batch_size=8)
    cache_split(examples, cat, None, None, cfg, path, batch_size=8)
    assert len(calls) == 1
    changed = dataclasses.replace(cfg, verbalize=dataclasses.replace(cfg.verbalize, description_chars=3))
    cache_split(examples, cat, None, None, changed, path, batch_size=8)
    assert len(calls) == 2
    other_catalog = Catalog((ToolSpec("a", "A-different"), ToolSpec("Finish", "F")))
    cache_split(examples, other_catalog, None, None, changed, path, batch_size=8)
    assert len(calls) == 3


def test_train_head_uses_text_initialised_tool_embeddings():
    cfg = load_config(ROOT / "config.yaml")
    train = _clustered_split(60, 6, 16, seed=1)
    evaluation = _clustered_split(30, 6, 16, seed=2)
    init = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
    head, history = train_head(train, evaluation, cfg, num_tools=6, hidden_dim=16, tool_init=init)
    assert "eval_top1_before_training" in history
    assert head.tool_embedding.weight.shape == (6, 16)


def test_train_head_keeps_initial_head_when_training_does_not_improve():
    import dataclasses

    cfg = load_config(ROOT / "config.yaml")
    frozen = dataclasses.replace(cfg, tier1=dataclasses.replace(cfg.tier1, lr=0.0, epochs=2))
    train = _clustered_split(60, 6, 16, seed=1)
    evaluation = _clustered_split(30, 6, 16, seed=2)
    init = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
    head, history = train_head(train, evaluation, frozen, num_tools=6, hidden_dim=16, tool_init=init)
    assert torch.allclose(head.tool_embedding.weight.detach(), init)
    assert history["best_top1"] == history["eval_top1_before_training"]
