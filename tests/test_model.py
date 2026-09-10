import os

import pytest
import torch

from data import Catalog, Example, ToolSpec
from model import ScoringHead, build_candidate_mask, labels_tensor, pool_hidden, resolve_device


def test_mean_pool_respects_mask():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert torch.allclose(pool_hidden(h, m, "mean"), torch.tensor([[2.0, 2.0]]))


def test_last_pool_picks_last_real_token_with_right_padding():
    h = torch.tensor([[[1.0], [3.0], [100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert pool_hidden(h, m, "last").item() == 3.0


def test_pool_returns_float32_even_for_half_input():
    h = torch.randn(2, 4, 8).half()
    m = torch.ones(2, 4, dtype=torch.long)
    assert pool_hidden(h, m, "mean").dtype == torch.float32


def test_pool_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        pool_hidden(torch.zeros(1, 1, 1), torch.ones(1, 1), "max")


def test_head_masks_non_candidates_and_trains():
    head = ScoringHead(num_tools=5, hidden_dim=8, head_hidden=16, temperature=0.1)
    h = torch.randn(2, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], dtype=torch.bool)
    logits = head(h, mask)
    assert logits.shape == (2, 5)
    assert torch.isinf(logits[0, 2]) and torch.isfinite(logits[0, 0])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 4]))
    loss.backward()
    assert head.tool_embedding.weight.grad is not None
    assert head.proj[0].weight.grad is not None


def test_head_without_mask_scores_all_tools():
    head = ScoringHead(num_tools=3, hidden_dim=4, head_hidden=4, temperature=1.0)
    assert torch.isfinite(head(torch.randn(1, 4), None)).all()


def test_candidate_mask_and_labels():
    cat = Catalog((ToolSpec("a", ""), ToolSpec("b", ""), ToolSpec("Finish", "")))
    ex = [Example("1", "q", (), ("b", "Finish"), "Finish"), Example("2", "q", (), ("a",), "a")]
    assert build_candidate_mask(ex, cat).tolist() == [[False, True, True], [True, False, False]]
    assert labels_tensor(ex, cat).tolist() == [2, 0]


def test_resolve_device_cpu_always_ok():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_falls_back_when_unavailable():
    assert resolve_device("cuda").type in ("cuda", "cpu")


@pytest.mark.skipif(os.environ.get("ACTIONRANK_SLOW") != "1", reason="set ACTIONRANK_SLOW=1")
def test_encode_prompts_real_backbone():
    from config import load_config
    from model import encode_prompts, load_backbone

    cfg = load_config().model
    tok, backbone = load_backbone(cfg)
    h = encode_prompts(tok, backbone, ["Task: hello\nNext tool:", "Task: a much longer prompt here\nNext tool:"], cfg)
    assert h.shape == (2, backbone.config.hidden_size)
    assert torch.isfinite(h).all()


def test_rank_excludes_masked_tools_when_fewer_than_k_candidates():
    from model import ActionRankModel

    head = ScoringHead(num_tools=6, hidden_dim=4, head_hidden=4, temperature=1.0)
    model = ActionRankModel.__new__(ActionRankModel)
    torch.nn.Module.__init__(model)
    model.head, model.cfg, model.tokenizer, model.backbone = head, None, None, None
    mask = torch.tensor([[True, False, True, False, False, False]])
    ranked = model.rank_logits(head(torch.randn(1, 4), mask), k=5)
    assert ranked == [(0, 2)] or ranked == [(2, 0)]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a non-CPU device")
def test_actionrank_model_moves_head_to_backbone_device():
    from model import ActionRankModel

    backbone = torch.nn.Linear(2, 2).to("mps")
    head = ScoringHead(num_tools=3, hidden_dim=4, head_hidden=4, temperature=1.0)  # on cpu
    model = ActionRankModel(None, backbone, head, None)
    assert next(model.head.parameters()).device.type == "mps"


def test_load_head_rejects_mismatched_catalog(tmp_path):
    from model import load_head, save_head

    cat = Catalog((ToolSpec("a", ""), ToolSpec("b", "")))
    other = Catalog((ToolSpec("b", ""), ToolSpec("a", "")))
    head = ScoringHead(num_tools=2, hidden_dim=4, head_hidden=4, temperature=1.0)
    save_head(head, tmp_path / "h.pt", cat)
    assert load_head(tmp_path / "h.pt", cat).tool_embedding.num_embeddings == 2
    with pytest.raises(ValueError, match="catalog"):
        load_head(tmp_path / "h.pt", other)


def test_head_starts_as_plain_cosine_similarity():
    head = ScoringHead(num_tools=3, hidden_dim=4, head_hidden=8, temperature=0.5)
    init = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    head.init_tool_embeddings(init)
    h = torch.randn(2, 4)
    expected = torch.nn.functional.normalize(h, dim=-1) @ init.T / 0.5
    assert torch.allclose(head(h, None), expected, atol=1e-5)


def test_init_tool_embeddings_rejects_wrong_shape():
    head = ScoringHead(num_tools=3, hidden_dim=4, head_hidden=8, temperature=0.5)
    with pytest.raises(ValueError):
        head.init_tool_embeddings(torch.zeros(2, 4))


def test_encode_tool_descriptions_prompts_each_tool(monkeypatch):
    import model as model_module

    seen = []

    def fake_encode(tokenizer, backbone, prompts, cfg, grad=False):
        seen.extend(prompts)
        return torch.ones(len(prompts), 4)

    monkeypatch.setattr(model_module, "encode_prompts", fake_encode)
    cat = Catalog((ToolSpec("a", "Alpha"), ToolSpec("b", "Beta"), ToolSpec("Finish", "Stop")))
    vectors = model_module.encode_tool_descriptions(None, None, cat, cfg=None, batch_size=2)
    assert vectors.shape == (3, 4)
    assert seen[0] == "Tool: a\nDescription: Alpha" and len(seen) == 3


def test_load_head_rejects_checkpoint_from_another_architecture(tmp_path):
    from model import load_head, save_head

    cat = Catalog((ToolSpec("a", ""), ToolSpec("b", "")))
    head = ScoringHead(num_tools=2, hidden_dim=4, head_hidden=4, temperature=1.0)
    save_head(head, tmp_path / "h.pt", cat)
    saved = torch.load(tmp_path / "h.pt")
    saved.pop("architecture")
    torch.save(saved, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="architecture"):
        load_head(tmp_path / "old.pt", cat)
