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
