"""Joint ranking + language-modeling objective for the span scorer (GenRec's Phase 2 loss)."""
import dataclasses

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from config import load_config
from data import Catalog, Example, ToolSpec
from model import SpanActionRankModel, SpanScoringHead, build_joint_batch, encode_with_spans_lm


def _tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(load_config().model.backbone, local_files_only=True)
    except Exception:  # pragma: no cover - only when the backbone tokenizer is not cached locally
        pytest.skip("backbone tokenizer not cached locally")


def _tiny_backbone(vocab: int):
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=vocab, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).eval()


def _examples():
    cat = Catalog((ToolSpec("alpha_tool", "does alpha things"), ToolSpec("beta_tool", "does beta things"),
                   ToolSpec("gamma_tool", "does gamma things"), ToolSpec("Finish", "stop")))
    ex = (Example("1", "find alpha", (), ("alpha_tool", "beta_tool", "Finish"), "alpha_tool"),
          Example("2", "find gamma please, a longer query", (), ("gamma_tool", "beta_tool", "Finish"), "gamma_tool"))
    return cat, ex


def test_build_joint_batch_answer_scope_labels_only_answer_tokens():
    tok = _tokenizer()
    prompts = ["Task: a\nNext tool:", "Task: a longer prompt here\nNext tool:"]
    answers = [" alpha_tool", " beta_tool"]
    batch = build_joint_batch(tok, prompts, answers, max_tokens=64, lm_scope="answer")
    for i in range(2):
        p = int(batch["prompt_len"][i])
        n_ans = len(tok(answers[i], add_special_tokens=False)["input_ids"]) + 1  # + EOS
        labels = batch["labels"][i]
        assert (labels[:p] == -100).all(), "prompt tokens must be ignored in answer scope"
        assert (labels[p:p + n_ans] != -100).all(), "every answer token (and EOS) is supervised"
        assert (labels[p + n_ans:] == -100).all(), "padding is ignored"
        assert batch["input_ids"][i, p + n_ans - 1] == tok.eos_token_id


def test_build_joint_batch_all_scope_labels_prompt_and_answer():
    tok = _tokenizer()
    batch = build_joint_batch(tok, ["Task: a\nNext tool:"], [" alpha_tool"], max_tokens=64, lm_scope="all")
    seq_len = int(batch["attention_mask"][0].sum())
    assert (batch["labels"][0, :seq_len] == batch["input_ids"][0, :seq_len]).all()
    assert (batch["labels"][0, seq_len:] == -100).all()


def test_build_joint_batch_rejects_unknown_scope():
    tok = _tokenizer()
    with pytest.raises(ValueError):
        build_joint_batch(tok, ["x"], [" y"], max_tokens=16, lm_scope="nope")


def test_joint_forward_ranking_logits_do_not_see_the_answer():
    """The pooled query and the span vectors must come from prompt positions only, so the catalog logits
    are identical whether or not the answer is appended. Otherwise the label leaks into the score."""
    tok = _tokenizer()
    cat, ex = _examples()
    cfg = load_config()
    mcfg = dataclasses.replace(cfg.model, pooling="last", max_prompt_tokens=256, device="cpu")
    backbone = _tiny_backbone(len(tok))
    head = SpanScoringHead(32, 16, 0.05)
    model = SpanActionRankModel(tok, backbone, head, mcfg, cat)
    plain = model(ex, cfg.verbalize, grad=False)
    joint, lm_loss = model.forward_with_lm(ex, cfg.verbalize, lm_scope="all")
    finite = torch.isfinite(plain)
    assert torch.equal(finite, torch.isfinite(joint))
    assert torch.allclose(plain[finite], joint[finite], atol=1e-4)
    assert lm_loss.ndim == 0 and torch.isfinite(lm_loss) and lm_loss.item() > 0


def test_joint_forward_lm_loss_is_answer_only_when_scoped():
    tok = _tokenizer()
    cat, ex = _examples()
    cfg = load_config()
    mcfg = dataclasses.replace(cfg.model, pooling="last", max_prompt_tokens=256, device="cpu")
    model = SpanActionRankModel(tok, _tiny_backbone(len(tok)), SpanScoringHead(32, 16, 0.05), mcfg, cat)
    _, lm_all = model.forward_with_lm(ex, cfg.verbalize, lm_scope="all")
    _, lm_ans = model.forward_with_lm(ex, cfg.verbalize, lm_scope="answer")
    assert lm_all.item() != pytest.approx(lm_ans.item())


def test_train_epoch_combines_rank_and_lm_loss(monkeypatch):
    import train_tier2

    cfg = load_config()
    cfg = dataclasses.replace(cfg, tier2=dataclasses.replace(cfg.tier2, batch_size=2, grad_accum=1, lm_weight=0.5))
    cat, ex = _examples()

    class FakeSpan(SpanActionRankModel):
        def __init__(self):  # noqa: D401 - bypass nn.Module wiring; only the forward hooks are used
            torch.nn.Module.__init__(self)
            self.catalog = cat
            self.head = SpanScoringHead(4, 4, 1.0)

        @property
        def device(self):
            return torch.device("cpu")

        def forward_with_lm(self, examples, verbalize_cfg, lm_scope="all"):
            logits = torch.zeros(len(examples), len(cat), requires_grad=True)
            return logits, torch.tensor(2.0, requires_grad=True)

    model = FakeSpan()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    monkeypatch.setattr(optimizer, "step", lambda: None)
    losses = train_tier2._train_epoch(model, optimizer, ex, cat, cfg, torch.Generator().manual_seed(0))
    # rank loss: uniform over 3 candidates = ln 3; plus 0.5 * 2.0
    assert losses[0] == pytest.approx(torch.log(torch.tensor(3.0)).item() + 1.0, abs=1e-4)


def test_lm_weight_requires_span_head():
    import train_tier2

    cfg = load_config()
    cfg = dataclasses.replace(cfg, tier2=dataclasses.replace(cfg.tier2, head="table", lm_weight=1.0))
    with pytest.raises(ValueError):
        train_tier2.check_joint_config(cfg)
