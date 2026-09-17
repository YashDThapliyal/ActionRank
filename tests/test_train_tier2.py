import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from config import load_config
from train_tier2 import trainable_fraction, wrap_lora


def _tiny_backbone():
    cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=100, max_position_embeddings=64)
    return Qwen2ForCausalLM(cfg).eval()


def test_wrap_lora_adds_trainable_params_only_in_q_and_v():
    cfg = load_config().tier2
    model = wrap_lora(_tiny_backbone(), cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora" in n for n in trainable)
    assert all(("q_proj" in n) or ("v_proj" in n) for n in trainable)
    assert 0 < trainable_fraction(model) < 0.2


def test_wrapped_model_exposes_decoder_and_backprops():
    cfg = load_config().tier2
    model = wrap_lora(_tiny_backbone(), cfg)
    ids = torch.randint(0, 100, (2, 7))
    out = model.get_decoder()(input_ids=ids, attention_mask=torch.ones_like(ids))
    out.last_hidden_state.float().pow(2).mean().backward()
    grads = [p.grad for n, p in model.named_parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_train_epoch_reports_per_step_losses_and_optimizer_steps(monkeypatch):
    import dataclasses

    import train_tier2
    from config import load_config
    from data import Catalog, Example, ToolSpec
    from model import ScoringHead

    cfg = load_config()
    cfg = dataclasses.replace(cfg, tier2=dataclasses.replace(cfg.tier2, batch_size=2, grad_accum=2))
    cat = Catalog((ToolSpec("a", "A"), ToolSpec("b", "B")))
    examples = tuple(Example(str(i), "q", (), ("a", "b"), "a" if i % 2 else "b") for i in range(6))

    class FakeModel:
        device = torch.device("cpu")
        head = ScoringHead(2, 4, 4, 1.0)

        def train(self):
            return self

        def __call__(self, prompts, mask, grad=False):
            return self.head(torch.randn(len(prompts), 4), mask)

    model = FakeModel()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.1)
    steps = []
    monkeypatch.setattr(optimizer, "step", lambda: steps.append(1))
    losses = train_tier2._train_epoch(model, optimizer, examples, cat, cfg, torch.Generator().manual_seed(0))
    assert len(losses) == 3          # 6 examples / batch 2 = 3 micro-batches
    assert len(steps) == 2           # accumulation 2 -> steps after micro-batch 2 and the final one
    assert all(isinstance(x, float) for x in losses)


def _tiny_backbone_with_vocab(vocab: int):
    cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=vocab, max_position_embeddings=2048)
    return Qwen2ForCausalLM(cfg).eval()


def test_span_tier2_forward_backprops_into_lora_and_head():
    import dataclasses

    from transformers import AutoTokenizer

    from config import load_config
    from data import Catalog, Example, ToolSpec
    from model import SpanActionRankModel, SpanScoringHead, build_candidate_mask, labels_tensor
    from train_tier2 import tier2_logits, wrap_lora

    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg.model.backbone)
    backbone = wrap_lora(_tiny_backbone_with_vocab(len(tok)), cfg.tier2)
    cat = Catalog((ToolSpec("get_a", "Alpha tool"), ToolSpec("get_b", "Beta tool"), ToolSpec("Finish", "Stop")))
    head = SpanScoringHead(hidden_dim=32, head_hidden=16, temperature=0.1)
    model = SpanActionRankModel(tok, backbone, head, dataclasses.replace(cfg.model, pooling="last"), cat)
    batch = [Example("1", "do a", (), ("get_a", "get_b", "Finish"), "get_a"), Example("2", "done", (), ("get_b", "Finish"), "Finish")]
    logits = tier2_logits(model, batch, cat, cfg, grad=True)
    assert logits.shape == (2, 3) and torch.isinf(logits[1, 0])
    loss = torch.nn.functional.cross_entropy(logits, labels_tensor(batch, cat))
    loss.backward()
    lora_grads = [p.grad for n, p in backbone.named_parameters() if "lora_B" in n]
    assert lora_grads and all(g is not None and g.abs().sum() > 0 for g in lora_grads)
    assert head.tool_proj[0].weight.grad is not None


def test_tier2_head_kind_defaults_to_table_and_reads_meta(tmp_path):
    from train_tier2 import tier2_head_kind

    assert tier2_head_kind(tmp_path) == "table"
    (tmp_path / "meta.json").write_text('{"head": "span"}')
    assert tier2_head_kind(tmp_path) == "span"


def test_load_trainable_adapter_restores_lora_and_keeps_it_trainable(tmp_path):
    from config import load_config
    from train_tier2 import load_trainable_adapter, wrap_lora

    cfg = load_config().tier2
    first = wrap_lora(_tiny_backbone(), cfg)
    with torch.no_grad():
        for n, p in first.named_parameters():
            if "lora_B" in n:
                p.fill_(0.5)
    first.save_pretrained(str(tmp_path / "adapter"))
    resumed = load_trainable_adapter(_tiny_backbone(), tmp_path / "adapter")
    lora_b = [(n, p) for n, p in resumed.named_parameters() if "lora_B" in n]
    assert lora_b and all(torch.all(p == 0.5) for _, p in lora_b)
    assert all(p.requires_grad for _, p in lora_b)
    assert not any(p.requires_grad for n, p in resumed.named_parameters() if "lora" not in n)


def _lora_a_weights(model):
    return [p.detach().clone() for n, p in model.named_parameters() if "lora_A" in n]


def test_wrap_lora_is_deterministic_under_a_seed():
    cfg = load_config().tier2
    a = _lora_a_weights(wrap_lora(_tiny_backbone(), cfg, seed=1))
    b = _lora_a_weights(wrap_lora(_tiny_backbone(), cfg, seed=1))
    c = _lora_a_weights(wrap_lora(_tiny_backbone(), cfg, seed=2))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert any(not torch.equal(x, y) for x, y in zip(a, c))


def test_training_seed_falls_back_to_split_seed():
    import dataclasses

    from train_tier2 import training_seed

    cfg = load_config()
    assert training_seed(cfg) == cfg.data.split_seed
    seeded = dataclasses.replace(cfg, tier2=dataclasses.replace(cfg.tier2, train_seed=99))
    assert training_seed(seeded) == 99
