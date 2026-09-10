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
