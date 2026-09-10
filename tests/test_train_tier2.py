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
