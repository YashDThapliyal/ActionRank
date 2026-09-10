import torch

from baseline_sft import build_sft_batch, generation_step_count
from config import load_config
from data import Catalog, Example, ToolSpec


class FakeTokenizer:
    """Deterministic stand-in: one token per whitespace-separated word, EOS = 0, PAD = 0."""

    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages) + " <assistant>"

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=None, add_special_tokens=True):
        seqs = [[hash(w) % 1000 + 1 for w in t.split()] for t in texts]
        width = max(len(s) for s in seqs)
        ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs])
        att = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
        return {"input_ids": ids, "attention_mask": att}


def test_sft_batch_masks_loss_to_answer_tokens_only():
    cat = Catalog((ToolSpec("get_a", "A"), ToolSpec("Finish", "F")))
    examples = [Example("1", "find a", (), ("get_a", "Finish"), "get_a"), Example("2", "done", (), ("get_a", "Finish"), "Finish")]
    batch = build_sft_batch(FakeTokenizer(), examples, cat, load_config().verbalize, max_tokens=512)
    ids, labels, att = batch["input_ids"], batch["labels"], batch["attention_mask"]
    assert ids.shape == labels.shape == att.shape
    for row, ex in enumerate(examples):
        supervised = (labels[row] != -100).nonzero().flatten().tolist()
        assert len(supervised) == 2                      # tool name token + EOS
        assert ids[row, supervised[-1]].item() == 0      # last supervised token is EOS
        assert labels[row, supervised[0]].item() == ids[row, supervised[0]].item()
        assert (labels[row, : supervised[0]] == -100).all()   # prompt tokens are not supervised
    assert (labels[att == 0] == -100).all()              # padding never supervised


def test_generation_step_count_matches_accumulation():
    assert generation_step_count(n_examples=10568, batch_size=2, grad_accum=8) == 661
    assert generation_step_count(n_examples=6, batch_size=2, grad_accum=2) == 2
