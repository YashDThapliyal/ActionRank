"""Fine-tuned generation baseline: LoRA on the same backbone, trained to emit the next tool name.

Same adapter budget as Tier 2 (rank, targets, lr, batch, accumulation) so the comparison with ActionRank is
symmetric: 'trained to generate the name' vs. 'trained to score the catalog'.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm

from config import Config, load_config
from data import Catalog, Dataset, Example, load_dataset
from model import load_backbone
from train_tier2 import load_trainable_adapter, trainable_fraction, wrap_lora, training_seed
from verbalize import build_baseline_messages

log = logging.getLogger(__name__)
IGNORE = -100
EVAL_SUBSET = 100
PAD_BUCKET = 64              # pad every batch to a multiple of this so the MPS allocator reuses blocks
MPS_MEMORY_FRACTION = 0.7    # fail fast on OOM instead of letting the allocator grow into swap
EXPECTED_S_PER_BATCH = 2.0   # batch-1 micro-batch time measured in the smoke run
THROUGHPUT_MAX_RATIO = 3.0
THROUGHPUT_CHECK_AFTER = 100


def _ids(tokenizer, text: str) -> list[int]:
    out = tokenizer([text], return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    ids, att = out["input_ids"][0], out["attention_mask"][0]
    return ids[att.bool()].tolist()


def build_sft_batch(tokenizer, examples: Sequence[Example], catalog: Catalog, verbalize_cfg,
                    max_tokens: int) -> dict[str, torch.Tensor]:
    """Chat prompt + tool name + EOS, right padded; labels are IGNORE everywhere except the answer tokens."""
    rows: list[tuple[list[int], list[int]]] = []
    for ex in examples:
        prompt = tokenizer.apply_chat_template(build_baseline_messages(ex, catalog, verbalize_cfg), tokenize=False,
                                               add_generation_prompt=True)
        answer = _ids(tokenizer, ex.label) + [tokenizer.eos_token_id]
        prompt_ids = _ids(tokenizer, prompt)[-(max_tokens - len(answer)):]  # left-truncate the prompt
        rows.append((prompt_ids, answer))
    longest = max(len(p) + len(a) for p, a in rows)
    width = math.ceil(longest / PAD_BUCKET) * PAD_BUCKET
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    for r, (p, a) in enumerate(rows):
        seq = p + a
        input_ids[r, :len(seq)] = torch.tensor(seq)
        labels[r, len(p):len(seq)] = torch.tensor(a)
        attention[r, :len(seq)] = 1
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def generation_step_count(n_examples: int, batch_size: int, grad_accum: int) -> int:
    return math.ceil(math.ceil(n_examples / batch_size) / grad_accum)


def make_throughput_guard(expected_s: float, max_ratio: float, check_after: int):
    """Returns guard(step, elapsed_s) that raises once the sustained s/step exceeds max_ratio x expected."""
    def guard(step: int, elapsed_s: float) -> None:
        if step < check_after:
            return
        per_step = elapsed_s / step
        if per_step > expected_s * max_ratio:
            raise RuntimeError(f"SFT throughput {per_step:.1f} s/step after {step} steps exceeds {max_ratio}x the "
                               f"expected {expected_s} s/step; aborting (memory thrash?)")
    return guard


def _release_accelerator_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _cap_accelerator_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)


def _batches(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


@torch.no_grad()
def _quick_top1(model, tokenizer, examples: Sequence[Example], catalog: Catalog, cfg: Config) -> float:
    """Greedy-generation top-1 on a small subset (used before/after training as a sanity signal)."""
    import dataclasses

    from baseline import predict_one

    greedy = dataclasses.replace(cfg, baseline=dataclasses.replace(cfg.baseline, num_beams=1))  # no beam pass here
    model.eval()
    hits = sum(predict_one(tokenizer, model, ex, catalog, greedy).top1 == ex.label for ex in examples)
    return hits / max(len(examples), 1)


def train_sft(ds: Dataset, cfg: Config, tokenizer, backbone, limit: int | None = None,
              resume: Path | None = None) -> Path:
    t2, b = cfg.tier2, cfg.baseline
    train_ex = ds.train[:limit] if limit else ds.train
    eval_ex = ds.eval[:EVAL_SUBSET]
    if resume is not None:
        log.info("resuming adapter from %s", resume)
        model = load_trainable_adapter(backbone, resume / "adapter")
    else:
        model = wrap_lora(backbone, t2, seed=training_seed(cfg))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    device = next(model.parameters()).device
    _cap_accelerator_memory(device)
    batch_size, grad_accum = b.sft_batch_size, b.sft_grad_accum
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t2.lr)
    log.info("device %s; trainable fraction %.4f; train examples %d; batch %d x accum %d; optimizer steps/epoch %d", device,
             trainable_fraction(model), len(train_ex), batch_size, grad_accum, generation_step_count(len(train_ex), batch_size, grad_accum))
    guard = make_throughput_guard(EXPECTED_S_PER_BATCH, THROUGHPUT_MAX_RATIO, THROUGHPUT_CHECK_AFTER)
    history: dict = {"step_losses": [], "train_loss": [], "eval_top1": [_quick_top1(model, tokenizer, eval_ex, ds.catalog, cfg)]}
    log.info("eval top1 (n=%d) before training: %.4f", len(eval_ex), history["eval_top1"][0])
    generator = torch.Generator().manual_seed(training_seed(cfg))
    started = time.perf_counter()
    for epoch in range(b.sft_epochs):
        model.train()
        order = torch.randperm(len(train_ex), generator=generator).tolist()
        batches = list(_batches([train_ex[i] for i in order], batch_size))
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        for step, batch in enumerate(tqdm(batches, desc=f"sft epoch {epoch + 1}", leave=False), start=1):
            inputs = {k: v.to(device) for k, v in build_sft_batch(tokenizer, batch, ds.catalog, cfg.verbalize, cfg.model.max_prompt_tokens).items()}
            loss = model(**inputs).loss
            (loss / grad_accum).backward()
            losses.append(loss.item())
            del loss, inputs
            if step % grad_accum == 0 or step == len(batches):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _release_accelerator_cache(device)
            guard(step, time.perf_counter() - epoch_started)
            if step % 500 == 0:
                log.info("step %d/%d  mean loss last 500: %.4f  %.2f s/step", step, len(batches), sum(losses[-500:]) / 500, (time.perf_counter() - epoch_started) / step)
        top1 = _quick_top1(model, tokenizer, eval_ex, ds.catalog, cfg)
        history["step_losses"].append(losses)
        history["train_loss"].append(sum(losses) / max(len(losses), 1))
        history["eval_top1"].append(top1)
        log.info("epoch %d loss %.4f eval top1 (n=%d) %.4f", epoch + 1, history["train_loss"][-1], len(eval_ex), top1)
    out = Path(b.sft_checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "adapter"))
    (out / "history.json").write_text(json.dumps({"history": history, "n_train": len(train_ex), "n_eval_subset": len(eval_ex),
                                                  "train_seconds": time.perf_counter() - started,
                                                  "resumed_from": str(resume) if resume else None}, indent=1))
    return out


def load_sft_model(cfg: Config, tokenizer=None, backbone=None):
    """Fresh backbone + merged adapter (merged so generation latency is that of a plain model)."""
    from peft import PeftModel

    if tokenizer is None or backbone is None:
        tokenizer, backbone = load_backbone(cfg.model)
    model = PeftModel.from_pretrained(backbone, str(Path(cfg.baseline.sft_checkpoint_dir) / "adapter"))
    return tokenizer, model.merge_and_unload().eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the generation baseline with LoRA")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N training examples")
    parser.add_argument("--resume", type=Path, default=None, help="continue from this SFT checkpoint dir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    ds = load_dataset(cfg)
    tokenizer, backbone = load_backbone(cfg.model)
    out = train_sft(ds, cfg, tokenizer, backbone, limit=args.limit, resume=args.resume)
    print(f"saved SFT baseline to {out}")
    print((out / "history.json").read_text()[:600])


if __name__ == "__main__":
    main()
