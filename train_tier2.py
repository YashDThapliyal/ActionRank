"""Tier 2: LoRA on q_proj/v_proj, fine-tuned jointly with the scoring head."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm

from config import Config, Tier2Config, load_config
from data import Catalog, Dataset, Example, load_dataset
from model import (ActionRankModel, ScoringHead, build_candidate_mask, encode_prompts, labels_tensor,
                   load_backbone, load_head, save_head)
from train_tier1 import topk_accuracy
from verbalize import build_prompt

log = logging.getLogger(__name__)
LORA_TARGETS = ("q_proj", "v_proj")
EVAL_SUBSET = 200


def wrap_lora(backbone, cfg: Tier2Config):
    """Inject LoRA adapters into q/v projections; only adapter weights are trainable."""
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(LORA_TARGETS), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(backbone, lora)


def trainable_fraction(model) -> float:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable / total if total else 0.0


def _init_head(cfg: Config, catalog: Catalog, hidden_dim: int) -> ScoringHead:
    path = Path(cfg.tier1.checkpoint)
    if path.exists():
        try:
            head = load_head(path, catalog)
        except ValueError as err:
            log.warning("%s; training head from scratch", err)
        else:
            if head.tool_embedding.embedding_dim == hidden_dim:
                log.info("initialising head from %s", path)
                return head
            log.warning("tier 1 head at %s has a different hidden size; training head from scratch", path)
    return ScoringHead(len(catalog), hidden_dim, cfg.model.head_hidden, cfg.model.score_temperature)


def _batches(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


@torch.no_grad()
def _eval_top1(model: ActionRankModel, examples: Sequence[Example], catalog: Catalog, cfg: Config) -> float:
    model.eval()
    logits, labels = [], labels_tensor(examples, catalog)
    for batch in _batches(list(examples), cfg.tier1.cache_batch_size):
        prompts = [build_prompt(ex, catalog, cfg.verbalize) for ex in batch]
        logits.append(model(prompts, build_candidate_mask(batch, catalog).to(model.device)).cpu())
    return topk_accuracy(torch.cat(logits), labels, 1)


def _train_epoch(model: ActionRankModel, optimizer, examples: Sequence[Example], catalog: Catalog, cfg: Config,
                 generator: torch.Generator) -> float:
    t2, losses = cfg.tier2, []
    order = torch.randperm(len(examples), generator=generator).tolist()
    batches = list(_batches([examples[i] for i in order], t2.batch_size))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(batches, desc="tier2 train", leave=False), start=1):
        prompts = [build_prompt(ex, catalog, cfg.verbalize) for ex in batch]
        logits = model(prompts, build_candidate_mask(batch, catalog).to(model.device), grad=True)
        loss = torch.nn.functional.cross_entropy(logits, labels_tensor(batch, catalog).to(model.device))
        (loss / t2.grad_accum).backward()
        losses.append(loss.item())
        if step % t2.grad_accum == 0 or step == len(batches):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return sum(losses) / max(len(losses), 1)


def train_tier2(ds: Dataset, cfg: Config, tokenizer, backbone, limit: int | None = None) -> Path:
    """Joint AdamW over LoRA + head; saves adapter and head under cfg.tier2.checkpoint_dir."""
    t2 = cfg.tier2
    train_ex = ds.train[:limit or t2.max_train_examples]
    eval_ex = ds.eval[:EVAL_SUBSET]
    peft_model = wrap_lora(backbone, t2)
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.enable_input_require_grads()
    device = next(peft_model.parameters()).device
    head = _init_head(cfg, ds.catalog, backbone.config.hidden_size).to(device)
    model = ActionRankModel(tokenizer, peft_model, head, cfg.model)
    params = [p for p in peft_model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=t2.lr)
    log.info("trainable fraction of backbone: %.4f; train examples %d", trainable_fraction(peft_model), len(train_ex))
    history: dict[str, list[float]] = {"train_loss": [], "eval_top1": [_eval_top1(model, eval_ex, ds.catalog, cfg)]}
    log.info("eval top1 before training: %.4f", history["eval_top1"][0])
    generator = torch.Generator().manual_seed(cfg.data.split_seed)
    started = time.perf_counter()
    for epoch in range(t2.epochs):
        loss = _train_epoch(model, optimizer, train_ex, ds.catalog, cfg, generator)
        top1 = _eval_top1(model, eval_ex, ds.catalog, cfg)
        history["train_loss"].append(loss)
        history["eval_top1"].append(top1)
        log.info("epoch %d loss %.4f eval top1 %.4f", epoch + 1, loss, top1)
    out = Path(t2.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out / "adapter"))
    save_head(head.cpu(), out / "head.pt", ds.catalog)
    (out / "history.json").write_text(json.dumps({"history": history, "n_train": len(train_ex), "n_eval_subset": len(eval_ex),
                                                  "train_seconds": time.perf_counter() - started}, indent=1))
    return out


def load_tier2(cfg: Config, catalog: Catalog, tokenizer=None, backbone=None) -> ActionRankModel:
    from peft import PeftModel

    out = Path(cfg.tier2.checkpoint_dir)
    if tokenizer is None or backbone is None:
        tokenizer, backbone = load_backbone(cfg.model)
    peft_model = PeftModel.from_pretrained(backbone, str(out / "adapter")).eval()
    return ActionRankModel(tokenizer, peft_model, load_head(out / "head.pt", catalog), cfg.model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2: LoRA fine-tune backbone + head")
    parser.add_argument("--limit", type=int, default=None, help="override tier2.max_train_examples")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    ds = load_dataset(cfg)
    tokenizer, backbone = load_backbone(cfg.model)
    out = train_tier2(ds, cfg, tokenizer, backbone, limit=args.limit)
    print(f"saved tier 2 checkpoint to {out}")
    print((out / "history.json").read_text())


if __name__ == "__main__":
    main()
