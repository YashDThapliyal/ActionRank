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
from model import (ActionRankModel, ScoringHead, SpanActionRankModel, SpanScoringHead, build_candidate_mask,
                   encode_prompts, labels_tensor, load_backbone, load_head, load_span_head, save_head, save_span_head)
from torch import Tensor

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


def load_trainable_adapter(backbone, adapter_dir: Path):
    """Reload a saved LoRA adapter for continued training (adapter weights trainable, base frozen)."""
    from peft import PeftModel

    return PeftModel.from_pretrained(backbone, str(adapter_dir), is_trainable=True)


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


def _init_span_head(cfg: Config, catalog: Catalog, hidden_dim: int) -> SpanScoringHead:
    path = Path(cfg.tier1.span_checkpoint)
    if path.exists():
        try:
            head = load_span_head(path, catalog)
        except ValueError as err:
            log.warning("%s; training span head from scratch", err)
        else:
            if head.hidden_dim == hidden_dim:
                log.info("initialising span head from %s", path)
                return head
    return SpanScoringHead(hidden_dim, cfg.model.head_hidden, cfg.model.score_temperature)


def tier2_logits(model, batch: Sequence[Example], catalog: Catalog, cfg: Config, grad: bool) -> Tensor:
    """Catalog-width logits for a batch from either model type (table head or span head)."""
    mask = build_candidate_mask(batch, catalog)
    if isinstance(model, SpanActionRankModel):
        logits = model(batch, cfg.verbalize, grad=grad)
        return logits.masked_fill(~mask.to(logits.device), float("-inf"))
    prompts = [build_prompt(ex, catalog, cfg.verbalize) for ex in batch]
    return model(prompts, mask.to(model.device), grad=grad)


def tier2_head_kind(checkpoint_dir: Path) -> str:
    meta = checkpoint_dir / "meta.json"
    return json.loads(meta.read_text()).get("head", "table") if meta.exists() else "table"


def _batches(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


@torch.no_grad()
def _eval_top1(model: ActionRankModel, examples: Sequence[Example], catalog: Catalog, cfg: Config) -> float:
    model.eval()
    logits, labels = [], labels_tensor(examples, catalog)
    for batch in _batches(list(examples), cfg.tier1.cache_batch_size):
        logits.append(tier2_logits(model, batch, catalog, cfg, grad=False).cpu())
    return topk_accuracy(torch.cat(logits), labels, 1)


def _train_epoch(model: ActionRankModel, optimizer, examples: Sequence[Example], catalog: Catalog, cfg: Config,
                 generator: torch.Generator) -> list[float]:
    """One pass over the examples; returns the loss of every micro-batch (an optimizer step happens every
    `grad_accum` micro-batches and after the last one)."""
    t2, losses = cfg.tier2, []
    order = torch.randperm(len(examples), generator=generator).tolist()
    batches = list(_batches([examples[i] for i in order], t2.batch_size))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(batches, desc="tier2 train", leave=False), start=1):
        logits = tier2_logits(model, batch, catalog, cfg, grad=True)
        loss = torch.nn.functional.cross_entropy(logits, labels_tensor(batch, catalog).to(logits.device))
        (loss / t2.grad_accum).backward()
        losses.append(loss.item())
        if step % t2.grad_accum == 0 or step == len(batches):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return losses


def train_tier2(ds: Dataset, cfg: Config, tokenizer, backbone, limit: int | None = None,
                resume: Path | None = None) -> Path:
    """Joint AdamW over LoRA + head; saves adapter and head under cfg.tier2.checkpoint_dir.

    `resume` continues from a previous Tier 2 checkpoint directory (adapter + head) instead of starting
    from a fresh adapter and the Tier 1 head; the optimizer state is not restored."""
    t2 = cfg.tier2
    train_ex = ds.train[:limit or t2.max_train_examples]
    eval_ex = ds.eval[:EVAL_SUBSET]
    if resume is not None:
        if tier2_head_kind(resume) != t2.head:
            raise ValueError(f"checkpoint {resume} has head {tier2_head_kind(resume)!r}, config asks for {t2.head!r}")
        log.info("resuming adapter + head from %s", resume)
        peft_model = load_trainable_adapter(backbone, resume / "adapter")
    else:
        peft_model = wrap_lora(backbone, t2)
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.enable_input_require_grads()
    device = next(peft_model.parameters()).device
    if t2.head == "span":
        head = (load_span_head(resume / "head.pt", ds.catalog) if resume else
                _init_span_head(cfg, ds.catalog, backbone.config.hidden_size)).to(device)
        model = SpanActionRankModel(tokenizer, peft_model, head, cfg.model, ds.catalog)
    elif t2.head == "table":
        head = (load_head(resume / "head.pt", ds.catalog) if resume else
                _init_head(cfg, ds.catalog, backbone.config.hidden_size)).to(device)
        model = ActionRankModel(tokenizer, peft_model, head, cfg.model)
    else:
        raise ValueError(f"tier2.head must be 'table' or 'span', got {t2.head!r}")
    params = [p for p in peft_model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=t2.lr)
    log.info("device %s; trainable fraction of backbone: %.4f; train examples %d",
             device, trainable_fraction(peft_model), len(train_ex))
    history: dict[str, list] = {"train_loss": [], "step_losses": [], "optimizer_steps_per_epoch": [],
                                "eval_top1": [_eval_top1(model, eval_ex, ds.catalog, cfg)]}
    log.info("eval top1 before training: %.4f", history["eval_top1"][0])
    generator = torch.Generator().manual_seed(cfg.data.split_seed)
    started = time.perf_counter()
    for epoch in range(t2.epochs):
        step_losses = _train_epoch(model, optimizer, train_ex, ds.catalog, cfg, generator)
        loss = sum(step_losses) / max(len(step_losses), 1)
        top1 = _eval_top1(model, eval_ex, ds.catalog, cfg)
        history["train_loss"].append(loss)
        history["step_losses"].append(step_losses)
        history["optimizer_steps_per_epoch"].append(-(-len(step_losses) // t2.grad_accum))
        history["eval_top1"].append(top1)
        log.info("epoch %d loss %.4f eval top1 %.4f", epoch + 1, loss, top1)
    out = Path(t2.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out / "adapter"))
    if t2.head == "span":
        save_span_head(head.cpu(), out / "head.pt", ds.catalog)
    else:
        save_head(head.cpu(), out / "head.pt", ds.catalog)
    (out / "meta.json").write_text(json.dumps({"head": t2.head, "pooling": cfg.model.pooling,
                                               "resumed_from": str(resume) if resume else None}))
    (out / "history.json").write_text(json.dumps({"history": history, "n_train": len(train_ex), "n_eval_subset": len(eval_ex),
                                                  "train_seconds": time.perf_counter() - started}, indent=1))
    return out


def load_tier2(cfg: Config, catalog: Catalog, tokenizer=None, backbone=None) -> ActionRankModel:
    from peft import PeftModel

    out = Path(cfg.tier2.checkpoint_dir)
    if tokenizer is None or backbone is None:
        tokenizer, backbone = load_backbone(cfg.model)
    peft_model = PeftModel.from_pretrained(backbone, str(out / "adapter")).eval()
    if tier2_head_kind(out) == "span":
        return SpanActionRankModel(tokenizer, peft_model, load_span_head(out / "head.pt", catalog), cfg.model, catalog)
    return ActionRankModel(tokenizer, peft_model, load_head(out / "head.pt", catalog), cfg.model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2: LoRA fine-tune backbone + head")
    parser.add_argument("--limit", type=int, default=None, help="override tier2.max_train_examples")
    parser.add_argument("--resume", type=Path, default=None, help="continue from this Tier 2 checkpoint dir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    ds = load_dataset(cfg)
    tokenizer, backbone = load_backbone(cfg.model)
    out = train_tier2(ds, cfg, tokenizer, backbone, limit=args.limit, resume=args.resume)
    print(f"saved tier 2 checkpoint to {out}")
    print((out / "history.json").read_text())


if __name__ == "__main__":
    main()
