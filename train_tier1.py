"""Tier 1: run the frozen backbone once per example, cache pooled vectors, train only the head."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from tqdm import tqdm

from config import Config, load_config
from data import Catalog, Example, load_dataset
from model import catalog_fingerprint
from model import (ScoringHead, SpanScoringHead, build_candidate_mask, encode_prompts, encode_tool_descriptions,
                   encode_with_spans, labels_tensor, load_backbone, pad_tool_vectors, resolve_device, save_head,
                   save_span_head)
from verbalize import build_prompt, build_prompt_with_spans

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedSplit:
    h: Tensor            # [N, D] float32 pooled backbone states
    labels: Tensor       # [N] long catalog indices
    candidate_mask: Tensor  # [N, num_tools] bool
    query_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.h.shape[0])


def load_cached(path: Path) -> CachedSplit:
    saved = torch.load(path, map_location="cpu")
    return CachedSplit(h=saved["h"], labels=saved["labels"], candidate_mask=saved["candidate_mask"],
                       query_ids=tuple(saved["query_ids"]))


def _batches(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def cache_fingerprint(prompts: Sequence[str], examples: Sequence[Example], catalog: Catalog, cfg: Config) -> str:
    """Hash of everything that determines the cached vectors: model settings, prompts, labels, candidates."""
    digest = hashlib.sha256()
    m = cfg.model
    digest.update(f"{m.backbone}|{m.dtype}|{m.pooling}|{m.max_prompt_tokens}|{catalog_fingerprint(catalog)}".encode())
    for prompt, ex in zip(prompts, examples):
        digest.update(prompt.encode())
        digest.update(f"\x00{ex.label}\x00{'|'.join(ex.candidates)}\x01".encode())
    return digest.hexdigest()


def _cached_matches(path: Path, fingerprint: str) -> bool:
    saved = torch.load(path, map_location="cpu")
    return saved.get("fingerprint") == fingerprint


def cache_split(examples: Sequence[Example], catalog: Catalog, tokenizer, backbone, cfg: Config,
                path: Path, batch_size: int | None = None, refresh: bool = False) -> CachedSplit:
    """Encode every example's prompt with the frozen backbone and cache the pooled vectors."""
    prompts = [build_prompt(ex, catalog, cfg.verbalize) for ex in examples]
    fingerprint = cache_fingerprint(prompts, examples, catalog, cfg)
    if path.exists() and not refresh:
        if _cached_matches(path, fingerprint):
            log.info("reusing cache %s (%d rows)", path, len(examples))
            return load_cached(path)
        log.warning("cache %s does not match the current inputs/config; re-encoding", path)
    size = batch_size or cfg.tier1.cache_batch_size
    chunks: list[Tensor] = []
    for batch in tqdm(list(_batches(prompts, size)), desc=f"encode {path.stem}", leave=False):
        chunks.append(encode_prompts(tokenizer, backbone, batch, cfg.model).detach().cpu())
    split = CachedSplit(h=torch.cat(chunks) if chunks else torch.empty(0, 0),
                        labels=labels_tensor(examples, catalog),
                        candidate_mask=build_candidate_mask(examples, catalog),
                        query_ids=tuple(ex.query_id for ex in examples))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"h": split.h, "labels": split.labels, "candidate_mask": split.candidate_mask,
                "query_ids": list(split.query_ids), "fingerprint": fingerprint}, path)
    return split


def cache_tool_vectors(catalog: Catalog, tokenizer, backbone, cfg: Config, path: Path, refresh: bool = False) -> Tensor:
    """Encode (and cache) every catalog tool's description with the frozen backbone."""
    m = cfg.model
    text = (f"{m.backbone}|{m.dtype}|{m.pooling}|{m.max_prompt_tokens}|{catalog_fingerprint(catalog)}|"
            + "\n".join(f"{t.name}: {t.description}" for t in catalog.tools))
    fingerprint = hashlib.sha256(text.encode()).hexdigest()
    if path.exists() and not refresh and _cached_matches(path, fingerprint):
        log.info("reusing tool vectors %s", path)
        return torch.load(path, map_location="cpu")["vectors"]
    vectors = encode_tool_descriptions(tokenizer, backbone, catalog, m, cfg.tier1.cache_batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vectors, "fingerprint": fingerprint}, path)
    return vectors


def _select(split: CachedSplit, rows: Tensor) -> CachedSplit:
    return CachedSplit(h=split.h[rows], labels=split.labels[rows], candidate_mask=split.candidate_mask[rows],
                       query_ids=tuple(split.query_ids[int(i)] for i in rows))


def split_cached(split: CachedSplit, fraction: float, seed: int) -> tuple[CachedSplit, CachedSplit]:
    """Hold out `fraction` of the *trajectories* (query ids) of a cached split for validation."""
    qids = sorted(set(split.query_ids))
    order = torch.randperm(len(qids), generator=torch.Generator().manual_seed(seed)).tolist()
    held = {qids[i] for i in order[:round(len(qids) * fraction)]}
    is_val = torch.tensor([q in held for q in split.query_ids], dtype=torch.bool)
    return _select(split, (~is_val).nonzero().flatten()), _select(split, is_val.nonzero().flatten())


def topk_accuracy(logits: Tensor, labels: Tensor, k: int) -> float:
    k = min(k, logits.shape[-1])
    hits = (logits.topk(k, dim=-1).indices == labels.unsqueeze(-1)).any(dim=-1)
    return int(hits.sum().item()) / len(labels) if len(labels) else 0.0


@torch.no_grad()
def evaluate_head(head: ScoringHead, split: CachedSplit, device: torch.device, batch_size: int) -> dict[str, float]:
    head.eval()
    logits = torch.cat([head(split.h[i:i + batch_size].to(device), split.candidate_mask[i:i + batch_size].to(device)).cpu()
                        for i in range(0, len(split), batch_size)])
    return {"top1": topk_accuracy(logits, split.labels, 1), "top5": topk_accuracy(logits, split.labels, 5)}


def train_head(train: CachedSplit, validation: CachedSplit, cfg: Config, num_tools: int,
               hidden_dim: int, tool_init: Tensor | None = None) -> tuple[ScoringHead, dict]:
    """AdamW on the head only; cross-entropy over candidate-masked catalog logits.

    `tool_init` (encoded tool descriptions) seeds the embedding table so tools unseen in training still
    rank sensibly. The returned head is the epoch with the best top-1 on `validation`, which must be carved
    out of the training trajectories (never the held-out eval split).
    """
    t1, device = cfg.tier1, resolve_device(cfg.model.device)
    head = ScoringHead(num_tools, hidden_dim, cfg.model.head_hidden, cfg.model.score_temperature)
    if tool_init is not None:
        head.init_tool_embeddings(tool_init)
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=t1.lr, weight_decay=t1.weight_decay)
    history: dict = {"train_loss": [], "eval_top1": [], "eval_top5": [],
                     "eval_top1_before_training": evaluate_head(head, validation, device, t1.batch_size)["top1"]}
    log.info("validation top1 before training: %.4f", history["eval_top1_before_training"])
    best_state, best_top1 = copy.deepcopy(head.state_dict()), history["eval_top1_before_training"]
    generator = torch.Generator().manual_seed(cfg.data.split_seed)
    for epoch in range(t1.epochs):
        head.train()
        order, losses = torch.randperm(len(train), generator=generator), []
        for idx in _batches(order, t1.batch_size):
            logits = head(train.h[idx].to(device), train.candidate_mask[idx].to(device))
            loss = torch.nn.functional.cross_entropy(logits, train.labels[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scores = evaluate_head(head, validation, device, t1.batch_size)
        history["train_loss"].append(sum(losses) / max(len(losses), 1))
        history["eval_top1"].append(scores["top1"])
        history["eval_top5"].append(scores["top5"])
        log.info("epoch %d loss %.4f eval top1 %.4f top5 %.4f", epoch + 1, history["train_loss"][-1], scores["top1"], scores["top5"])
        if scores["top1"] > best_top1:
            best_top1, best_state = scores["top1"], copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    history["best_top1"] = best_top1
    return head.eval().cpu(), history


# --------------------------------------------------------------------------- span head (tier1.head: span)

@dataclass(frozen=True)
class CachedSpanSplit:
    q: Tensor           # [N, D] pooled prompt vectors
    tools: Tensor       # [N, K, D] pooled candidate-span vectors (zero padded)
    tool_mask: Tensor   # [N, K] bool
    tool_idx: Tensor    # [N, K] catalog indices (0 where padded; masked out by tool_mask)
    labels: Tensor      # [N] catalog indices
    query_ids: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.q.shape[0])


_SPAN_FIELDS = ("q", "tools", "tool_mask", "tool_idx", "labels")


def load_cached_spans(path: Path) -> CachedSpanSplit:
    saved = torch.load(path, map_location="cpu")
    return CachedSpanSplit(**{f: saved[f] for f in _SPAN_FIELDS}, query_ids=tuple(saved["query_ids"]))


def cache_span_split(examples: Sequence[Example], catalog: Catalog, tokenizer, backbone, cfg: Config, path: Path,
                     batch_size: int | None = None, refresh: bool = False, progress=None) -> CachedSpanSplit:
    """Prefill every prompt once; cache the pooled query vector and each candidate's span vector.

    `progress(n_done, seconds)` is called after every batch and may raise to abort the run early."""
    rendered = [build_prompt_with_spans(ex, catalog, cfg.verbalize) for ex in examples]
    fingerprint = "span|" + cache_fingerprint([r[0] for r in rendered], examples, catalog, cfg)
    if path.exists() and not refresh:
        if _cached_matches(path, fingerprint):
            log.info("reusing span cache %s (%d rows)", path, len(examples))
            return load_cached_spans(path)
        log.warning("span cache %s does not match the current inputs/config; re-encoding", path)
    size = batch_size or cfg.tier1.cache_batch_size
    width = max((len(ex.candidates) for ex in examples), default=0)
    q_chunks, tool_chunks, mask_chunks, idx_chunks = [], [], [], []
    started = time.perf_counter()
    batches = list(_batches(list(range(len(examples))), size))
    for batch in tqdm(batches, desc=f"encode {path.stem}", leave=False):
        rows = [examples[i] for i in batch]
        q, tools = encode_with_spans(tokenizer, backbone, [rendered[i][0] for i in batch], [rendered[i][1] for i in batch], cfg.model)
        padded, mask, idx = pad_tool_vectors(tools, rows, catalog, width)
        q_chunks.append(q.detach().cpu()); tool_chunks.append(padded.detach().cpu().half()); mask_chunks.append(mask); idx_chunks.append(idx)
        if progress is not None:
            progress(batch[-1] + 1, time.perf_counter() - started)
    split = CachedSpanSplit(q=torch.cat(q_chunks) if q_chunks else torch.empty(0, 0),
                            tools=torch.cat(tool_chunks) if tool_chunks else torch.empty(0, 0, 0),
                            tool_mask=torch.cat(mask_chunks) if mask_chunks else torch.empty(0, 0, dtype=torch.bool),
                            tool_idx=torch.cat(idx_chunks) if idx_chunks else torch.empty(0, 0, dtype=torch.long),
                            labels=labels_tensor(examples, catalog), query_ids=tuple(ex.query_id for ex in examples))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**{f: getattr(split, f) for f in _SPAN_FIELDS}, "query_ids": list(split.query_ids), "fingerprint": fingerprint}, path)
    return split


def _select_spans(split: CachedSpanSplit, rows: Tensor) -> CachedSpanSplit:
    return CachedSpanSplit(**{f: getattr(split, f)[rows] for f in _SPAN_FIELDS},
                           query_ids=tuple(split.query_ids[int(i)] for i in rows))


def split_cached_spans(split: CachedSpanSplit, fraction: float, seed: int) -> tuple[CachedSpanSplit, CachedSpanSplit]:
    """Hold out `fraction` of the trajectories (query ids) for validation."""
    qids = sorted(set(split.query_ids))
    order = torch.randperm(len(qids), generator=torch.Generator().manual_seed(seed)).tolist()
    held = {qids[i] for i in order[:round(len(qids) * fraction)]}
    is_val = torch.tensor([q in held for q in split.query_ids], dtype=torch.bool)
    return _select_spans(split, (~is_val).nonzero().flatten()), _select_spans(split, is_val.nonzero().flatten())


def _span_logits(head: SpanScoringHead, split: CachedSpanSplit, rows, device: torch.device, num_tools: int) -> Tensor:
    return head(split.q[rows].to(device), split.tools[rows].to(device), split.tool_mask[rows].to(device),
                split.tool_idx[rows].to(device), num_tools)


@torch.no_grad()
def evaluate_span_head(head: SpanScoringHead, split: CachedSpanSplit, device: torch.device, batch_size: int,
                       num_tools: int) -> dict[str, float]:
    head.eval()
    logits = torch.cat([_span_logits(head, split, slice(i, i + batch_size), device, num_tools).cpu()
                        for i in range(0, len(split), batch_size)])
    return {"top1": topk_accuracy(logits, split.labels, 1), "top5": topk_accuracy(logits, split.labels, 5)}


def train_span_head(train: CachedSpanSplit, validation: CachedSpanSplit, cfg: Config, num_tools: int,
                    hidden_dim: int) -> tuple[SpanScoringHead, dict]:
    """AdamW on the span head; CE over catalog-width logits (finite only at candidates); best epoch by validation."""
    t1, device = cfg.tier1, resolve_device(cfg.model.device)
    head = SpanScoringHead(hidden_dim, cfg.model.head_hidden, cfg.model.score_temperature).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=t1.lr, weight_decay=t1.weight_decay)
    before = evaluate_span_head(head, validation, device, t1.batch_size, num_tools)["top1"]
    history: dict = {"train_loss": [], "eval_top1": [], "eval_top5": [], "eval_top1_before_training": before}
    log.info("validation top1 before training: %.4f", before)
    best_state, best_top1 = copy.deepcopy(head.state_dict()), before
    generator = torch.Generator().manual_seed(cfg.data.split_seed)
    for epoch in range(t1.epochs):
        head.train()
        order, losses = torch.randperm(len(train), generator=generator), []
        for idx in _batches(order, t1.batch_size):
            logits = _span_logits(head, train, idx, device, num_tools)
            loss = torch.nn.functional.cross_entropy(logits, train.labels[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scores = evaluate_span_head(head, validation, device, t1.batch_size, num_tools)
        history["train_loss"].append(sum(losses) / max(len(losses), 1))
        history["eval_top1"].append(scores["top1"])
        history["eval_top5"].append(scores["top5"])
        log.info("epoch %d loss %.4f eval top1 %.4f top5 %.4f", epoch + 1, history["train_loss"][-1], scores["top1"], scores["top5"])
        if scores["top1"] > best_top1:
            best_top1, best_state = scores["top1"], copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)
    history["best_top1"] = best_top1
    return head.eval().cpu(), history


def run_span_tier1(cfg: Config, ds, tokenizer, backbone, limit: int | None = None, refresh: bool = False,
                   progress=None) -> dict:
    """Cache span vectors for train/eval, train the span head, save it, evaluate on the held-out cache."""
    train_ex, eval_ex = ds.train[:limit], ds.eval[:limit]
    cache_dir, suffix = Path(cfg.tier1.cache_dir), (f"_{limit}" if limit else "")
    started = time.perf_counter()
    train = cache_span_split(train_ex, ds.catalog, tokenizer, backbone, cfg, cache_dir / f"span_train{suffix}.pt", refresh=refresh, progress=progress)
    evaluation = cache_span_split(eval_ex, ds.catalog, tokenizer, backbone, cfg, cache_dir / f"span_eval{suffix}.pt", refresh=refresh, progress=progress)
    encode_seconds = time.perf_counter() - started
    train_rows, validation = split_cached_spans(train, 0.1, cfg.data.split_seed)
    head, history = train_span_head(train_rows, validation, cfg, len(ds.catalog), int(train.q.shape[1]))
    save_span_head(head, Path(cfg.tier1.span_checkpoint), ds.catalog)
    device = resolve_device(cfg.model.device)
    held_out = evaluate_span_head(head.to(device), evaluation, device, cfg.tier1.batch_size, len(ds.catalog))
    summary = {"history": history, "encode_seconds": encode_seconds, "n_train": len(train_rows), "n_validation": len(validation),
               "n_eval": len(evaluation), "best_validation_top1": history["best_top1"], "held_out_eval": held_out,
               "limit": limit, "head": "span"}
    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tier1_span_history.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 1: cache backbone vectors and train the scoring head")
    parser.add_argument("--refresh", action="store_true", help="re-encode even if a cache exists")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N train / N eval examples")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    ds = load_dataset(cfg)
    tokenizer, backbone = load_backbone(cfg.model)
    if cfg.tier1.head == "span":
        summary = run_span_tier1(cfg, ds, tokenizer, backbone, limit=args.limit, refresh=args.refresh)
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=1))
        return
    if cfg.tier1.head != "table":
        raise ValueError(f"tier1.head must be 'table' or 'span', got {cfg.tier1.head!r}")
    train_ex, eval_ex = ds.train[:args.limit], ds.eval[:args.limit]
    validation_fraction = 0.1
    cache_dir = Path(cfg.tier1.cache_dir)
    suffix = f"_{args.limit}" if args.limit else ""
    started = time.perf_counter()
    train = cache_split(train_ex, ds.catalog, tokenizer, backbone, cfg, cache_dir / f"train{suffix}.pt", refresh=args.refresh)
    evaluation = cache_split(eval_ex, ds.catalog, tokenizer, backbone, cfg, cache_dir / f"eval{suffix}.pt", refresh=args.refresh)
    encode_seconds = time.perf_counter() - started
    tool_init = None
    if cfg.model.tool_init == "text":
        tool_init = cache_tool_vectors(ds.catalog, tokenizer, backbone, cfg, cache_dir / "tools.pt", refresh=args.refresh)
    elif cfg.model.tool_init != "random":
        raise ValueError(f"model.tool_init must be 'text' or 'random', got {cfg.model.tool_init!r}")
    train_rows, validation = split_cached(train, validation_fraction, cfg.data.split_seed)
    head, history = train_head(train_rows, validation, cfg, len(ds.catalog), int(train.h.shape[1]), tool_init)
    save_head(head, Path(cfg.tier1.checkpoint), ds.catalog)
    held_out = evaluate_head(head.to(resolve_device(cfg.model.device)), evaluation, resolve_device(cfg.model.device), cfg.tier1.batch_size)
    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {"history": history, "encode_seconds": encode_seconds, "n_train": len(train_rows),
               "n_validation": len(validation), "n_eval": len(evaluation),
               "best_validation_top1": history["best_top1"], "held_out_eval": held_out, "limit": args.limit,
               "tool_init": cfg.model.tool_init}
    (results_dir / "tier1_history.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
