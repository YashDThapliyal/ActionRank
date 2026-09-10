"""Backbone encoder (prefill only), pooling, and catalog-aware scoring head."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from config import ModelConfig
from data import Catalog, Example

log = logging.getLogger(__name__)

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to CPU (with a warning) if unavailable."""
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested != "cpu":
        log.warning("device %r unavailable, falling back to cpu", requested)
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until queued kernels finish (needed for honest latency timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_backbone(cfg: ModelConfig):
    """Load tokenizer + causal LM in cfg.dtype on cfg.device, eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cfg.dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype {cfg.dtype!r}; choose from {sorted(_DTYPES)}")
    device = resolve_device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.backbone, dtype=_DTYPES[cfg.dtype])
    model.to(device)
    model.eval()
    return tokenizer, model


def pool_hidden(hidden: Tensor, attention_mask: Tensor, pooling: str) -> Tensor:
    """Pool [B,T,D] hidden states to [B,D] float32 using the attention mask (right padding)."""
    mask = attention_mask.to(hidden.device)
    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(torch.float32)
        summed = (hidden.to(torch.float32) * weights).sum(dim=1)
        return summed / weights.sum(dim=1).clamp(min=1.0)
    if pooling == "last":
        last_idx = (mask.sum(dim=1) - 1).clamp(min=0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last_idx].to(torch.float32)
    raise ValueError(f"unknown pooling {pooling!r}; choose 'mean' or 'last'")


def tokenize_prompts(tokenizer, prompts: Sequence[str], max_tokens: int, device: torch.device) -> dict[str, Tensor]:
    """Right-padded, left-truncated batch so the catalog and 'Next tool:' suffix survive truncation."""
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    batch = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
    return {k: v.to(device) for k, v in batch.items()}


def encode_prompts(tokenizer, model, prompts: Sequence[str], cfg: ModelConfig, grad: bool = False) -> Tensor:
    """Prefill-only forward through the decoder (no LM head), pooled to [B,D] float32."""
    device = next(model.parameters()).device
    batch = tokenize_prompts(tokenizer, prompts, cfg.max_prompt_tokens, device)
    with torch.set_grad_enabled(grad):
        out = model.get_decoder()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return pool_hidden(out.last_hidden_state, batch["attention_mask"], cfg.pooling)


class ScoringHead(nn.Module):
    """score(h, tool) = cos(proj(h), E[tool]) / temperature, masked to candidates."""

    def __init__(self, num_tools: int, hidden_dim: int, head_hidden: int, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.tool_embedding = nn.Embedding(num_tools, hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, hidden_dim))
        self.temperature = temperature
        nn.init.normal_(self.tool_embedding.weight, std=0.02)

    def forward(self, h: Tensor, candidate_mask: Tensor | None) -> Tensor:
        query = nn.functional.normalize(self.proj(h.to(torch.float32)), dim=-1)
        keys = nn.functional.normalize(self.tool_embedding.weight, dim=-1)
        logits = query @ keys.T / self.temperature
        if candidate_mask is None:
            return logits
        return logits.masked_fill(~candidate_mask.to(logits.device), float("-inf"))


def build_candidate_mask(examples: Sequence[Example], catalog: Catalog) -> Tensor:
    mask = torch.zeros(len(examples), len(catalog), dtype=torch.bool)
    for row, ex in enumerate(examples):
        for name in ex.candidates:
            mask[row, catalog.index(name)] = True
    return mask


def labels_tensor(examples: Sequence[Example], catalog: Catalog) -> Tensor:
    return torch.tensor([catalog.index(ex.label) for ex in examples], dtype=torch.long)


def save_head(head: ScoringHead, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "num_tools": head.tool_embedding.num_embeddings,
                "hidden_dim": head.tool_embedding.embedding_dim, "head_hidden": head.proj[0].out_features,
                "temperature": head.temperature}, path)


def load_head(path: Path) -> ScoringHead:
    saved = torch.load(path, map_location="cpu")
    head = ScoringHead(saved["num_tools"], saved["hidden_dim"], saved["head_hidden"], saved["temperature"])
    head.load_state_dict(saved["state_dict"])
    return head.eval()


class ActionRankModel(nn.Module):
    """Backbone + head. forward() returns catalog logits for a batch of prompts."""

    def __init__(self, tokenizer, backbone, head: ScoringHead, cfg: ModelConfig) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.head = head
        self.cfg = cfg

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def forward(self, prompts: Sequence[str], candidate_mask: Tensor | None, grad: bool = False) -> Tensor:
        h = encode_prompts(self.tokenizer, self.backbone, prompts, self.cfg, grad=grad)
        return self.head(h, candidate_mask)

    @torch.no_grad()
    def rank(self, prompts: Sequence[str], candidate_mask: Tensor | None, k: int) -> Tensor:
        return self.forward(prompts, candidate_mask).topk(k, dim=-1).indices
