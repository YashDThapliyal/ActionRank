"""Backbone encoder (prefill only), pooling, and catalog-aware scoring head."""
from __future__ import annotations

import hashlib
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


def encode_tool_descriptions(tokenizer, backbone, catalog: Catalog, cfg: ModelConfig, batch_size: int) -> Tensor:
    """Pooled backbone vector for every catalog tool's `name: description`, in catalog order ([num_tools, D])."""
    prompts = [f"Tool: {t.name}\nDescription: {t.description}" for t in catalog.tools]
    chunks = [encode_prompts(tokenizer, backbone, prompts[i:i + batch_size], cfg).detach().cpu()
              for i in range(0, len(prompts), batch_size)]
    return torch.cat(chunks)


class ScoringHead(nn.Module):
    """score(h, tool) = cos(h + mlp(h), E[tool]) / temperature, masked to candidates.

    The projection is residual with a zero-initialised output layer, so an untrained head scores tools by
    plain cosine similarity between the prompt vector and the tool vector (meaningful when E is initialised
    from encoded tool descriptions).
    """

    def __init__(self, num_tools: int, hidden_dim: int, head_hidden: int, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.tool_embedding = nn.Embedding(num_tools, hidden_dim)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, hidden_dim))
        self.temperature = temperature
        nn.init.normal_(self.tool_embedding.weight, std=0.02)
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def init_tool_embeddings(self, vectors: Tensor) -> None:
        """Overwrite the tool table with externally computed vectors (e.g. encoded descriptions)."""
        expected = tuple(self.tool_embedding.weight.shape)
        if tuple(vectors.shape) != expected:
            raise ValueError(f"tool vectors have shape {tuple(vectors.shape)}, expected {expected}")
        with torch.no_grad():
            self.tool_embedding.weight.copy_(vectors.to(self.tool_embedding.weight.dtype))

    def forward(self, h: Tensor, candidate_mask: Tensor | None) -> Tensor:
        h32 = h.to(torch.float32)
        query = nn.functional.normalize(h32 + self.proj(h32), dim=-1)
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


def catalog_fingerprint(catalog: Catalog) -> str:
    """Stable hash of the catalog's tool order, so a head is only used with the catalog it was trained on."""
    return hashlib.sha256("\n".join(catalog.names).encode()).hexdigest()


def save_head(head: ScoringHead, path: Path, catalog: Catalog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "num_tools": head.tool_embedding.num_embeddings,
                "hidden_dim": head.tool_embedding.embedding_dim, "head_hidden": head.proj[0].out_features,
                "temperature": head.temperature, "catalog_sha": catalog_fingerprint(catalog)}, path)


def load_head(path: Path, catalog: Catalog) -> ScoringHead:
    """Load a saved head and verify it was trained on exactly this catalog (same tools, same order)."""
    saved = torch.load(path, map_location="cpu")
    if saved.get("catalog_sha") != catalog_fingerprint(catalog):
        raise ValueError(f"head at {path} was trained on a different catalog; re-run training")
    head = ScoringHead(saved["num_tools"], saved["hidden_dim"], saved["head_hidden"], saved["temperature"])
    head.load_state_dict(saved["state_dict"])
    return head.eval()


class ActionRankModel(nn.Module):
    """Backbone + head. forward() returns catalog logits for a batch of prompts."""

    def __init__(self, tokenizer, backbone, head: ScoringHead, cfg: ModelConfig) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.head = head.to(next(backbone.parameters()).device)
        self.cfg = cfg

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def forward(self, prompts: Sequence[str], candidate_mask: Tensor | None, grad: bool = False) -> Tensor:
        h = encode_prompts(self.tokenizer, self.backbone, prompts, self.cfg, grad=grad)
        return self.head(h, candidate_mask)

    @staticmethod
    def rank_logits(logits: Tensor, k: int) -> list[tuple[int, ...]]:
        """Top-k catalog indices per row, excluding masked (-inf) tools even when fewer than k remain."""
        k = min(k, logits.shape[-1])
        values, indices = logits.topk(k, dim=-1)
        return [tuple(int(i) for i, v in zip(row_idx, row_val) if torch.isfinite(v))
                for row_idx, row_val in zip(indices.cpu(), values.cpu())]

    @torch.no_grad()
    def rank(self, prompts: Sequence[str], candidate_mask: Tensor | None, k: int) -> list[tuple[int, ...]]:
        return self.rank_logits(self.forward(prompts, candidate_mask), k)
