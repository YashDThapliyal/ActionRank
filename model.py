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


HEAD_ARCHITECTURE = "residual-cosine-v1"  # bump whenever ScoringHead.forward changes; old checkpoints are rejected


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


# --------------------------------------------------------------------------- span pooling

SPAN_HEAD_ARCHITECTURE = "span-cosine-v1"


def span_token_mask(offsets: Tensor, spans: Sequence[tuple[int, int]]) -> Tensor:
    """[K, T] bool: token t belongs to span k when their character ranges overlap.

    Raises if any span covers zero tokens (e.g. the candidate line was truncated away): a zero-token span
    must never be silently scored.
    """
    span_t = torch.as_tensor(list(spans), dtype=offsets.dtype, device=offsets.device)
    starts, ends = offsets[:, 0].unsqueeze(0), offsets[:, 1].unsqueeze(0)
    mask = (starts < span_t[:, 1:2]) & (ends > span_t[:, 0:1])
    empty = (mask.sum(dim=1) == 0).nonzero().flatten().tolist()
    if empty:
        raise ValueError(f"candidate span(s) {empty} cover zero tokens; prompt was truncated past the catalog")
    return mask


def pool_spans(hidden: Tensor, token_mask: Tensor) -> Tensor:
    """Mean of hidden[T, D] over each row of token_mask[K, T] -> [K, D] float32."""
    weights = token_mask.to(hidden.device, torch.float32)
    return (weights @ hidden.to(torch.float32)) / weights.sum(dim=1, keepdim=True).clamp(min=1.0)


def encode_with_spans(tokenizer, backbone, prompts: Sequence[str], spans: Sequence[Sequence[tuple[int, int]]],
                      cfg: ModelConfig, grad: bool = False, pool_on_cpu: bool = True) -> tuple[Tensor, list[Tensor]]:
    """One prefill per batch; returns the pooled query vectors [B, D] and, per example, its candidate
    span vectors [K_i, D]. Span pooling runs on CPU from a single copy of the hidden states unless
    gradients are needed (on MPS the dozen tiny ops it takes cost more than the copy)."""
    device = next(backbone.parameters()).device
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    batch = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg.max_prompt_tokens, return_offsets_mapping=True)
    offsets = batch.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        hidden = backbone.get_decoder()(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
    query = pool_hidden(hidden, inputs["attention_mask"], cfg.pooling)
    source = hidden.detach().to(torch.float32).cpu() if (pool_on_cpu and not grad) else hidden
    tools = [pool_spans(source[i], span_token_mask(offsets[i], spans[i])) for i in range(len(prompts))]
    return query, tools


IGNORE_INDEX = -100


def build_joint_batch(tokenizer, prompts: Sequence[str], answers: Sequence[str], max_tokens: int,
                      lm_scope: str = "all") -> dict[str, Tensor]:
    """Prompt (left-truncated to max_tokens, as in inference) followed by the answer tokens and EOS, right padded.

    Returns input_ids, attention_mask, labels (IGNORE_INDEX where not supervised), prompt_len [B] and the
    prompt token offsets [B, Tp, 2] (zero beyond the prompt) so span pooling can stay on prompt positions.
    lm_scope 'all' supervises every real token (GenRec's LM objective over inputs and outputs); 'answer'
    supervises only the answer and EOS."""
    if lm_scope not in ("all", "answer"):
        raise ValueError(f"lm_scope must be 'all' or 'answer', got {lm_scope!r}")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    pb = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=max_tokens,
                   return_offsets_mapping=True)
    offsets = pb["offset_mapping"]
    prompt_len = pb["attention_mask"].sum(dim=1)
    answer_ids = [tokenizer(a, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id] for a in answers]
    width = max(int(prompt_len[i]) + len(answer_ids[i]) for i in range(len(prompts)))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    attention = torch.zeros((len(prompts), width), dtype=torch.long)
    labels = torch.full((len(prompts), width), IGNORE_INDEX, dtype=torch.long)
    for i in range(len(prompts)):
        plen = int(prompt_len[i])
        seq = pb["input_ids"][i, :plen].tolist() + answer_ids[i]
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attention[i, :len(seq)] = 1
        if lm_scope == "all":
            labels[i, :len(seq)] = torch.tensor(seq)
        else:
            labels[i, plen:len(seq)] = torch.tensor(answer_ids[i])
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels,
            "prompt_len": prompt_len, "offset_mapping": offsets}


def encode_with_spans_lm(tokenizer, backbone, prompts: Sequence[str], spans: Sequence[Sequence[tuple[int, int]]],
                         answers: Sequence[str], cfg: ModelConfig, lm_scope: str = "all") -> tuple[Tensor, list[Tensor], Tensor]:
    """One forward pass over prompt + answer. The pooled query and the candidate span vectors are read from
    prompt positions only (causal attention keeps them independent of the answer), and the LM head gives the
    next-token loss. Returns (query [B, D], per-example span vectors, lm_loss scalar)."""
    device = next(backbone.parameters()).device
    batch = build_joint_batch(tokenizer, prompts, answers, cfg.max_prompt_tokens, lm_scope)
    input_ids, attention = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    hidden = backbone.get_decoder()(input_ids=input_ids, attention_mask=attention).last_hidden_state
    total = hidden.shape[1]
    prompt_mask = torch.zeros_like(attention)
    for i, plen in enumerate(batch["prompt_len"].tolist()):
        prompt_mask[i, :plen] = 1
    query = pool_hidden(hidden, prompt_mask, cfg.pooling)
    tools = []
    for i in range(len(prompts)):
        mask = span_token_mask(batch["offset_mapping"][i], spans[i])
        padded = torch.zeros(mask.shape[0], total, dtype=mask.dtype)
        padded[:, :mask.shape[1]] = mask
        tools.append(pool_spans(hidden[i], padded))
    logits = backbone.get_output_embeddings()(hidden)
    shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1]).to(torch.float32)
    shift_labels = batch["labels"][:, 1:].reshape(-1).to(device)
    lm_loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
    return query, tools, lm_loss


class SpanScoringHead(nn.Module):
    """score(q, tool_k) = cos(q + mlp(q), t_k + mlp'(t_k)) / temperature for the K candidate spans of a prompt,
    scattered into catalog-width logits (-inf everywhere else). Both projections are residual with
    zero-initialised output layers, so an untrained head is plain cosine similarity."""

    def __init__(self, hidden_dim: int, head_hidden: int, temperature: float) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.proj = nn.Sequential(nn.Linear(hidden_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, hidden_dim))
        self.tool_proj = nn.Sequential(nn.Linear(hidden_dim, head_hidden), nn.GELU(), nn.Linear(head_hidden, hidden_dim))
        for block in (self.proj, self.tool_proj):
            nn.init.zeros_(block[2].weight)
            nn.init.zeros_(block[2].bias)
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.head_hidden = head_hidden

    def forward(self, q: Tensor, tools: Tensor, tool_mask: Tensor, tool_idx: Tensor, num_tools: int) -> Tensor:
        q32, t32 = q.to(torch.float32), tools.to(torch.float32)
        qn = nn.functional.normalize(q32 + self.proj(q32), dim=-1)
        tn = nn.functional.normalize(t32 + self.tool_proj(t32), dim=-1)
        scores = torch.einsum("bd,bkd->bk", qn, tn) / self.temperature
        scores = scores.masked_fill(~tool_mask.to(scores.device), float("-inf"))
        logits = torch.full((q.shape[0], num_tools), float("-inf"), device=scores.device)
        # amax so a padded slot (idx 0, score -inf) can never clobber a real candidate at the same index
        return logits.scatter_reduce(1, tool_idx.to(scores.device), scores, reduce="amax", include_self=True)


def save_span_head(head: SpanScoringHead, path: Path, catalog: Catalog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "hidden_dim": head.hidden_dim, "head_hidden": head.head_hidden,
                "temperature": head.temperature, "catalog_sha": catalog_fingerprint(catalog),
                "architecture": SPAN_HEAD_ARCHITECTURE}, path)


def load_span_head(path: Path, catalog: Catalog) -> SpanScoringHead:
    saved = torch.load(path, map_location="cpu")
    if saved.get("architecture") != SPAN_HEAD_ARCHITECTURE:
        raise ValueError(f"head at {path} has architecture {saved.get('architecture')!r}, expected "
                         f"{SPAN_HEAD_ARCHITECTURE!r}; re-run training")
    if saved.get("catalog_sha") != catalog_fingerprint(catalog):
        raise ValueError(f"head at {path} was trained on a different catalog; re-run training")
    head = SpanScoringHead(saved["hidden_dim"], saved["head_hidden"], saved["temperature"])
    head.load_state_dict(saved["state_dict"])
    return head.eval()


def pad_tool_vectors(tools: Sequence[Tensor], examples: Sequence[Example], catalog: Catalog,
                     width: int | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Pad per-example [K_i, D] span vectors to [B, K, D] with a bool mask and catalog indices."""
    width = width or max(t.shape[0] for t in tools)
    dim, device = tools[0].shape[1], tools[0].device
    padded = torch.zeros(len(tools), width, dim, dtype=torch.float32, device=device)
    mask = torch.zeros(len(tools), width, dtype=torch.bool)
    idx = torch.zeros(len(tools), width, dtype=torch.long)
    for row, (vecs, ex) in enumerate(zip(tools, examples)):
        k = vecs.shape[0]
        padded[row, :k] = vecs.to(torch.float32)
        mask[row, :k] = True
        idx[row, :k] = torch.tensor([catalog.index(c) for c in ex.candidates])
    return padded, mask, idx


class SpanActionRankModel(nn.Module):
    """Backbone + SpanScoringHead: candidates are scored from their own lines inside the prompt."""

    span_pool_device = "cpu"

    def __init__(self, tokenizer, backbone, head: SpanScoringHead, cfg: ModelConfig, catalog: Catalog) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.head = head.to(next(backbone.parameters()).device)
        self.cfg = cfg
        self.catalog = catalog

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def forward(self, examples: Sequence[Example], verbalize_cfg, grad: bool = False) -> Tensor:
        from verbalize import build_prompt_with_spans

        rendered = [build_prompt_with_spans(ex, self.catalog, verbalize_cfg) for ex in examples]
        q, tools = encode_with_spans(self.tokenizer, self.backbone, [r[0] for r in rendered], [r[1] for r in rendered],
                                     self.cfg, grad=grad, pool_on_cpu=not grad)
        padded, mask, idx = pad_tool_vectors(tools, examples, self.catalog)
        head_device = self.head.proj[0].weight.device
        return self.head(q.to(head_device), padded.to(head_device), mask, idx, len(self.catalog))

    def forward_with_lm(self, examples: Sequence[Example], verbalize_cfg, lm_scope: str = "all") -> tuple[Tensor, Tensor]:
        """Training-only joint pass: catalog-width ranking logits plus the next-token loss over the
        verbalized prompt and the label (GenRec's Phase 2 objective). Inference never calls this."""
        from verbalize import build_prompt_with_spans

        rendered = [build_prompt_with_spans(ex, self.catalog, verbalize_cfg) for ex in examples]
        answers = [f" {ex.label}" for ex in examples]
        q, tools, lm_loss = encode_with_spans_lm(self.tokenizer, self.backbone, [r[0] for r in rendered],
                                                 [r[1] for r in rendered], answers, self.cfg, lm_scope)
        padded, mask, idx = pad_tool_vectors(tools, examples, self.catalog)
        head_device = self.head.proj[0].weight.device
        return self.head(q.to(head_device), padded.to(head_device), mask, idx, len(self.catalog)), lm_loss

    @torch.no_grad()
    def rank_example(self, example: Example, catalog: Catalog, k: int, verbalize_cfg=None) -> tuple[int, ...]:
        if verbalize_cfg is None:
            from config import load_config

            verbalize_cfg = load_config().verbalize
        logits = self.forward([example], verbalize_cfg)
        logits = logits.masked_fill(~build_candidate_mask([example], catalog).to(logits.device), float("-inf"))
        return ActionRankModel.rank_logits(logits, k)[0]


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
                "temperature": head.temperature, "catalog_sha": catalog_fingerprint(catalog),
                "architecture": HEAD_ARCHITECTURE}, path)


def load_head(path: Path, catalog: Catalog) -> ScoringHead:
    """Load a saved head and verify it was trained on exactly this catalog (same tools, same order)."""
    saved = torch.load(path, map_location="cpu")
    if saved.get("architecture") != HEAD_ARCHITECTURE:
        raise ValueError(f"head at {path} has architecture {saved.get('architecture')!r}, expected "
                         f"{HEAD_ARCHITECTURE!r}; re-run training")
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

    @torch.no_grad()
    def rank_example(self, example: Example, catalog: Catalog, k: int, verbalize_cfg=None) -> tuple[int, ...]:
        """Catalog indices of the top-k candidates for one example (prompt built here)."""
        from verbalize import build_prompt

        if verbalize_cfg is None:
            from config import load_config

            verbalize_cfg = load_config().verbalize
        prompt = build_prompt(example, catalog, verbalize_cfg)
        return self.rank([prompt], build_candidate_mask([example], catalog).to(self.device), k)[0]
