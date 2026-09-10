"""Prototype: per-decision latency of the current table head vs. in-prompt span pooling, per candidate count."""
import sys, time, statistics, collections
sys.path.insert(0, "/Users/yash/Documents/ActionRank")
import torch
from pathlib import Path
from config import load_config
from data import load_dataset
from model import load_backbone, load_head, ActionRankModel, build_candidate_mask, synchronize, pool_hidden
from verbalize import build_prompt, render_catalog

cfg = load_config(); ds = load_dataset(cfg)
tok, backbone = load_backbone(cfg.model)
model = ActionRankModel(tok, backbone, load_head(Path(cfg.tier1.checkpoint), ds.catalog), cfg.model)
dev = model.device; N = len(ds.catalog); D = backbone.config.hidden_size
proj = torch.nn.Linear(D, D).to(dev)  # stand-in for a trained query projection (cost only)

def current(ex):
    prompt = build_prompt(ex, ds.catalog, cfg.verbalize)
    mask = build_candidate_mask([ex], ds.catalog).to(dev)
    return model.rank([prompt], mask, 5)[0]

@torch.no_grad()
def span(ex):
    prompt = build_prompt(ex, ds.catalog, cfg.verbalize)
    # character spans of each candidate line (what verbalize.py would return directly)
    block = render_catalog(ex.candidates, ds.catalog, cfg.verbalize)
    base = prompt.index(block); spans = []; cursor = base
    for name in ex.candidates:
        line = f"- {name}: "; s = prompt.index(line, cursor); e = prompt.index("\n", s) if "\n" in prompt[s:] else len(prompt)
        spans.append((s, e)); cursor = e
    tok.padding_side = "right"; tok.truncation_side = "left"
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=cfg.model.max_prompt_tokens)
    offsets = enc.pop("offset_mapping")[0]
    enc = {k: v.to(dev) for k, v in enc.items()}
    hidden = backbone.get_decoder()(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[0]
    starts, ends = offsets[:, 0], offsets[:, 1]
    span_t = torch.tensor(spans)
    tok_mask = (starts.unsqueeze(0) < span_t[:, 1:2]) & (ends.unsqueeze(0) > span_t[:, 0:1])  # [K, T]
    tok_mask = tok_mask.to(dev).float()
    tool_vecs = (tok_mask @ hidden.float()) / tok_mask.sum(1, keepdim=True).clamp(min=1)          # [K, D]
    q = pool_hidden(hidden.unsqueeze(0), enc["attention_mask"], cfg.model.pooling)                  # [1, D]
    q = torch.nn.functional.normalize(q + proj(q), dim=-1)
    scores = (q @ torch.nn.functional.normalize(tool_vecs, dim=-1).T) / cfg.model.score_temperature  # [1, K]
    logits = torch.full((1, N), float("-inf"), device=dev)                                          # scatter -> catalog width
    idx = torch.tensor([ds.catalog.index(c) for c in ex.candidates], device=dev)
    logits[0, idx] = scores[0]
    logits = logits.masked_fill(~build_candidate_mask([ex], ds.catalog).to(dev), float("-inf"))     # existing mask, now a no-op check
    return ActionRankModel.rank_logits(logits, 5)[0], int(tok_mask.sum(1).min().item())

ev = ds.eval[:cfg.eval.max_eval_examples]
by_k = collections.defaultdict(list)
for ex in ev: by_k[len(ex.candidates)].append(ex)
sample = [ex for k in sorted(by_k) for ex in by_k[k][:4]]
for ex in sample[:3]: current(ex); span(ex)  # warmup
rows = collections.defaultdict(lambda: {"cur": [], "span": [], "tokens": []})
for ex in sample:
    k = len(ex.candidates)
    synchronize(dev); t0 = time.perf_counter(); current(ex); synchronize(dev); rows[k]["cur"].append(1000 * (time.perf_counter() - t0))
    synchronize(dev); t0 = time.perf_counter(); picks, min_span_tokens = span(ex); synchronize(dev); rows[k]["span"].append(1000 * (time.perf_counter() - t0))
    rows[k]["tokens"].append(len(tok(build_prompt(ex, ds.catalog, cfg.verbalize))["input_ids"]))
    assert all(ds.catalog.names[i] in ex.candidates for i in picks) and min_span_tokens >= 1
print(f"{'cands':>5} {'n':>2} {'prompt toks':>11} {'current ms':>10} {'span ms':>8} {'delta ms':>8}")
allc, alls = [], []
for k in sorted(rows):
    c, s = statistics.mean(rows[k]["cur"]), statistics.mean(rows[k]["span"]); allc += rows[k]["cur"]; alls += rows[k]["span"]
    print(f"{k:5d} {len(rows[k]['cur']):2d} {statistics.mean(rows[k]['tokens']):11.0f} {c:10.1f} {s:8.1f} {s - c:+8.1f}")
print(f"overall mean: current {statistics.mean(allc):.1f} ms, span {statistics.mean(alls):.1f} ms, delta {statistics.mean(alls)-statistics.mean(allc):+.1f} ms ({(statistics.mean(alls)/statistics.mean(allc)-1)*100:+.1f}%)")
print("all span picks were valid candidates; every candidate span covered >= 1 token")
