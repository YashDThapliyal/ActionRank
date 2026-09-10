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
dev = next(backbone.parameters()).device; N = len(ds.catalog); D = backbone.config.hidden_size
proj_cpu = torch.nn.Linear(D, D)

def spans_for(ex, prompt):
    block = render_catalog(ex.candidates, ds.catalog, cfg.verbalize); cursor = prompt.index(block); out = []
    for name in ex.candidates:
        s = prompt.index(f"- {name}: ", cursor); e = prompt.find("\n", s); e = len(prompt) if e < 0 else e
        out.append((s, e)); cursor = e
    return out

@torch.no_grad()
def span_cpu(ex):
    t = {}
    synchronize(dev); t0 = time.perf_counter()
    prompt = build_prompt(ex, ds.catalog, cfg.verbalize); spans = spans_for(ex, prompt)
    tok.padding_side = "right"; tok.truncation_side = "left"
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=cfg.model.max_prompt_tokens)
    offsets = enc.pop("offset_mapping")[0]; ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
    t["tokenize+offsets"] = time.perf_counter() - t0; synchronize(dev); t1 = time.perf_counter()
    hidden = backbone.get_decoder()(input_ids=ids, attention_mask=am).last_hidden_state
    synchronize(dev); t["prefill"] = time.perf_counter() - t1; t2 = time.perf_counter()
    h = hidden[0].float().cpu()                                   # one copy [T, D]
    t["copy_to_cpu"] = time.perf_counter() - t2; t3 = time.perf_counter()
    sp = torch.tensor(spans); m = ((offsets[:, 0].unsqueeze(0) < sp[:, 1:2]) & (offsets[:, 1].unsqueeze(0) > sp[:, 0:1])).float()
    tool_vecs = (m @ h) / m.sum(1, keepdim=True).clamp(min=1)
    q = h.mean(0, keepdim=True); q = torch.nn.functional.normalize(q + proj_cpu(q), dim=-1)
    scores = (q @ torch.nn.functional.normalize(tool_vecs, dim=-1).T) / cfg.model.score_temperature
    logits = torch.full((1, N), float("-inf")); logits[0, [ds.catalog.index(c) for c in ex.candidates]] = scores[0]
    logits = logits.masked_fill(~build_candidate_mask([ex], ds.catalog), float("-inf"))
    picks = ActionRankModel.rank_logits(logits, 5)[0]
    t["span_pool+scatter+mask+topk (cpu)"] = time.perf_counter() - t3
    t["total"] = time.perf_counter() - t0
    return t

ev = ds.eval[:cfg.eval.max_eval_examples]; by_k = collections.defaultdict(list)
for ex in ev: by_k[len(ex.candidates)].append(ex)
sample = [ex for k in sorted(by_k) for ex in by_k[k][:4]]
for ex in sample[:3]: span_cpu(ex)
acc = collections.defaultdict(list); per_k = collections.defaultdict(list)
for ex in sample:
    t = span_cpu(ex)
    for k, v in t.items(): acc[k].append(1000 * v)
    per_k[len(ex.candidates)].append(1000 * t["total"])
for k, v in acc.items(): print(f"{k:38s} mean {statistics.mean(v):6.1f} ms")
print("total by candidate count:", {k: round(statistics.mean(v)) for k, v in sorted(per_k.items())})
