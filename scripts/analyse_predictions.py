"""Error analysis over results/predictions_*.jsonl: Finish behaviour and unseen-tool accuracy per system."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config
from data import load_dataset, FINISH
cfg = load_config(); ds = load_dataset(cfg)
train_labels = Counter(ex.label for ex in ds.train)
rows = []
for path in sorted(Path(cfg.eval.results_dir).glob("predictions_*.jsonl")):
    system = path.stem.replace("predictions_", "")
    preds = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(preds)
    finish_lab = [p for p in preds if p["label"] == FINISH]
    non_finish = [p for p in preds if p["label"] != FINISH]
    unseen = [p for p in non_finish if train_labels[p["label"]] == 0]
    seen = [p for p in non_finish if train_labels[p["label"]] > 0]
    acc = lambda ps: (sum(p["top1"] == p["label"] for p in ps) / len(ps)) if ps else float("nan")
    rows.append({
        "system": system, "n": n,
        "predicts_finish_rate": sum(p["top1"] == FINISH for p in preds) / n,
        "finish_recall": acc(finish_lab), "n_finish": len(finish_lab),
        "top1_non_finish": acc(non_finish),
        "top1_seen_tools": acc(seen), "n_seen": len(seen),
        "top1_unseen_tools": acc(unseen), "n_unseen": len(unseen),
    })
hdr = "| system | predicts Finish | Finish recall (n) | top-1 non-Finish | top-1 seen tools (n) | top-1 unseen tools (n) |"
print(hdr); print("|---|---:|---:|---:|---:|---:|")
for r in rows:
    print(f"| {r['system']} | {r['predicts_finish_rate']:.1%} | {r['finish_recall']:.1%} ({r['n_finish']}) | {r['top1_non_finish']:.1%} | "
          f"{r['top1_seen_tools']:.1%} ({r['n_seen']}) | {r['top1_unseen_tools']:.1%} ({r['n_unseen']}) |")
