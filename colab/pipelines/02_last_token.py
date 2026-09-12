"""Detached on the Colab VM: last-token pooling variant. Re-encodes with pooling=last, trains the Tier 1
table head, Tier 2 (LoRA + head, 2 epochs, all steps) from it, and the Tier 1 span head. Packages checkpoints."""
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path("/content/ActionRank")
LOG = Path("/content/run.log")


def log(msg: str) -> None:
    with open(LOG, "a") as fh:
        fh.write(msg.rstrip() + "\n")


def sh(cmd: str) -> None:
    log(f"$ {cmd}")
    with open(LOG, "a") as fh:
        proc = subprocess.run(cmd, shell=True, cwd=ROOT if ROOT.exists() else "/content", stdout=fh, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        log(f"COMMAND FAILED ({proc.returncode}): {cmd}")
        Path("/content/FAILED").write_text(cmd)
        raise SystemExit(1)


def package(stage: str) -> None:
    out = Path("/content/actionrank_results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in list((ROOT / "checkpoints").rglob("*")) + list((ROOT / "results").glob("*.json")):
            if path.is_file():
                zf.write(path, path.relative_to(ROOT))
    log(f"RESULTS READY after {stage}: {out} {out.stat().st_size // 1_000_000} MB")


LOG.write_text("")
sh("rm -rf /content/ActionRank && mkdir -p /content/ActionRank && unzip -qo /content/actionrank_colab.zip -d /content/ActionRank")
sh("rm -f checkpoints/tier1_head.pt")  # the shipped head is mean-pooled; retrain under last-token pooling
sh("pip -q install -r requirements.txt 2>&1 | grep -v 'already satisfied' || true")
sh("pip -q uninstall -y torchao || true")
os.environ["ACTIONRANK_DEVICE"] = "cuda"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("sed -i 's/  pooling: mean /  pooling: last /; s/  cache_batch_size: 16/  cache_batch_size: 64/' config.yaml && grep -nE 'pooling|cache_batch_size|head:' config.yaml")
sh("sed -i 's/  head: span /  head: table /' config.yaml")
sh("ACTIONRANK_DEVICE=cuda python -u train_tier1.py 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("tier1-table-last")
sh("sed -i 's/  max_train_examples: 400/  max_train_examples: 10568/; s/^  epochs: 1$/  epochs: 2/; s/  batch_size: 2$/  batch_size: 8/; s/  grad_accum: 8/  grad_accum: 2/' config.yaml")
sh("ACTIONRANK_DEVICE=cuda python -u train_tier2.py 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("tier2-last")
sh("sed -i 's/  head: table /  head: span /' config.yaml")
sh("ACTIONRANK_DEVICE=cuda python -u train_tier1.py 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("tier1-span-last")
log("ALL DONE")
Path("/content/DONE").write_text("ok")
