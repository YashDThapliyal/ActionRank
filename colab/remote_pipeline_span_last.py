"""Detached on the Colab VM: Tier 2 for the span head with last-token pooling (LoRA + SpanScoringHead,
initialised from the shipped last-pooled span head), 3 epochs over all training steps. Packages checkpoints."""
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
        for path in list((ROOT / "checkpoints" / "tier2").rglob("*")) + list((ROOT / "results").glob("*.json")):
            if path.is_file():
                zf.write(path, path.relative_to(ROOT))
    log(f"RESULTS READY after {stage}: {out} {out.stat().st_size // 1_000_000} MB")


LOG.write_text("")
sh("rm -rf /content/ActionRank && mkdir -p /content/ActionRank && unzip -qo /content/actionrank_colab.zip -d /content/ActionRank")
sh("pip -q install -r requirements.txt 2>&1 | grep -v 'already satisfied' || true")
sh("pip -q uninstall -y torchao || true")
os.environ["ACTIONRANK_DEVICE"] = "cuda"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("sed -i 's/  pooling: mean /  pooling: last /' config.yaml")
sh("sed -i 's/  head: table             # table | span/  head: span              # table | span/; s/  max_train_examples: 400/  max_train_examples: 10568/; s/^  epochs: 1$/  epochs: 3/; s/  batch_size: 2$/  batch_size: 8/; s/  grad_accum: 8/  grad_accum: 2/' config.yaml")
sh("grep -nE 'pooling|^  head:|max_train_examples|^  epochs|batch_size|grad_accum' config.yaml")
sh("ACTIONRANK_DEVICE=cuda python -u train_tier2.py 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("tier2-span-last")
log("ALL DONE")
Path("/content/DONE").write_text("ok")
