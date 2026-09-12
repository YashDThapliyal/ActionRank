"""Runs detached on the Colab VM (launched by run_remote.py). Logs to /content/run.log; writes /content/DONE
or /content/FAILED at the end. Packages results after every stage so a preempted session still yields something."""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

RUN_TIER2 = True
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
        for folder in ("checkpoints/baseline_sft", "checkpoints/tier2"):
            for path in (ROOT / folder).rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ROOT))
        for path in (ROOT / "results").glob("*.json"):
            zf.write(path, path.relative_to(ROOT))
    log(f"RESULTS READY after {stage}: {out} {out.stat().st_size // 1_000_000} MB")


LOG.write_text("")
sh("rm -rf /content/ActionRank && mkdir -p /content/ActionRank && unzip -qo /content/actionrank_colab.zip -d /content/ActionRank")
sh("pip -q install -r requirements.txt 2>&1 | grep -v 'already satisfied' || true")
sh("pip -q uninstall -y torchao || true")  # Colab preinstalls torchao 0.10, incompatible with transformers 5
os.environ["ACTIONRANK_DEVICE"] = "cuda"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("sed -i 's/  sft_batch_size: 1/  sft_batch_size: 8/; s/  sft_grad_accum: 16/  sft_grad_accum: 2/' config.yaml")
sh("ACTIONRANK_DEVICE=cuda python -u baseline_sft.py 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("baseline_sft")
if RUN_TIER2:
    sh("sed -i 's/  max_train_examples: 400/  max_train_examples: 10568/; s/^  epochs: 1$/  epochs: 2/; s/  batch_size: 2$/  batch_size: 8/; s/  grad_accum: 8/  grad_accum: 2/' config.yaml")
    sh("ACTIONRANK_DEVICE=cuda python -u train_tier2.py 2>&1 | grep --line-buffered -v 'HTTP Request'")
    package("tier2")
log("ALL DONE")
Path("/content/DONE").write_text("ok")
