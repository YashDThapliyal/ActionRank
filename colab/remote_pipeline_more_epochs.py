"""Detached on the Colab VM: continue training both fine-tuned systems to 6 epochs total.
  1. span scorer (last-token pooling): resume checkpoints/tier2_span (3 epochs) for +3 epochs
  2. generation baseline: resume checkpoints/baseline_sft (1 epoch) for +5 epochs
Packages checkpoints after each stage."""
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
        for folder in ("checkpoints/tier2_span6", "checkpoints/baseline_sft6"):
            for path in (ROOT / folder).rglob("*"):
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
sh("sed -i 's/  head: table             # table | span/  head: span              # table | span/; s/  max_train_examples: 400/  max_train_examples: 10568/; s/^  epochs: 1$/  epochs: 3/; s/  batch_size: 2$/  batch_size: 8/; s/  grad_accum: 8/  grad_accum: 2/; s#checkpoint_dir: checkpoints/tier2#checkpoint_dir: checkpoints/tier2_span6#' config.yaml")
sh("sed -i 's/  sft_epochs: 1 /  sft_epochs: 5 /; s/  sft_batch_size: 1/  sft_batch_size: 8/; s/  sft_grad_accum: 16/  sft_grad_accum: 2/; s#sft_checkpoint_dir: checkpoints/baseline_sft#sft_checkpoint_dir: checkpoints/baseline_sft6#' config.yaml")
sh("grep -nE 'pooling|^  head:|max_train_examples|^  epochs|batch_size|grad_accum|checkpoint_dir|sft_' config.yaml")
sh("ACTIONRANK_DEVICE=cuda python -u train_tier2.py --resume checkpoints/tier2_span 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("tier2-span-6epochs")
sh("ACTIONRANK_DEVICE=cuda python -u baseline_sft.py --resume checkpoints/baseline_sft 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("baseline-sft-6epochs")
log("ALL DONE")
Path("/content/DONE").write_text("ok")
