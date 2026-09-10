"""Executed on the Colab kernel via `colab exec -f colab/run_remote.py`.

Expects /content/actionrank_colab.zip (uploaded with `colab upload`). Trains the fine-tuned generation
baseline on the GPU and, if RUN_TIER2, Tier 2 at scale; packages checkpoints + histories into
/content/actionrank_results.zip for `colab download`.
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

RUN_TIER2 = True
ROOT = Path("/content/ActionRank")


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    proc = subprocess.run(cmd, shell=True, cwd=ROOT if ROOT.exists() else "/content", stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if "HTTP Request" not in ln]
    print("\n".join(lines[-80:]), flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {cmd}")


sh("rm -rf /content/ActionRank && mkdir -p /content/ActionRank && unzip -qo /content/actionrank_colab.zip -d /content/ActionRank")
sh("pip -q install -r requirements.txt")
os.environ["ACTIONRANK_DEVICE"] = "cuda"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
# bigger batches on a real GPU; same effective batch (16) as the local runs
sh("sed -i 's/  sft_batch_size: 1/  sft_batch_size: 8/; s/  sft_grad_accum: 16/  sft_grad_accum: 2/' config.yaml")


def package(stage: str) -> None:
    """Zip whatever exists so far; re-done after every stage in case the free-tier session is preempted."""
    out = Path("/content/actionrank_results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("checkpoints/baseline_sft", "checkpoints/tier2"):
            for path in (ROOT / folder).rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ROOT))
        for path in (ROOT / "results").glob("*.json"):
            zf.write(path, path.relative_to(ROOT))
    print(f"RESULTS READY after {stage}: {out} {out.stat().st_size // 1_000_000} MB", flush=True)


sh("ACTIONRANK_DEVICE=cuda python baseline_sft.py")
package("baseline_sft")
if RUN_TIER2:
    sh("sed -i 's/  max_train_examples: 400/  max_train_examples: 10568/; s/^  epochs: 1$/  epochs: 2/; s/  batch_size: 2$/  batch_size: 8/; s/  grad_accum: 8/  grad_accum: 2/' config.yaml")
    sh("ACTIONRANK_DEVICE=cuda python train_tier2.py")
    package("tier2")
print("ALL DONE", flush=True)
