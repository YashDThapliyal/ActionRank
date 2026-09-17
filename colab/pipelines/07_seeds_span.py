"""Detached on the Colab VM: two seed replicates of the final span scorer (configs/seeds/span_s2.yaml, span_s3.yaml).
For each seed: train 3 epochs, package the checkpoint at once, evaluate on held-out steps [503, 1855) (the
unmonitored slice; accuracy only, A100 latency is not reported), package again. Results land in results/07-seeds/."""
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path("/content/ActionRank")
LOG = Path("/content/run.log")
SEEDS = (2, 3)


def log(msg: str) -> None:
    with open(LOG, "a") as fh:
        fh.write(msg.rstrip() + "\n")


def sh(cmd: str, env: dict | None = None) -> None:
    log(f"$ {cmd}")
    with open(LOG, "a") as fh:
        proc = subprocess.run(cmd, shell=True, cwd=ROOT if ROOT.exists() else "/content", stdout=fh,
                              stderr=subprocess.STDOUT, text=True, env={**os.environ, **(env or {})})
    if proc.returncode != 0:
        log(f"COMMAND FAILED ({proc.returncode}): {cmd}")
        Path("/content/FAILED").write_text(cmd)
        raise SystemExit(1)


def package(stage: str) -> None:
    out = Path("/content/actionrank_results.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in ("checkpoints/seeds", "results/07-seeds"):
            for path in (ROOT / folder).rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ROOT))
    log(f"RESULTS READY after {stage}: {out} {out.stat().st_size // 1_000_000} MB")


LOG.write_text("")
sh("rm -rf /content/ActionRank && mkdir -p /content/ActionRank && unzip -qo /content/actionrank_colab.zip -d /content/ActionRank")
sh("pip -q install -r requirements.txt 2>&1 | grep -v 'already satisfied' || true")
sh("pip -q uninstall -y torchao || true")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
os.environ["ACTIONRANK_DEVICE"] = "cuda"
for seed in SEEDS:
    cfg = f"configs/seeds/span_s{seed}.yaml"
    env = {"ACTIONRANK_CONFIG": cfg}
    sh(f"grep -nE 'train_seed|pooling|^  head:|max_train_examples|^  epochs|batch_size|grad_accum|checkpoint_dir|results_dir' {cfg}", env)
    sh("python -u train_tier2.py 2>&1 | grep --line-buffered -v 'HTTP Request'", env)
    package(f"span-seed{seed}-train")
    sh("python -u eval.py --systems tier2 --offset 503 --all-remaining 2>&1 | grep --line-buffered -v 'HTTP Request'", env)
    package(f"span-seed{seed}-eval")
log("ALL DONE")
Path("/content/DONE").write_text("ok")
