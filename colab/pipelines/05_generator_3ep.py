"""Detached on the Colab VM: fine-tune the generation baseline for 3 epochs from scratch (matched budget with
the 3-epoch span scorer), package the checkpoint immediately, then score it on the 500 held-out steps as a
safety net and package again. Uses configs/generator_3ep.yaml via ACTIONRANK_CONFIG (the upload zip must include
configs/); batch/accum bumped for the A100. Results land in results_sft3 on the VM and are moved into
results/05-generator-lora-3ep locally."""
import os
import subprocess
import zipfile
from pathlib import Path

ROOT = Path("/content/ActionRank")
LOG = Path("/content/run.log")
CFG = "configs/generator_3ep.yaml"


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
        for folder in ("checkpoints/baseline_sft3", "results_sft3"):
            for path in (ROOT / folder).rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(ROOT))
    log(f"RESULTS READY after {stage}: {out} {out.stat().st_size // 1_000_000} MB")


LOG.write_text("")
sh("rm -rf /content/ActionRank && mkdir -p /content/ActionRank && unzip -qo /content/actionrank_colab.zip -d /content/ActionRank")
sh("pip -q install -r requirements.txt 2>&1 | grep -v 'already satisfied' || true")
sh("pip -q uninstall -y torchao || true")
os.environ["ACTIONRANK_DEVICE"] = "cuda"
os.environ["ACTIONRANK_CONFIG"] = CFG
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"sed -i 's/  sft_batch_size: 1 /  sft_batch_size: 8 /; s/  sft_grad_accum: 16/  sft_grad_accum: 2/; s#results_dir: results/05-generator-lora-3ep#results_dir: results_sft3#' {CFG}")
sh(f"grep -nE 'sft_|results_dir|pooling|^  head:' {CFG}")
sh("python -u baseline_sft.py 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("sft-3epochs-train")
sh("python -u eval.py --systems baseline_sft 2>&1 | grep --line-buffered -v 'HTTP Request'")
package("sft-3epochs-eval")
log("ALL DONE")
Path("/content/DONE").write_text("ok")
