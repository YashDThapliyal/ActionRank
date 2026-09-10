"""`colab exec -f colab/tail_remote.py`: print run status and the last lines of the remote log."""
from pathlib import Path

state = "DONE" if Path("/content/DONE").exists() else "FAILED" if Path("/content/FAILED").exists() else "RUNNING"
print(f"STATE: {state}")
log = Path("/content/run.log")
lines = log.read_text(errors="replace").replace("\r", "\n").splitlines() if log.exists() else []
keep = [ln for ln in lines if ln.strip()]
progress = [ln for ln in keep if "it/s]" in ln or "s/it]" in ln]
print("LAST PROGRESS:", progress[-1][-90:] if progress else "-")
for ln in [ln for ln in keep if "it/s]" not in ln and "s/it]" not in ln][-12:]:
    print(ln[:150])
