"""`colab exec -f colab/run_remote.py`: launch remote_pipeline.py detached on the VM and return at once.
Requires /content/actionrank_colab.zip and /content/remote_pipeline.py (both via `colab upload`)."""
import subprocess
from pathlib import Path

for marker in ("/content/DONE", "/content/FAILED"):
    Path(marker).unlink(missing_ok=True)
proc = subprocess.Popen("nohup python /content/remote_pipeline.py > /content/pipeline_stdout.log 2>&1 &",
                        shell=True, cwd="/content")
proc.wait()
print("launched remote_pipeline.py detached; poll with colab/tail_remote.py")
