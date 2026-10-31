"""Kaggle script (GPU T4 x2): run GPUlayer's multi-GPU validation and keep the log as output.

Pushed by benchmarks/cloud/kaggle.sh. Needs internet on (phone-verified Kaggle account).
"""

import subprocess
import sys

REF = "main"


def sh(cmd: str) -> int:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


sh("nvidia-smi && nvidia-smi topo -m")
sh(f"git clone -q --branch {REF} https://github.com/antonellof/gpulayer /kaggle/working/gpulayer")
sh(f"{sys.executable} -m pip install -q -e /kaggle/working/gpulayer")
TORCH = "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
sh(f'{sys.executable} -c "{TORCH}"')
rc = sh(
    f"cd /kaggle/working/gpulayer && {sys.executable} scripts/validate_multigpu.py "
    "2>&1 | tee /kaggle/working/validate_multigpu.log"
)
sh("rm -rf /kaggle/working/gpulayer")  # keep only the log as output
sys.exit(rc)
