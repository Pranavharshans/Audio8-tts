#!/bin/bash -l
#SBATCH --job-name=audio8-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=audio8-setup-%j.out
#SBATCH --error=audio8-setup-%j.err

set -euo pipefail
unset SLURM_EXPORT_ENV

PROJECT_ROOT="$(realpath "${1:?usage: $0 PROJECT_ROOT VENV [WITH_DEEPSPEED]}")"
VENV="${2:?VENV is required and should normally be under \$WORK}"
WITH_DEEPSPEED="${3:-false}"

module purge
module load python

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export PIP_CACHE_DIR="$(dirname "${VENV}")/pip-cache"

mkdir -p "$(dirname "${VENV}")"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -r "${PROJECT_ROOT}/requirements-train.txt"
if [[ "${WITH_DEEPSPEED}" == "true" ]]; then
  python -m pip install -r "${PROJECT_ROOT}/requirements-deepspeed.txt"
elif python -m pip show deepspeed >/dev/null 2>&1; then
  # Accelerate imports any installed DeepSpeed package even when ZeRO is off.
  # Remove stale installs from single-GPU environments that lack CUDA nvcc.
  python -m pip uninstall -y deepspeed
fi

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the Slurm-assigned GPU")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
