#!/bin/bash
# Contract check of every geometry / flow backend on one A100 (FASRC Cannon).
# Usage (from the repo root):  mkdir -p jobs/logs && sbatch scripts/slurm/check_backends_a100.sh [extra args for scripts/check_backends.py]
#SBATCH --job-name=geco_check
#SBATCH -p gpu,seas_gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-02:00
#SBATCH -o jobs/logs/geco_check_%j.out
#SBATCH -e jobs/logs/geco_check_%j.err

set -euo pipefail
# Under sbatch $0 is a spooled copy in /var/slurmd, so locate the repo via the
# submission directory (the usage line above submits from the repo root).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p jobs/logs reports

# --- environment (site-specific; override at submission time) ---------------
# CONDA_SH=<conda.sh> GECO_ENV=<env> sbatch scripts/slurm/check_backends_a100.sh
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; fi
conda activate "${GECO_ENV:-geco}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Weight caches follow $HF_HOME / $TORCH_HOME, which sbatch inherits from your
# shell.  If your compute nodes have no internet, pre-download the weights first
# (README "Model weights") and submit with HF_HUB_OFFLINE=1.
# ---------------------------------------------------------------------------

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'bf16', torch.cuda.is_bf16_supported())"
python scripts/check_backends.py --report "reports/check_backends_${SLURM_JOB_ID}.json" "$@"
