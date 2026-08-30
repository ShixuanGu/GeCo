#!/bin/bash
# End-to-end test of the release on one A100: every geometry x flow combination of
# demo_detection.py on examples/deform_house, plus GeCo-Eval_evaluation.py on a tiny
# fixture, plus the backend contract checker.
# Usage (from the repo root):  mkdir -p jobs/logs && sbatch scripts/slurm/run_all_backends_a100.sh
#SBATCH --job-name=geco_all
#SBATCH -p gpu,seas_gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 0-06:00
#SBATCH -o jobs/logs/geco_all_%j.out
#SBATCH -e jobs/logs/geco_all_%j.err

set -uo pipefail
# Under sbatch $0 is a spooled copy in /var/slurmd, so locate the repo via the
# submission directory (the usage line above submits from the repo root).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p jobs/logs reports

# Environment (site-specific; override at submission time, see check_backends_a100.sh):
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then source "$CONDA_SH"; fi
conda activate "${GECO_ENV:-geco}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

OUT=reports/run_${SLURM_JOB_ID}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpu.txt"
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.version.cuda)" | tee -a "$OUT/gpu.txt"

status=0
run() {  # run <name> <cmd...>
  local name=$1; shift
  echo "===== [$name] $*"
  local t0=$(date +%s)
  if "$@" > "$OUT/$name.log" 2>&1; then echo "[$name] OK ($(( $(date +%s) - t0 )) s)"; else echo "[$name] FAILED ($(( $(date +%s) - t0 )) s) -- see $OUT/$name.log"; status=1; fi
}

# 1) contract checker on all backends (ptlflow only if installed)
FLOWS="ufm,sea_raft,raft"
python -c "import ptlflow" 2>/dev/null && FLOWS="$FLOWS,ptlflow:dpflow:sintel"
run check_backends python scripts/check_backends.py --geometry vggt,vggt_omega --flow "$FLOWS" --report "$OUT/check_backends.json" --strict

# 1b) unit tests: output contracts + bit-equivalence of the default backends with the legacy code path
run pytest python -m pytest tests -q

# 2) demo_detection.py: every geometry x flow combination (short window to keep the runtime reasonable)
for G in vggt vggt_omega; do
  for FL in ufm sea_raft raft; do
    run "demo_${G}_${FL}" python demo_detection.py --frame_path examples/deform_house --outdir "$OUT/demo_${G}_${FL}" \
        --geometry "$G" --flow "$FL" --window_size 8
  done
done
if python -c "import ptlflow" 2>/dev/null; then
  run demo_vggt_ptlflow_dpflow python demo_detection.py --frame_path examples/deform_house --outdir "$OUT/demo_vggt_ptlflow_dpflow" \
      --geometry vggt --flow ptlflow --flow_model dpflow --flow_ckpt sintel --window_size 8
fi

# 3) full default demo exactly as in the README (window 20, all 49 frames)
run demo_default python demo_detection.py --frame_path examples/deform_house --outdir "$OUT/demo_default"

# 4) GeCo-Eval evaluation on a tiny fixture (one clip symlinked from examples/) with two backend combos
FIXTURE=reports/fixture/GeCo-Eval
if [ ! -d "$FIXTURE/Gen_Veo3.1/indoor_prompts/b1_0" ]; then
  mkdir -p "$FIXTURE/Gen_Veo3.1/indoor_prompts/b1_0"
  ln -sf "$PWD"/examples/deform_house/*.png "$FIXTURE/Gen_Veo3.1/indoor_prompts/b1_0/"
fi
run eval_vggt_ufm       bash -c "cd '$OUT' && python ../../GeCo-Eval_evaluation.py --frames_root ../../reports/fixture/GeCo-Eval --models Gen_Veo3.1 --print_per_clip"
run eval_omega_searaft  bash -c "cd '$OUT' && python ../../GeCo-Eval_evaluation.py --frames_root ../../reports/fixture/GeCo-Eval --models Gen_Veo3.1 --print_per_clip --geometry vggt_omega --flow sea_raft --csv_dir eval_omega_searaft"

echo "ALL DONE status=$status"; exit $status
