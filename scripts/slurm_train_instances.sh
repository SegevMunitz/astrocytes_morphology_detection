#!/usr/bin/env bash
#SBATCH --job-name=astroseg-train
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=.astroseg_runtime/cluster_logs/%x_%A_%a.out
#SBATCH --error=.astroseg_runtime/cluster_logs/%x_%A_%a.err

set -euo pipefail

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cluster/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

fold="${SLURM_ARRAY_TASK_ID:-${ASTROSEG_FOLD:-0}}"
epochs="${ASTROSEG_EPOCHS:-100}"
run_name="${ASTROSEG_RUN_NAME:-cross_validation}"
output_dir=".astroseg_runtime/outputs/checkpoints/${run_name}/fold_${fold}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/train_instances.py \
    --config configs/train_instances_drive.yaml \
    --fold "${fold}" \
    --epochs "${epochs}" \
    --output-dir "${output_dir}"
