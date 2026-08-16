#!/usr/bin/env bash
#SBATCH --job-name=astroseg-lr-sweep
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=cluster_logs/%x_%A_%a.out
#SBATCH --error=cluster_logs/%x_%A_%a.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
export ASTROSEG_DATA_ROOT="${data_root}"
export ASTROSEG_DATASET_NAME="${ASTROSEG_DATASET_NAME:-multichannel_20260816}"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cluster/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

learning_rates=(0.1 0.15 0.2 0.25 0.3 0.35 0.4)
learning_rate="${learning_rates[${SLURM_ARRAY_TASK_ID}]}"
slug="${learning_rate/./p}"
epochs="${ASTROSEG_EPOCHS:-100}"
run_name="${ASTROSEG_RUN_NAME:-multichannel_lr_sweep_20260816}"
output_dir="${data_root}/outputs/checkpoints/${run_name}/lr_${slug}/fold_cellpose_split"

echo "Starting fixed-LR run: learning_rate=${learning_rate}, output=${output_dir}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/train_instances.py \
    --config configs/train_instances_cluster.yaml \
    --train-id-list configs/cellpose_split_original_channels/train_ids.txt \
    --validation-id-list configs/cellpose_split_original_channels/validation_ids.txt \
    --learning-rate "${learning_rate}" \
    --disable-lr-scheduler \
    --epochs "${epochs}" \
    --output-dir "${output_dir}"
