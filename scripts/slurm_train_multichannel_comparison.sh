#!/usr/bin/env bash
#SBATCH --job-name=astroseg-compare
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

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

epochs="${ASTROSEG_EPOCHS:-100}"
run_name="${ASTROSEG_RUN_NAME:-multichannel_cellpose_split}"
output_dir="${data_root}/outputs/checkpoints/${run_name}/fold_cellpose_split"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/train_instances.py \
    --config configs/train_instances_cluster.yaml \
    --train-id-list configs/cellpose_split_original_channels/train_ids.txt \
    --validation-id-list configs/cellpose_split_original_channels/validation_ids.txt \
    --epochs "${epochs}" \
    --output-dir "${output_dir}"
