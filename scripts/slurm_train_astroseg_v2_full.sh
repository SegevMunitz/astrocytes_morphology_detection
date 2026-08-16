#!/usr/bin/env bash
#SBATCH --job-name=astroseg-v2-full
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=teslaV100S|A100|rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
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

seeds=(42 1337 2026)
seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
epochs="${ASTROSEG_EPOCHS:-60}"
learning_rate="${ASTROSEG_LEARNING_RATE:-0.0006}"
run_name="${ASTROSEG_RUN_NAME:-astroseg_v2_full_20260816}"
output_dir="${data_root}/outputs/checkpoints/${run_name}/seed_${seed}"

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing full-data run: ${output_dir}" >&2
    exit 1
fi

echo "AstroSeg v2 full-data random seed=${seed}; lr=${learning_rate}; epochs=${epochs}"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/train_instances.py \
    --config configs/train_astroseg_v2_cluster.yaml \
    --train-all \
    --seed "${seed}" \
    --learning-rate "${learning_rate}" \
    --epochs "${epochs}" \
    --output-dir "${output_dir}"
