#!/usr/bin/env bash
#SBATCH --job-name=cellpose3-train
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
run_name="${CELLPOSE_RUN_NAME:-cyto2_cp3_500ep_lr0p1}"
run_directory="${data_root}/outputs/cellpose/training_runs/${run_name}"
model_cache="${data_root}/outputs/cellpose/pretrained_models"

if [[ -e "${run_directory}" ]]; then
    echo "Refusing to overwrite existing Cellpose run: ${run_directory}" >&2
    exit 1
fi
mkdir -p "${model_cache}" "$(dirname "${run_directory}")"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cellpose3/bin/activate

export CELLPOSE_LOCAL_MODELS_PATH="${model_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/prepare_cellpose_training_data.py \
    --input-dir "${data_root}/training_images" \
    --staging-dir "${run_directory}" \
    --chan 1 \
    --chan2 3

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python -m cellpose \
    --train \
    --use_gpu \
    --dir "${run_directory}" \
    --mask_filter _seg.npy \
    --pretrained_model cyto2_cp3 \
    --chan 1 \
    --chan2 3 \
    --n_epochs 500 \
    --learning_rate 0.1 \
    --weight_decay 0.00001 \
    --SGD 1 \
    --batch_size 8 \
    --save_every 100 \
    --model_name_out "${run_name}" \
    --verbose

echo "Cellpose model: ${run_directory}/models/${run_name}"
