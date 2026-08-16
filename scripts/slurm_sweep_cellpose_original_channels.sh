#!/usr/bin/env bash
#SBATCH --job-name=cellpose-original-ch
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
sweep_name="${CELLPOSE_SWEEP_NAME:-lr_sweep_original_channels_20260810}"
split_directory="${CELLPOSE_SPLIT_DIR:-configs/cellpose_split_original_channels}"
model_types=(cyto2_cp3 cyto3)
learning_rates=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4)
learning_rate_tokens=(0p05 0p1 0p15 0p2 0p25 0p3 0p35 0p4)
task_id="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
model_index=$((task_id / 8))
learning_rate_index=$((task_id % 8))
model_type="${model_types[model_index]}"
learning_rate="${learning_rates[learning_rate_index]}"
token="${learning_rate_tokens[learning_rate_index]}"
task_root="${data_root}/outputs/cellpose/lr_sweeps/${sweep_name}/${model_type}/lr_${token}"
train_directory="${task_root}/train"
validation_directory="${task_root}/validation"
result_directory="${task_root}/result"
model_cache="${data_root}/outputs/cellpose/pretrained_models"

if [[ -e "${task_root}" ]]; then
    echo "Refusing to overwrite existing sweep task: ${task_root}" >&2
    exit 1
fi
mkdir -p "${task_root}"
exec >"${task_root}/slurm.out" 2>"${task_root}/slurm.err"

echo "Slurm job ${SLURM_ARRAY_JOB_ID}_${task_id}: model=${model_type}, learning_rate=${learning_rate}"
echo "Input: original four-plane TIFFs; Cellpose channels: cell=1, nucleus=3"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cellpose3/bin/activate

export CELLPOSE_LOCAL_MODELS_PATH="${model_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/prepare_cellpose_training_data.py \
    --input-dir "${data_root}/training_images" \
    --staging-dir "${train_directory}" \
    --image-list "${split_directory}/train_ids.txt" \
    --chan 1 \
    --chan2 3
python scripts/prepare_cellpose_training_data.py \
    --input-dir "${data_root}/training_images" \
    --staging-dir "${validation_directory}" \
    --image-list "${split_directory}/validation_ids.txt" \
    --chan 1 \
    --chan2 3

python scripts/train_cellpose_model.py \
    --train-dir "${train_directory}" \
    --validation-dir "${validation_directory}" \
    --output-dir "${result_directory}" \
    --model-name "${model_type}_lr_${token}" \
    --pretrained-model "${model_type}" \
    --learning-rate "${learning_rate}" \
    --weight-decay 0.00001 \
    --momentum 0.9 \
    --epochs 500 \
    --batch-size 8 \
    --save-every 50 \
    --chan 1 \
    --chan2 3
