#!/usr/bin/env bash
#SBATCH --job-name=cp3ch-transfer
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=RTX2080Ti|teslaV100S|A100|rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=cluster_logs/%x_%A_%a.out
#SBATCH --error=cluster_logs/%x_%A_%a.err

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
source_checkpoint="${CELLPOSE_TRANSFER_SOURCE:-${data_root}/outputs/cellpose/lr_sweeps/lr_sweep_original_channels_20260810/cyto2_cp3/lr_0p25/result/models/cyto2_cp3_lr_0p25_epoch_0450}"
run_name="${CELLPOSE_TRANSFER_RUN_NAME:-three_channel_transfer_20260816}"
learning_rates=(0.0005 0.001 0.003)
learning_rate="${learning_rates[${SLURM_ARRAY_TASK_ID}]}"
token="${learning_rate/./p}"
task_root="${data_root}/outputs/cellpose/three_channel/${run_name}/lr_${token}"
train_directory="${task_root}/train"
validation_directory="${task_root}/validation"
result_directory="${task_root}/result"

if [[ -e "${task_root}" ]]; then
    echo "Refusing to overwrite existing transfer task: ${task_root}" >&2
    exit 1
fi
mkdir -p "${task_root}"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cellpose3/bin/activate

cluster_site_packages="${project_root}/.venv-cluster/lib/python3.11/site-packages"
export PYTHONPATH="${project_root}/src:${cluster_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/prepare_cellpose_training_data.py \
    --input-dir "${data_root}/training_images" \
    --staging-dir "${train_directory}" \
    --image-list configs/cellpose_split_original_channels/train_ids.txt \
    --chan 1 \
    --chan2 3
python scripts/prepare_cellpose_training_data.py \
    --input-dir "${data_root}/training_images" \
    --staging-dir "${validation_directory}" \
    --image-list configs/cellpose_split_original_channels/validation_ids.txt \
    --chan 1 \
    --chan2 3

python scripts/train_cellpose_model.py \
    --train-dir "${train_directory}" \
    --validation-dir "${validation_directory}" \
    --output-dir "${result_directory}" \
    --model-name "cellpose_3ch_lr_${token}" \
    --pretrained-model "${source_checkpoint}" \
    --three-channel-transfer \
    --auxiliary-dropout-probability 0.5 \
    --optimizer adamw \
    --learning-rate "${learning_rate}" \
    --weight-decay 0.0001 \
    --epochs 300 \
    --batch-size 8 \
    --save-every 25
