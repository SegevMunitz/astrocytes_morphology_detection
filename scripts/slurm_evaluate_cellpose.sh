#!/usr/bin/env bash
#SBATCH --job-name=cellpose-evaluate
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
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
checkpoint="${CELLPOSE_CHECKPOINT:-${data_root}/outputs/cellpose/lr_sweeps/lr_sweep_original_channels_20260810/cyto3/lr_0p4/result/models/cyto3_lr_0p4_epoch_0450}"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cellpose3/bin/activate

cluster_site_packages="${project_root}/.venv-cluster/lib/python3.11/site-packages"
export PYTHONPATH="${project_root}/src:${cluster_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
export CELLPOSE_LOCAL_MODELS_PATH="${data_root}/outputs/cellpose/pretrained_models"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/evaluate_cellpose_model.py \
    --image-dir "${data_root}/training_images" \
    --mask-dir "${data_root}/training_masks" \
    --image-list configs/cellpose_split_original_channels/validation_ids.txt \
    --checkpoint "${checkpoint}" \
    --output-dir "${data_root}/outputs/evaluations/cellpose_original_channels" \
    --chan 1 \
    --chan2 3
