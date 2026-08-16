#!/usr/bin/env bash
#SBATCH --job-name=cp3ch-predict
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=RTX2080Ti|teslaV100S|A100|rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
checkpoint="${CELLPOSE_CHECKPOINT:-${data_root}/outputs/cellpose/three_channel/three_channel_transfer_20260816/lr_0p003/result/models/cellpose_3ch_lr_0p003_epoch_0250}"
output_directory="${CELLPOSE_PREDICTION_OUTPUT:-${data_root}/outputs/predictions/three_channel_transfer_20260816}"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cellpose3/bin/activate

cluster_site_packages="${project_root}/.venv-cluster/lib/python3.11/site-packages"
export PYTHONPATH="${project_root}/src:${cluster_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/predict_cellpose_three_channel.py \
    --image-dir "${data_root}/test_images" \
    --checkpoint "${checkpoint}" \
    --output-dir "${output_directory}" \
    --flow-threshold 0.4 \
    --cellprob-threshold 0.0
