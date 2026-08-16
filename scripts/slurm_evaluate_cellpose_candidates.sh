#!/usr/bin/env bash
#SBATCH --job-name=cp3ch-eval
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=RTX2080Ti|teslaV100S|A100|rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-8%3
#SBATCH --output=cluster_logs/%x_%A_%a.out
#SBATCH --error=cluster_logs/%x_%A_%a.err

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
run_name="${CELLPOSE_TRANSFER_RUN_NAME:-three_channel_transfer_20260816}"
run_root="${data_root}/outputs/cellpose/three_channel/${run_name}"
candidate_file="${run_root}/checkpoint_candidates.csv"
task_id="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
line="$(sed -n "$((task_id + 2))p" "${candidate_file}")"
if [[ -z "${line}" ]]; then
    echo "No checkpoint candidate at row ${task_id}" >&2
    exit 1
fi
IFS=, read -r learning_rate_run epoch validation_loss checkpoint reason <<<"${line}"
evaluation_name="${learning_rate_run}_epoch_${epoch}_gfp_zero"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cellpose3/bin/activate

cluster_site_packages="${project_root}/.venv-cluster/lib/python3.11/site-packages"
export PYTHONPATH="${project_root}/src:${cluster_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/evaluate_cellpose_model.py \
    --image-dir "${data_root}/training_images" \
    --mask-dir "${data_root}/training_masks" \
    --image-list configs/cellpose_split_original_channels/validation_ids.txt \
    --checkpoint "${checkpoint}" \
    --output-dir "${data_root}/outputs/evaluations/${run_name}/${evaluation_name}" \
    --input-channels 3 \
    --zero-auxiliary
