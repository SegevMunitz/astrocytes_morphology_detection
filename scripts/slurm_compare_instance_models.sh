#!/usr/bin/env bash
#SBATCH --job-name=astroseg-model-compare
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
custom_run="${ASTROSEG_CUSTOM_RUN_NAME:-multichannel_cellpose_split_20260816}"

module purge
module load python/3.11.13
source .venv-cluster/bin/activate

python scripts/compare_instance_models.py \
    --custom-dir "${data_root}/outputs/evaluations/${custom_run}" \
    --cellpose-dir "${data_root}/outputs/evaluations/cellpose_original_channels" \
    --output-dir "${data_root}/outputs/evaluations/comparison_${custom_run}_vs_cellpose"
