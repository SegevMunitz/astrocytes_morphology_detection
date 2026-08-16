#!/usr/bin/env bash
#SBATCH --job-name=cellpose-summary
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
sweep_name="${CELLPOSE_SWEEP_NAME:-lr_sweep_original_channels_20260810}"
sweep_directory="${data_root}/outputs/cellpose/lr_sweeps/${sweep_name}"

module purge
module load python/3.11.13
source .venv-cluster/bin/activate

python scripts/summarize_cellpose_sweep.py \
    --sweep-dir "${sweep_directory}" \
    --output "${sweep_directory}/summary.csv" \
    --excel-output "${sweep_directory}/summary.xlsx"
