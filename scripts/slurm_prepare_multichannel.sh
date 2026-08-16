#!/usr/bin/env bash
#SBATCH --job-name=astroseg-prepare-rgb
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
dataset_name="${ASTROSEG_DATASET_NAME:-multichannel_20260816}"

module purge
module load python/3.11.13
source .venv-cluster/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/prepare_multichannel_dataset.py \
    --data-root "${data_root}" \
    --output-dir "${data_root}/outputs/datasets/${dataset_name}" \
    --annotator "${ASTROSEG_ANNOTATOR:-Segev}"
