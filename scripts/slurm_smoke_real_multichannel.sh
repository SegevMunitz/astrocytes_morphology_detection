#!/usr/bin/env bash
#SBATCH --job-name=astroseg-real-smoke
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
export ASTROSEG_DATA_ROOT="${data_root}"
export ASTROSEG_DATASET_NAME="${ASTROSEG_DATASET_NAME:-multichannel_20260816}"

module purge
module load python/3.11.13
source .venv-cluster/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/smoke_test_real_multichannel.py \
    --config configs/train_instances_cluster.yaml
