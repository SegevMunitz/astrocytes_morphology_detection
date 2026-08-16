#!/usr/bin/env bash
#SBATCH --job-name=astroseg-calibrate
#SBATCH --partition=elscn.q
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
run_name="${ASTROSEG_RUN_NAME:-multichannel_lr_sweep_0p001_0p01_20260816/lr_0p0075}"
source_evaluation="${ASTROSEG_SOURCE_EVALUATION:-multichannel_lr_sweep_0p001_0p01_20260816/lr_0p0075}"
output_name="${ASTROSEG_CALIBRATION_NAME:-multichannel_lr_0p0075_strict_nuclei}"

module purge
module load python/3.11.13
source .venv-cluster/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python scripts/evaluate_instance_cross_validation.py \
    --run-dir "${data_root}/outputs/checkpoints/${run_name}" \
    --raw-heads-dir "${data_root}/outputs/evaluations/${source_evaluation}/raw_heads" \
    --output-dir "${data_root}/outputs/evaluations/${output_name}" \
    --tuning-image-list configs/cellpose_split_original_channels/validation_ids.txt \
    --foreground-thresholds 0.65,0.70,0.75,0.80,0.85 \
    --boundary-thresholds 0.65,0.70,0.75,0.80,0.85 \
    --max-nucleus-distances 2,4,8 \
    --min-nucleus-foreground-fractions 0.0,0.05,0.10,0.20,0.30 \
    --nucleus-support-expansion 4
