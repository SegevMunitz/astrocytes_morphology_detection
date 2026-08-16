#!/usr/bin/env bash
#SBATCH --job-name=astroseg-v2-eval
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=teslaV100S|A100|rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=cluster_logs/%x_%A_%a.out
#SBATCH --error=cluster_logs/%x_%A_%a.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
run_name="${ASTROSEG_RUN_NAME:-astroseg_v2_from_scratch_20260816}"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cluster/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

learning_rates=(0.0003 0.0006 0.001)
learning_rate="${learning_rates[${SLURM_ARRAY_TASK_ID}]}"
slug="${learning_rate/./p}"
run_dir="${data_root}/outputs/checkpoints/${run_name}/lr_${slug}"
output_dir="${data_root}/outputs/evaluations/${run_name}/lr_${slug}"

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing evaluation: ${output_dir}" >&2
    exit 1
fi

python scripts/evaluate_instance_cross_validation.py \
    --run-dir "${run_dir}" \
    --output-dir "${output_dir}" \
    --tuning-image-list configs/cellpose_split_original_channels/validation_ids.txt \
    --foreground-thresholds 0.40,0.50,0.60 \
    --boundary-thresholds 0.40,0.60,0.80 \
    --max-nucleus-distances 4,8 \
    --min-nucleus-foreground-fractions 0.0,0.10 \
    --nucleus-probability-thresholds 0.20,0.40,0.60 \
    --nucleus-support-expansion 4
