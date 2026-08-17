#!/usr/bin/env bash
#SBATCH --job-name=astroseg-v2-trainfit
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
run_name="${ASTROSEG_RUN_NAME:-astroseg_v2_full_20260816}"
prediction_root="${data_root}/outputs/predictions/${run_name}/train"
output_dir="${data_root}/outputs/evaluations/${run_name}/training_fit_only"

if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite training-fit evaluation: ${output_dir}" >&2
    exit 1
fi

module purge
module load python/3.11.13
source .venv-cluster/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

python scripts/evaluate_astrocyte_instances.py \
    --manifest "${prediction_root}/manifest.csv" \
    --output-dir "${output_dir}" \
    --iou-threshold 0.5

printf '%s\n' \
    'These metrics use images seen during training.' \
    'They are an overfit/sanity diagnostic and must not be reported as test performance.' \
    > "${output_dir}/NOT_A_TEST_SET.txt"
