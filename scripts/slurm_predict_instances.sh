#!/usr/bin/env bash
#SBATCH --job-name=astroseg-predict
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

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
export ASTROSEG_DATA_ROOT="${data_root}"
export ASTROSEG_DATASET_NAME="${ASTROSEG_DATASET_NAME:-multichannel_20260816}"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cluster/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

train_run_name="${ASTROSEG_TRAIN_RUN_NAME:-multichannel_cross_validation}"
if [[ -n "${ASTROSEG_CHECKPOINT:-}" ]]; then
    checkpoints=("${ASTROSEG_CHECKPOINT}")
else
    shopt -s nullglob
    checkpoints=("${data_root}/outputs/checkpoints/${train_run_name}"/fold_*/best.pt)
    shopt -u nullglob
fi
if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "No trained checkpoints found for ${train_run_name}" >&2
    exit 1
fi
run_name="${ASTROSEG_RUN_NAME:-multichannel_test_ensemble}"
output_dir="${data_root}/outputs/instance_predictions/${run_name}"
evaluation="${ASTROSEG_EVALUATION:-${data_root}/outputs/evaluations/${train_run_name}/evaluation.json}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/predict_astrocyte_instances.py \
    --config configs/train_instances_cluster.yaml \
    --checkpoint "${checkpoints[@]}" \
    --evaluation "${evaluation}" \
    --split test \
    --output-dir "${output_dir}" \
    --output-manifest "${output_dir}/manifest.csv"
