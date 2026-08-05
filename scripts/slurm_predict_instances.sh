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

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cluster/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

checkpoint="${ASTROSEG_CHECKPOINT:-${data_root}/outputs/checkpoints/cross_validation/fold_1/best.pt}"
run_name="${ASTROSEG_RUN_NAME:-latest}"
output_dir="${data_root}/outputs/instance_predictions/${run_name}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/predict_astrocyte_instances.py \
    --config configs/train_instances_cluster.yaml \
    --checkpoint "${checkpoint}" \
    --split test \
    --output-dir "${output_dir}" \
    --output-manifest "${output_dir}/manifest.csv"
