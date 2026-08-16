#!/usr/bin/env bash
#SBATCH --job-name=astroseg-v2-predict
#SBATCH --partition=gpu.q
#SBATCH --qos=gpu-qos
#SBATCH --constraint=teslaV100S|A100|rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=cluster_logs/%x_%A_%a.out
#SBATCH --error=cluster_logs/%x_%A_%a.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
export ASTROSEG_DATA_ROOT="${data_root}"
export ASTROSEG_DATASET_NAME="${ASTROSEG_DATASET_NAME:-multichannel_20260816}"
run_name="${ASTROSEG_RUN_NAME:-astroseg_v2_full_20260816}"

module purge
module load python/3.11.13
module load cuda/13.0
source .venv-cluster/bin/activate

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

splits=(train test)
split="${splits[${SLURM_ARRAY_TASK_ID}]}"
output_dir="${data_root}/outputs/predictions/${run_name}/${split}"
if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite predictions: ${output_dir}" >&2
    exit 1
fi

checkpoints=(
    "${data_root}/outputs/checkpoints/${run_name}/seed_42/ema_final.pt"
    "${data_root}/outputs/checkpoints/${run_name}/seed_1337/ema_final.pt"
    "${data_root}/outputs/checkpoints/${run_name}/seed_2026/ema_final.pt"
)
for checkpoint in "${checkpoints[@]}"; do
    test -f "${checkpoint}"
done

python scripts/predict_astrocyte_instances.py \
    --config configs/train_astroseg_v2_cluster.yaml \
    --checkpoint "${checkpoints[@]}" \
    --split "${split}" \
    --output-dir "${output_dir}" \
    --output-manifest "${output_dir}/manifest.csv"
