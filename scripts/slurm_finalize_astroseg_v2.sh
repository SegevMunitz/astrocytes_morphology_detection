#!/usr/bin/env bash
#SBATCH --job-name=astroseg-v2-rank
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
run_name="${ASTROSEG_RUN_NAME:-astroseg_v2_from_scratch_20260816}"
evaluation_root="${data_root}/outputs/evaluations/${run_name}"
ranking_directory="${evaluation_root}/ranking"

module purge
module load python/3.11.13
source .venv-cluster/bin/activate

python scripts/rank_instance_checkpoint_evaluations.py \
    --evaluation-root "${evaluation_root}" \
    --output-dir "${ranking_directory}"

best_directory="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_directory"])' "${ranking_directory}/best_checkpoint.json")"

declare -A baselines=(
    [cyto2]="cellpose_cyto2_cp3_lr_0p25_epoch_0450"
    [cyto3]="cellpose_cyto3_lr_0p35_epoch_0450"
    [three_channel_transfer]="three_channel_transfer_20260816/lr_0p003_epoch_250_gfp_zero"
)
for name in "${!baselines[@]}"; do
    python scripts/compare_instance_models.py \
        --custom-dir "${best_directory}" \
        --cellpose-dir "${data_root}/outputs/evaluations/${baselines[$name]}" \
        --output-dir "${evaluation_root}/comparison_best_vs_${name}"
done
