#!/usr/bin/env bash
#SBATCH --job-name=cellpose3-setup
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail
umask 0002

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"
data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
model_cache="${data_root}/outputs/cellpose/pretrained_models"
mkdir -p "${model_cache}"
export CELLPOSE_LOCAL_MODELS_PATH="${model_cache}"

module purge
module load python/3.11.13

# Cellpose 3 is isolated because it requires NumPy <2.1, while the main project
# follows a newer scientific-Python stack.
if [[ ! -x .venv-cellpose3/bin/python ]]; then
    python3 -m venv .venv-cellpose3
fi
source .venv-cellpose3/bin/activate
python -m pip install --upgrade pip
python -m pip install "cellpose==3.1.1.1" "packaging>=24"
python -m cellpose --version
python -c "import importlib.metadata, torch; print('cellpose', importlib.metadata.version('cellpose'), 'torch', torch.__version__)"
python -c "from cellpose import models; print('cyto2_cp3', models.CellposeModel(gpu=False, model_type='cyto2_cp3').pretrained_model); print('cyto3', models.CellposeModel(gpu=False, model_type='cyto3').pretrained_model)"
