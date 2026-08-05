#!/usr/bin/env bash
#SBATCH --job-name=astroseg-setup
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/%x_%j.out
#SBATCH --error=cluster_logs/%x_%j.err

set -euo pipefail

project_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${project_root}"

module purge
module load python/3.11.13

# The HUJI module exposes the selected interpreter as ``python3``; reuse the
# environment after the first setup because PyTorch's CUDA packages are large.
if [[ ! -x .venv-cluster/bin/python ]]; then
    python3 -m venv .venv-cluster
fi
source .venv-cluster/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"
