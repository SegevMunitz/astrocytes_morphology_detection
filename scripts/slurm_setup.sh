#!/usr/bin/env bash
#SBATCH --job-name=astroseg-setup
#SBATCH --partition=elscn.q
#SBATCH --qos=elsc-qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=.astroseg_runtime/cluster_logs/%x_%j.out
#SBATCH --error=.astroseg_runtime/cluster_logs/%x_%j.err

set -euo pipefail

module purge
module load python/3.11.13

# The HUJI module exposes the selected interpreter as ``python3``; the generic
# ``python`` command still points at the operating-system Python 3.9.
python3 -m venv --clear .venv-cluster
source .venv-cluster/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"
