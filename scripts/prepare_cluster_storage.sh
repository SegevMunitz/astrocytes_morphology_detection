#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

data_root="${ASTROSEG_DATA_ROOT:-${HOME}/astroseg_data}"
export ASTROSEG_DATA_ROOT="${data_root}"

for directory in training_images training_masks test_images outputs; do
    if [[ ! -d "${data_root}/${directory}" ]]; then
        echo "Missing cluster storage directory: ${data_root}/${directory}" >&2
        exit 1
    fi
done

# The virtual environment links against the cluster's Python module, so the
# same module must be loaded before invoking its interpreter.
module purge
module load python/3.11.13
source .venv-cluster/bin/activate
python scripts/relocate_runtime_manifest.py \
    --input "${data_root}/outputs/metadata/manifest_instances.csv" \
    --output "${data_root}/outputs/metadata/manifest_instances_cluster.csv" \
    --data-root "${data_root}"

echo "Cluster storage is ready at ${data_root}"
