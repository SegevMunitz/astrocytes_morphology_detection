#!/usr/bin/env bash
# Submit the independent-model workflow, Cellpose refinement, or both.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

usage() {
    cat <<'EOF'
Usage: scripts/submit_training_workflows.sh [--dry-run] WORKFLOW

WORKFLOW:
  new-model         train AstroSeg v2 from random initialization
  refine-cellpose   refine the configured Cyto2 checkpoint as a 3-channel model
  both              submit both independent Slurm arrays concurrently

Optional environment variables:
  ASTROSEG_DATA_ROOT, ASTROSEG_DATASET_NAME, ASTROSEG_RUN_NAME
  CELLPOSE_TRANSFER_SOURCE, CELLPOSE_TRANSFER_RUN_NAME
EOF
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
workflow="${1:-}"
if [[ -z "${workflow}" || $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

submit() {
    local job_script="$1"
    if [[ "${dry_run}" == "1" ]]; then
        printf 'sbatch %q\n' "${job_script}"
    else
        sbatch "${job_script}"
    fi
}

case "${workflow}" in
    new-model)
        submit scripts/slurm_sweep_astroseg_v2.sh
        ;;
    refine-cellpose)
        submit scripts/slurm_train_cellpose_three_channel.sh
        ;;
    both)
        submit scripts/slurm_sweep_astroseg_v2.sh
        submit scripts/slurm_train_cellpose_three_channel.sh
        ;;
    *)
        echo "Unknown workflow: ${workflow}" >&2
        usage >&2
        exit 2
        ;;
esac
