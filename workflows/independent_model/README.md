# Independent AstroSeg model

This workflow trains a new model from random initialization. It does not read a
Cellpose checkpoint and does not share weights with the refinement workflow.

The model consumes aligned GFAP/GFP/DAPI inputs and learns foreground,
compartment, touching-boundary, and nucleus-ownership heads. Its main entrypoint
is `scripts/train_instances.py`; cluster settings live in
`configs/train_astroseg_v2_cluster.yaml`.

From the repository root on the cluster:

```bash
export ASTROSEG_DATA_ROOT="$HOME/astroseg_data"
export ASTROSEG_RUN_NAME="astroseg_v2_from_scratch_$(date +%Y%m%d)"
bash scripts/submit_training_workflows.sh new-model
```

Evaluate and rank the completed learning-rate array with:

```bash
sbatch scripts/slurm_evaluate_astroseg_v2.sh
sbatch scripts/slurm_finalize_astroseg_v2.sh
```

Use a new run name for every submission; training scripts refuse to overwrite an
existing run directory.
