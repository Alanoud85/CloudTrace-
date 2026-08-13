# Reproducibility

## Reference artifacts

`reference_outputs/` contains the archived paper run: the run manifest, tables, figures, preprocessing artifacts, identity-relation hash references, robust scaler, and the seed-42 checkpoint.

The archived checkpoint records:

- view dimensions: 47, 21, 12;
- hidden dimension: 96;
- latent dimension: 32;
- component calibration medians and scales;
- score component weights;
- generalized Pareto tail parameters;
- the exact training configuration recorded for that run.

## Full rerun

```bash
python scripts/run_paper_experiment.py \
  --input data/nineteenFeaturesDf.csv \
  --config configs/paper.yaml \
  --output outputs/paper_run
```

The implementation writes a run manifest, audit tables, split tables, feature dictionary, seed-level metrics, aggregate benchmark tables, operational thresholds, future-period drift tables, candidate anomalies, model checkpoints, and figures.

## Determinism

Random seeds are set for Python, NumPy, and PyTorch. CUDA deterministic algorithms are requested when available. Small numerical differences can still occur across PyTorch/CUDA versions and hardware.

## Paper values

`python scripts/reproduce_reference_summary.py` reads the archived tables and prints the headline values without retraining. This provides a direct check that the tracked reference artifacts have not been altered.
