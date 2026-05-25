# dl multimodal 5fold

## Purpose

Validate the current multimodal Cycle-Aware BiLSTM setting with 5-fold wafer-level CV.

## Setup

- Source branch: `origin/exp/late-drift-pool`
- Source commits included by merge: `2637aaf`, `a3ebfe2`
- Config: `configs/exp_dl_multimodal_5fold.yaml`
- Experiment dir recorded in log: `outputs/experiments/2026-05-21_02-09_dl-multimodal-5fold`
- Cache: `cache/v1`
- Split: `splits/kfold5_wafer.npz`
- Targets: `oxide_etch`, `si_etch`
- Modality: multimodal
- Pooling: mean

## Results

- `oxide_etch`: RMSE mean `0.050740`, R2 mean `0.542603`
- `si_etch`: RMSE mean `0.396165`, R2 mean `0.987210`

## Artifacts

- `metrics.json`: aggregate and per-fold metrics
- `logs/epoch_log.csv`: epoch-level training log
- `logs/fold_metrics.csv`: fold-level metrics table
- `logs/sample_predictions.csv`: saved predictions
- `logs/stdout.log`: original stdout log

Related comparison: `outputs/experiments/2026-05-25_18-00_late-drift-vs-mean-compare/`.

## Notes

This experiment was received from a teammate merge and reorganized from `results/` into the project-standard `outputs/experiments/` layout.
