# late drift vs mean compare

## Purpose

Compare the oxide 5-fold results from the default mean pooling model against the mean-late-drift pooling candidate.

## Setup

- Created at: `2026-05-25 18:00`
- Baseline config: `configs/exp_dl_mean_oxide_5fold.yaml`
- Candidate config: `configs/exp_dl_late_drift_oxide_5fold.yaml`
- Target: `oxide_etch`
- Baseline pooling: `mean`
- Candidate pooling: `mean_late_drift`

## Results

- Mean pooling RMSE mean: `0.053645`
- Mean-late-drift pooling RMSE mean: `0.055352`
- Mean-late-drift minus mean RMSE: `+0.001708`
- Mean pooling R2 mean: `0.504656`
- Mean-late-drift pooling R2 mean: `0.478394`
- Improved folds for late-drift pooling: `2 / 5`

## Takeaway

Mean pooling remains better on average. The late-drift pooling variant improved folds 2 and 4, but the aggregate RMSE and R2 were worse than the mean pooling baseline.

## Artifacts

- `metrics.json`: aggregate comparison metrics
- `logs/late_drift_vs_mean_compare.csv`: fold-level comparison table
