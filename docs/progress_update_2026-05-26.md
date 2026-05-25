# Progress Update — 2026-05-26

## Summary

This note records the work done after the mid-term presentation and the current diagnosis around oxide-only 5-fold DL experiments.

The research focus is now narrowed to `oxide_etch`. `si_etch` remains useful as a sanity target, but it is already saturated by spatial patterns across most models, so new DL experiments should train `oxide_etch` only to save GPU time.

## Main Experiments Reviewed

### 1. Baseline Multimodal DL 5-Fold

- Experiment: `outputs/experiments/2026-05-21_02-09_dl-multimodal-5fold/`
- Config: `configs/exp_dl_multimodal_5fold.yaml`
- Model: OES + Process, 2D-CNN encoders, BiLSTM, mean cycle pooling, Fourier xy + xy-FiLM
- Target originally included both `oxide_etch` and `si_etch`; config has now been changed to oxide-only.

Oxide results:

| Fold | RMSE | R2 |
|---:|---:|---:|
| 0 | 0.0391 | 0.745 |
| 1 | 0.0447 | 0.653 |
| 2 | 0.0586 | 0.415 |
| 3 | 0.0384 | 0.751 |
| 4 | 0.0730 | 0.149 |

Observation:

- Folds 0, 1, and 3 perform well.
- Fold 2 is weaker.
- Fold 4 collapses most severely.
- Fold 4 predictions show strong mean shrinkage:
  - true std: about `0.079`
  - prediction std: about `0.028`
  - wafer-mean slope: about `0.073`

This means the model almost loses wafer-level oxide calibration on fold 4.

### 2. Method 5 — Process-Conditioned OES FiLM

- Config: `configs/exp_dl_multimodal_procfilm_singlefold.yaml`
- Experiment: `outputs/experiments/2026-05-25_17-38_dl-multimodal-procfilm-singlefold/`
- New option: `proc_condition_oes: "film"`
- Implementation: Process cycle embedding generates FiLM parameters `(gamma, beta)` to modulate the matching OES cycle embedding before multimodal fusion.

Oxide 5-fold result:

- RMSE: `0.0513 ± 0.0121`
- R2: `0.538 ± 0.215`

Fold 4 still collapsed:

- Fold 4 RMSE: `0.0731`
- Fold 4 R2: `0.147`
- prediction std: about `0.034`
- wafer-mean slope: about `0.084`

Interpretation:

- Process-conditioned OES FiLM did not fix the fold 4 failure.
- It behaves similarly to the mean-pool baseline on the difficult fold.
- The failure is therefore not specific to the new Process-conditioned FiLM module.

### 3. Method 4 — Mean + Late + Drift Pooling

- Config: `configs/exp_dl_late_drift_oxide_5fold.yaml`
- Comparison artifact: `outputs/experiments/2026-05-25_18-00_late-drift-vs-mean-compare/`
- Pooling: `mean_late_drift`
- Representation: concatenate global cycle mean, late-cycle mean, and late-minus-early drift.

Comparison summary:

- Mean pooling remains better on average.
- Late-drift pooling improved folds 2 and 4, but worsened the aggregate score.
- Fold 4 improved slightly from about `0.0713` to `0.0688` in the comparison table, but it remained much worse than folds 0, 1, and 3.

Interpretation:

- Explicit late/drift information helps fold 4 a little, but not enough.
- Fold 4 likely needs more stable process temporal calibration than this simple pooling change provides.

## Fold 4 Diagnosis

The repeated fold 4 degradation is the most important current issue.

### Split Type

The split is not an ordinary random sample split. It is:

```text
StratifiedGroupKFold(n_splits=5, stratify=lot, group=wafer)
```

Meaning:

- All 89 points from one wafer stay together.
- Lot composition is approximately balanced across folds.
- Because there are only 88 wafers, each validation fold contains only 17 or 18 wafers, so wafer-level composition still matters a lot.

### Fold 4 Is Not Impossible

XGBoost performs well on fold 4:

- XGBoost fold 4 oxide RMSE: `0.0487`
- XGBoost fold 4 oxide R2: `0.621`

Therefore fold 4 is not inherently unpredictable. The issue is specific to the current DL representation/training.

### Fold 4 Distribution

Fold 4 has a normal-looking average oxide level:

| Fold | Wafer Mean Oxide |
|---:|---:|
| 0 | 0.656 |
| 1 | 0.657 |
| 2 | 0.660 |
| 3 | 0.664 |
| 4 | 0.653 |

However, fold 4 contains both low-oxide and high-oxide wafers that the DL model tends to confuse.

Low-oxide examples:

| Wafer | Oxide Mean |
|---|---:|
| `2024-07-05_09` | 0.573 |
| `2024-07-02_08` | 0.575 |
| `2024-07-02_09` | 0.576 |
| `2024-07-05_04` | 0.582 |

High-oxide examples:

| Wafer | Oxide Mean |
|---|---:|
| `2024-07-11_01` | 0.699 |
| `2024-07-11_02` | 0.696 |
| `2024-07-09_10` | 0.695 |

The fold 4 problem is therefore better described as wafer-level calibration failure, not simple target mean shift.

## Seed Reset Experiment

The DL training script originally called `set_seed(seed)` only once at the beginning of the script. This meant fold 4 always received the random state after folds 0 through 3 had already consumed RNG state.

The training script has now been changed so that each target/fold resets the RNG:

```python
fold_seed = base_seed + target_idx * 1000 + f
set_seed(fold_seed)
```

For oxide-only training:

```text
fold 0 -> seed 42
fold 1 -> seed 43
fold 2 -> seed 44
fold 3 -> seed 45
fold 4 -> seed 46
```

A fold 4-only rerun was performed:

- Experiment: `outputs/experiments/2026-05-25_23-27_dl-multimodal-5fold/`
- Command pattern: `scripts.04_train_dl --config configs/exp_dl_multimodal_5fold.yaml --folds 4`

Result:

| Run | RMSE | R2 | Prediction Std | Wafer-Mean Slope |
|---|---:|---:|---:|---:|
| Original fold 4 | 0.0730 | 0.149 | 0.028 | 0.073 |
| Seed-reset fold 4 | 0.0677 | 0.268 | 0.050 | 0.312 |

Interpretation:

- Seed reset improved fold 4.
- The extreme mean-shrinkage failure was reduced.
- However, fold 4 remains weaker than the other folds.
- This suggests two effects are mixed:
  1. seed/optimization instability;
  2. fold-specific DL representation difficulty.

## New Diagnostic Script

Added:

```text
scripts/08_diagnose_seed_sweep.py
```

Purpose:

- Run the same fold with multiple experiment seeds.
- Summarize RMSE, R2, prediction std, wafer-mean slope, and worst wafers.
- Decide whether fold 4 failure is a one-seed optimization issue or robust across seeds.

Run fold 4 seed sweep:

```powershell
.\.venv\Scripts\python.exe -m scripts.08_diagnose_seed_sweep --config configs/exp_dl_multimodal_5fold.yaml --fold 4 --seeds 42,43,44,45,46 --run
```

Summarize existing experiments only:

```powershell
.\.venv\Scripts\python.exe -m scripts.08_diagnose_seed_sweep --experiments outputs/experiments/2026-05-25_23-27_dl-multimodal-5fold
```

The script writes a diagnosis experiment folder containing:

- `logs/seed_sweep_summary.csv`
- `logs/worst_wafers.csv`
- `metrics.json`
- per-run training logs

## Code / Config Changes

### Training Script

File:

```text
scripts/04_train_dl.py
```

Changes:

- Added fold-specific seed reset.
- Added `--folds` option for running selected fold ids, e.g. `--folds 4` or `--folds 2,4`.

### Configs Changed to Oxide-Only

The following DL configs now use:

```yaml
targets: ["oxide_etch"]
```

Changed configs:

- `configs/exp_dl_multimodal_5fold.yaml`
- `configs/exp_dl_multimodal_procfilm_singlefold.yaml`
- `configs/exp_dl_multimodal_singlefold.yaml`
- `configs/exp_dl_multimodal.yaml`
- `configs/exp_dl_proc_only.yaml`
- `configs/exp_dl_oes_only.yaml`
- `configs/exp_dl_oxide_v2_singlefold.yaml`
- `configs/exp_dl_multimodal_xgbfeat_singlefold.yaml`
- `configs/exp_dl_multimodal_xgbfeat_k5_singlefold.yaml`

Rationale:

- `si_etch` is spatially saturated and not the main research contribution.
- Skipping `si_etch` saves substantial GPU time.
- Future DL experiments should focus on `oxide_etch`.

## Current Working Interpretation

The current evidence suggests:

1. Fold 4 is not impossible, because XGBoost performs well there.
2. Current DL models struggle to recover the wafer-level oxide calibration on fold 4.
3. The failure appears as prediction variance shrinkage and low wafer-mean slope.
4. Seed reset partially improves the issue, but does not fully solve it.
5. XGBoost's advantage on fold 4 suggests that explicit process temporal features such as late, slope, and drift may be important.

Practical next steps:

1. Run seed sweep on fold 4 using `scripts/08_diagnose_seed_sweep.py`.
2. If all seeds remain weak, treat fold 4 as a robust DL representation failure.
3. Compare DL fold 4 errors against XGBoost SHAP/top temporal features.
4. Consider residual correction or explicit process feature injection as more promising than simply adding model complexity.
