---
name: project_results
description: 종합 실험 결과 수치표 — 2026-05-27 기준. 5-fold wafer CV + LOO-Lot CV 포함. oxide_etch 중심.
metadata:
  type: project
---

## 5-fold Wafer CV 결과 (oxide_etch)

| 모델 | R² (mean±std) | RMSE (mean±std) | fold 4 R² | 비고 |
|---|---|---|---|---|
| XGB baseline | 0.551±0.082 | 0.0514±0.004 | 0.621 | 기준선 |
| DL 5-fold (no wavelen sel) | 0.543±0.232 | 0.0507±0.013 | 0.149 | fold 4 collapse 최초 발견 |
| DL topk256 corr | 0.592±0.151 | 0.0476±0.010 | 0.376 | OES selection 효과 |
| DL longrun (lr=5e-4, ep=80) | 0.596±0.163 | 0.0474±0.011 | 0.320 | optimization 한계 확인 |
| **DL aux-loss ★** | **0.621±0.138** | **0.0468±0.009** | **0.397** | **current best** |
| DL pwnorm+instnorm | 0.561±0.209 | 0.0499±0.012 | 0.202 | 실패. fold 4 악화 |

## LOO-Lot CV 결과 (aux-loss 모델, oxide_etch)

| Held-out Lot | Day | Wafers | RMSE | R² | best_ep | target_mean_shift |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Lot 1 | 07-02 | 9 | 0.0328 | **0.806** | 14 | -0.079 |
| Lot 2 | 07-05 | 10 | 0.0430 | **0.681** | 13 | -0.087 |
| Lot 3 | 07-09 | 10 | 0.0698 | 0.236 | 10 | +0.021 |
| Lot 4 | 07-11 | 10 | 0.0731 | **-0.186** | 0 | +0.040 |
| Lot 5 | 07-19 | 10 | 0.0462 | 0.371 | 54 | +0.018 |
| Lot 6 | 08-01 | 10 | 0.0590 | **-0.007** | 0 | +0.006 |
| Lot 7 | 08-05 | 6 | 0.0376 | 0.560 | 32 | +0.028 |
| Lot 8 | 08-07 | 10 | 0.0516 | 0.168 | 17 | +0.029 |
| Lot 9 | 08-21 | 9 | 0.0421 | 0.451 | 28 | +0.027 |
| Lot 10 | 08-22 | 4 | 0.0672 | 0.152 | 32 | +0.008 |
| **Aggregate** | | | **0.0522±0.014** | **0.323±0.294** | | |

## Single-fold Modality Ablation (fold 0, oxide_etch)

| Modality | R² | RMSE |
|---|---|---|
| OES-only | 0.346 | 0.0626 |
| Proc-only | 0.640 | 0.0464 |
| Multimodal | 0.734 | 0.0399 |

## 핵심 발견 (논문 메시지)

1. **Multimodal DL이 XGB 대비 oxide RMSE -9%** (5-fold: 0.0514→0.0468). single-fold에서는 -22%.
2. **aux-loss가 fold 4 개선에 유일하게 유효** — 0.32→0.40.
3. **LOO-Lot R²=0.323** — 새 lot 일반화가 최대 약점. 5-fold R²=0.621의 절반.
4. **Lot 4, 6은 학습 자체 실패** (best_ep=0) — 해당 lot의 공정 특성이 나머지 9개 lot과 이질적.
5. **si_etch는 전 모델 R²≈0.99** — 모델 변별력 없음. oxide가 유일한 contribution 타겟.

## 출처 폴더

- XGB baseline: `2026-04-30_15-32_baseline-xgb/`
- DL aux-loss ★: `2026-05-27_04-16_dl-multimodal-oes-aux-wafer-mean-5fold/`
- LOO-Lot: `2026-05-27_16-06_dl-lot-validation-oxide-aux/`
- pwnorm+instnorm: `2026-05-27_19-02_dl-multimodal-pwnorm-instnorm-5fold/`
