---
name: Key Experiment References
description: 실험 폴더 라벨링 — best model, ablations, diagnostics. 2026-05-27 기준 aux-loss가 best, LOO-Lot 검증 완료.
metadata:
  type: reference
---

## 현재 Best Model

| 라벨 | 폴더 | 성능 |
|---|---|---|
| **DL aux-loss 5-fold ★ (current best)** | `2026-05-27_04-16_dl-multimodal-oes-aux-wafer-mean-5fold` | oxide R² 0.621±0.138, RMSE 0.0468±0.009 |

## Canonical 실험 (논문/발표 인용 대상)

| 라벨 | 폴더 | 용도 |
|---|---|---|
| **XGB baseline (Phase 2)** | `2026-04-30_15-32_baseline-xgb/` | 기준선. 5-fold. oxide R²=0.551. SHAP figure 포함. |
| **XGB baseline (재실행)** | `2026-05-27_14-00_baseline-xgb/` | 2026-05-27 시점 재실행. 비교 검증용. |
| **DL multimodal best (single-fold)** | `2026-05-01_00-56_dl-multimodal-singlefold/` | Phase 3 single-fold. FiLM+Fourier+mean pool. fold 0 oxide R²=0.734. |
| **DL 5-fold baseline (no wavelen sel)** | `2026-05-21_02-09_dl-multimodal-5fold/` | 첫 5-fold. fold 4 collapse 최초 발견 (R²=0.149). |
| **DL OES topk256 corr 5-fold** | `2026-05-26_02-18_dl-multimodal-oes-corr-topk-5fold/` | OES wavelength selection 첫 성공. R² 0.592. |
| **DL longrun 5-fold** | `2026-05-26_23-24_dl-multimodal-oes-topk256-longrun-5fold/` | lr=5e-4, ep=80. R² 0.596. |
| **DL aux-loss 5-fold ★** | `2026-05-27_04-16_dl-multimodal-oes-aux-wafer-mean-5fold/` | **Current best.** R² 0.621. fold 4 R²=0.397. |
| **DL pwnorm+instnorm 5-fold** | `2026-05-27_19-02_dl-multimodal-pwnorm-instnorm-5fold/` | per-wafer norm + InstanceNorm. R² 0.561. fold 4 악화→0.202. |
| **LOO-Lot 검증 (aux-loss)** | `2026-05-27_16-06_dl-lot-validation-oxide-aux/` | Leave-One-Lot-Out 10-fold. R² 0.323±0.294. lot 일반화 약점 확인. |

## Modality Ablation

| 라벨 | 폴더 | 결과 |
|---|---|---|
| **DL OES-only** | `2026-05-01_13-00_dl-oes-only-singlefold/` | oxide R²=0.346 |
| **DL Proc-only (single)** | `2026-05-04_11-18_dl-proc-only-singlefold/` | oxide R²=0.640 |
| **DL Proc-only (5-fold)** | `2026-05-26_01-42_dl-proc-only-singlefold/` | fold 4 R²=0.134 → DL process encoder 한계 |
| **DL attention pool (실패)** | `2026-05-01_03-16_dl-oxide-v2-attn-singlefold/` | oxide R²=0.643 (mean 0.734보다 나쁨) |

## fold 4 진단 실험 (2026-05-27)

| 라벨 | 폴더 | 결론 |
|---|---|---|
| **seed sweep** | `2026-05-27_00-58~01-30_*-seed{0,1,2,42,100}-fold4/` | 5 seed R²=0.30~0.41, residual corr=0.94 → structural |
| **multi-stat pool** | `2026-05-27_02-31_dl-multimodal-oes-multistat-pool-5fold/` | fold 4 불변. wafer_repr collapse 확인 |

## 실패한 시도 (반례 인용 가능)

- **drift OES wavelength**: `2026-05-26_18-38_*-drift-topk-5fold/` — late_mean보다 모든 fold 악화
- **top-k=128**: `2026-05-26_22-29_*-topk128-5fold/` — 256과 동일 결과
- **pwnorm+instnorm**: `2026-05-27_19-02_*-pwnorm-instnorm-5fold/` — fold 4 악화

## 무시해도 되는 폴더

- `2026-04-29_16-20_dl-smoke` — smoke test
- `2026-04-29_17-03~17-39_*-sanity*` — sanity check
- `2026-04-29_18-21~2026-04-30_22-41_*-singlefold` — FiLM 도입 전 디버깅
- `2026-04-30_23-14_*-singlefold` — FiLM 없는 케이스 (si RMSE 1.91, 반례 인용 O)
- `2026-04-30_13-46_*-singlefold` — Phase 3 초기 시도
- `2026-05-25_*` — late-drift pool, procfilm, seed-reset 등 중간 실험
- `2026-05-26_00-39_seed-sweep-diagnosis` — 진단 스크립트 실행 결과
- `2026-05-27_14-39_dl-lot-validation-oxide-aux` — 중단된 LOO 실행 (16-06이 완성본)

## 참고

- metrics.json이 1차 truth. NOTES.md는 자동 시드만.
- 폴더 이름 = `<YYYY-MM-DD_HH-MM>_<slug>` 시간순 정렬.
- LOO-Lot split: `cache/v1/splits/loo_lot.npz` (10-fold, fold_lot_mapping=[1..10])
- LOO-Lot 실행: `.venv\python.exe -m scripts.04_train_dl --config configs/exp_dl_lot_validation.yaml`
