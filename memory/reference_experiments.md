---
name: Key Experiment References
description: Pointers to the canonical experiment folders — which one is the "best", which are ablations, which to compare against
type: reference
---

`outputs/experiments/` 에 14개 폴더가 시간순으로 쌓여 있음. 다른 agent 가 "어느 게 최신/최고인지" 헷갈리지 않도록 라벨링.

## 인용해야 할 canonical 실험 (논문/발표 결과의 출처)

| 라벨 | 폴더 | 용도 |
|---|---|---|
| **XGB baseline (Phase 2 main)** | `outputs/experiments/2026-04-30_15-32_baseline-xgb/` | 모든 비교의 기준선. 5-fold. SHAP figure 포함. fold 4 R²=0.62. |
| **DL multimodal best (single-fold)** | `outputs/experiments/2026-05-01_00-56_dl-multimodal-singlefold/` | Phase 3 single-fold main. FiLM+Fourier+mean pool. cycle_attribution figure 포함. |
| **DL multimodal 5-fold baseline (no wavelength sel)** | `outputs/experiments/2026-05-21_02-09_dl-multimodal-5fold/` | 첫 5-fold 확장. fold 4 R²=0.149 (collapse 최초 발견). |
| **DL OES top-k=256 corr 5-fold ★** | `outputs/experiments/2026-05-26_02-18_dl-multimodal-oes-corr-topk-5fold/` | OES wavelength selection 첫 성공. aggregate R² 0.592, fold 4 R²=0.376 (이전 0.149 대비 +0.227). |
| **DL longrun (lr=5e-4, ep=80) 5-fold** | `outputs/experiments/2026-05-26_23-24_dl-multimodal-oes-topk256-longrun-5fold/` | best aggregate R² 0.596. fold 4 R²=0.32 (longrun도 더 못 잡음 — optimization 한계 확인). |
| **DL OES-only ablation** | `outputs/experiments/2026-05-01_13-00_dl-oes-only-singlefold/` | Modality ablation: OES 단독. R²=0.346. |
| **DL Process-only ablation (single)** | `outputs/experiments/2026-05-04_11-18_dl-proc-only-singlefold/` | Modality ablation: Proc 단독. R²=0.640. |
| **DL Process-only ablation (5-fold)** | `outputs/experiments/2026-05-26_01-42_dl-proc-only-singlefold/` | Proc 5-fold. fold 4 R²=0.134 → DL의 process encoder가 XGB stats 추출 못함 확인. |
| **DL attention pool (실패한 ablation)** | `outputs/experiments/2026-05-01_03-16_dl-oxide-v2-attn-singlefold/` | "Attention pool 시도했지만 mean 보다 나쁨" 보고용. |

## fold 4 진단 실험 (2026-05-27, 모두 collapse 확정용)

| 라벨 | 폴더 | 결과 / 결론 |
|---|---|---|
| **fold 4 seed sweep** | `outputs/experiments/2026-05-27_00-58~01-30_dl-multimodal-oes-corr-topk-5fold-seed{0,1,2,42,100}-fold4/` | 5 seed R² 0.30~0.41, residual corr=0.94, 5-seed ensemble R²=0.39 → **optimization-induced 아님 확정** |
| **multi-stat pool** | `outputs/experiments/2026-05-27_02-31_dl-multimodal-oes-multistat-pool-5fold/` | [mean,std,max,slope] pool. fold 4 R²=0.33 (불변). 11 wafer pred 4자리까지 동일 → **pool은 병목 아님 확정** |

## 실패한 ablation (반례 인용 가능)

- **drift OES wavelength selection**: `outputs/experiments/2026-05-26_18-38_dl-multimodal-oes-drift-topk-5fold/` — late_mean보다 모든 fold 악화 (fold 4 0.165). 88 wafer×~70 train 규모에서 late-early 차분은 noise 가 더 큼.
- **top-k=128**: `outputs/experiments/2026-05-26_22-29_dl-multimodal-oes-topk128-5fold/` — top-k=256과 사실상 동일 결과. wavelength 추가 줄이기 무의미.

## 무시해도 되는 폴더 (스모크/디버깅)

- `2026-04-29_16-20_dl-smoke` — 초기 smoke test
- `2026-04-29_17-03_dl-multimodal-sanity`, `..._17-39_..._sanity-v2` — sanity check
- `2026-04-29_18-21_dl-multimodal-singlefold` ~ `2026-04-30_22-41_..._singlefold` — FiLM 도입 전 디버깅 시도들 (si RMSE 폭발 케이스 포함)
- `2026-04-30_23-14_dl-multimodal-singlefold` — FiLM 없이 돌린 케이스 (si RMSE 1.91, 반례로 인용 가치 O)
- `2026-04-30_13-46_dl-multimodal-singlefold` — Phase 3 초기 시도

## 참고

- 모든 metrics 는 폴더 내 `metrics.json` 이 1차 truth. NOTES.md 는 자동 시드만 들어있고 비어 있는 경우 많음.
- 폴더 이름 = `<YYYY-MM-DD_HH-MM>_<slug>` 으로 시간순 정렬. 같은 날 여러 개면 시간으로 구분.
- 실험 폴더 내부 구조: `config.yaml` (실행 시 사용된 config 복사본), `metrics.json`, `checkpoints/`, `figures/`, `logs/`, `NOTES.md`.
