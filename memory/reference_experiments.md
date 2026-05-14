---
name: Key Experiment References
description: Pointers to the canonical experiment folders — which one is the "best", which are ablations, which to compare against
type: reference
---

`outputs/experiments/` 에 14개 폴더가 시간순으로 쌓여 있음. 다른 agent 가 "어느 게 최신/최고인지" 헷갈리지 않도록 라벨링.

## 인용해야 할 canonical 실험 (논문/발표 결과의 출처)

| 라벨 | 폴더 | 용도 |
|---|---|---|
| **XGB baseline (Phase 2 main)** | `outputs/experiments/2026-04-30_15-32_baseline-xgb/` | 모든 비교의 기준선. 5-fold. SHAP figure 포함. |
| **DL multimodal best ★** | `outputs/experiments/2026-05-01_00-56_dl-multimodal-singlefold/` | Phase 3 main 결과. FiLM+Fourier+mean pool. cycle_attribution figure 포함. |
| **DL OES-only ablation** | `outputs/experiments/2026-05-01_13-00_dl-oes-only-singlefold/` | Modality ablation: OES 단독. |
| **DL Process-only ablation** | `outputs/experiments/2026-05-04_11-18_dl-proc-only-singlefold/` | Modality ablation: Proc 단독. |
| **DL attention pool (실패한 ablation)** | `outputs/experiments/2026-05-01_03-16_dl-oxide-v2-attn-singlefold/` | "Attention pool 시도했지만 mean 보다 나쁨" 보고용. |

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
