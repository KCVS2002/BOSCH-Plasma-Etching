---
name: project_results
description: 완료된 실험 결과 수치 및 주요 발견사항 (2026-05-08 확인, fold 0 single-fold; 5-fold 확장은 미실시)
type: project
---
## 실험 결과 (oxide_etch 중심, fold 0 기준)

| 모델 | oxide_etch R² | oxide_etch RMSE | si_etch R² |
|---|---|---|---|
| Spatial-mean baseline | 0.156 | — | 0.985 |
| XGBoost (5-fold avg) | 0.551 | 0.0514 | 0.991 |
| DL OES-only | 0.346 | 0.0626 | 0.994 |
| DL Proc-only | 0.641 | 0.0464 | 0.995 |
| DL Multimodal ★ | 0.734 | 0.0399 | 0.996 |

**Why:** DL 실험은 현재 fold 0 단일 검증. 5-fold 전체 실행은 향후 예정.

## 주요 발견

1. **si_etch는 공간 패턴 지배** — Spatial baseline R²=0.985. 모든 모델이 R²>0.99. 모델 변별력 없음.

2. **oxide_etch가 핵심 타겟** — Spatial baseline 실패(R²=0.156). cycle-aware DL만이 의미 있는 성능 달성.

3. **OES-only (0.346) < XGBoost (0.551)** — OES 원시 시계열 단독으로는 feature-engineered XGBoost보다 열세. Process 정보가 oxide 예측에 필수.

4. **Proc-only (0.641) > XGBoost (0.551)** — cycle-aware 시계열 구조 자체의 효과 증거. XGBoost도 Process 피처를 쓰지만 통계 요약 방식의 한계.

5. **Multimodal (0.734) 최고** — OES와 Process가 상호 보완적. Cross-modal 이득 +0.094 R² (Proc-only 대비).

## XGBoost SHAP 분석 (oxide_etch, fold 0)

- 1위 피처: `proc_Heater2Temp_late` (후반부 히터 온도)
- 2위: `X` 좌표
- 그룹별: Process Temporal(0.038) ≈ XY(0.037) > Process Static(0.025) > OES(~0.021)
- **핵심 해석:** 공정 후반의 열적 drift가 oxide etch의 가장 강력한 예측 인자

## DL Gradient Attribution (Multimodal, oxide_etch, fold 0)

- 초반 사이클(1~30): 낮은 importance
- 후반 사이클(80~100): 급격히 상승, 특히 95~100 피크
- 후반부 std 리본 매우 넓음 → 웨이퍼마다 후반 drift 패턴이 다름
- **XGBoost와 수렴하는 결론:** "후반 사이클의 공정 상태 변화가 oxide_etch를 결정한다"

## 출처 폴더 (2026-05-08 확인)

- XGB baseline: `outputs/experiments/2026-04-30_15-32_baseline-xgb/` (5-fold)
- DL Multimodal ★: `outputs/experiments/2026-05-01_00-56_dl-multimodal-singlefold/`
- DL OES-only: `outputs/experiments/2026-05-01_13-00_dl-oes-only-singlefold/`
- DL Proc-only: `outputs/experiments/2026-05-04_11-18_dl-proc-only-singlefold/`
- DL Attention pool (실패한 ablation): `outputs/experiments/2026-05-01_03-16_dl-oxide-v2-attn-singlefold/` — oxide R²=0.643 (multimodal best 0.734 보다 -0.09)

## 추가 메시지 (2026-05-08 발표 후)

- **Multimodal 이 XGB 대비 oxide RMSE -22%** (0.0514 → 0.0399) → 졸업논문 종료기준 1번 (RMSE -20% 이상) 달성.
- **종료기준 2번 (fold std/mean ≤ 10%) 은 미검증** — 모든 DL 결과가 single-fold. 다음 우선순위 작업.

## How to apply

새 실험 결과가 나오면 이 파일의 표를 업데이트할 것. 발표 자료나 논문 작성 시 위 수치를 기준으로 사용.
