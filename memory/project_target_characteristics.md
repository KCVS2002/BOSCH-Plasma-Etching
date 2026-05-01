---
name: Target Variance Decomposition (si_etch vs oxide_etch)
description: si_etch is spatial-dominant (99% within-wafer), oxide_etch is process-driven — critical for interpreting all VM results
type: project
---

# 타겟 특성 — VM 결과 해석의 핵심 (2026-05-01 분석)

이 프로젝트의 두 타겟은 분산 구조가 완전히 달라서, **모든 모델 비교/평가는 이 차이를 인지한 채로 해석해야 한다.**

## 분산 분해 (fold 0 기준 측정)

| 타겟 | Within-wafer std | Between-wafer std | 분산 비율 |
|---|---|---|---|
| si_etch | **3.61 μm** | 0.40 μm | within 99% / between 1% |
| oxide_etch | 0.064 μm | 0.044 μm | within 68% / between 32% |

## Spatial-mean baseline (학습 없이, (X,Y) 위치별 train 평균 lookup)

| 타겟 | RMSE | R² |
|---|---|---|
| si_etch | **0.449** | **0.985** |
| oxide_etch | 0.071 | 0.156 |

## 해석

**si_etch는 spatial-dominant.**
- 웨이퍼끼리 평균은 거의 같고 (between std 0.40 μm), 위치별 패턴(within std 3.61 μm)이 모든 웨이퍼에서 거의 동일하게 반복.
- 학습 없이 (X, Y) lookup 만으로 R²=0.985 달성. OES/Process cycle 데이터의 기여는 제한적.
- 도메인 원인: SF₆ 식각의 **loading effect / chamber 공간 비균일성**이 모든 웨이퍼에 동일 패턴으로 찍힘.
- 모델 비교에서 si_etch RMSE 차이는 작게 나오는 게 정상. R²>0.98 못 맞추면 모델이 망가진 것.

**oxide_etch는 process-driven.**
- spatial baseline R²=0.16 → 위치만으론 거의 안 풀림.
- 웨이퍼별 cycle 동역학(RF, pressure, gas ratio 변동)이 결과 결정.
- 본 연구에서 cycle-aware DL 의 가치를 직접 검증하는 타겟.
- 반도체 fab 의 selectivity (Si/oxide etch ratio) 와 mask loss 모니터링이 곧 oxide_etch 예측 문제.

## 졸업논문/평가에서의 framing

- si_etch 는 **sanity target** 으로 다루기 (모두 풀어야 정상).
- oxide_etch 가 **본 연구의 핵심 contribution 검증 타겟**.
- 모든 평가표에 **spatial-mean baseline 을 함께 보고** — VM 분야 evaluation 함정 (Lynn 2009, Susto 2015) 회피.

## How to apply

- 새 모델/실험 결과를 받으면 두 타겟 따로 해석:
  - si_etch RMSE 0.2~0.4 = 정상 범위 (spatial baseline 0.45 부근부터 시작)
  - oxide_etch RMSE 개선이 주된 평가 포인트, R² 변화에 주목
- "DL 이 baseline 이긴다" 주장은 **oxide 결과로 만들 것**. si 차이만으로 주장하면 reviewer 에게 "그거 spatial 보간 차이 아니냐" 반박 받음.
- 새 ablation/실험 설계 시 spatial-only baseline 을 비교군에 항상 포함.

## Why

- 2026-05-01: DL FiLM+Fourier 적용 후 fold 0 결과 (si RMSE 0.232 vs XGB 0.265, oxide 0.040 vs 0.051) 가 "한 번에 너무 좋게 나와서 버그 의심" 으로 시작된 점검에서 발견. 코드 무결성 점검과 동시에 raw 데이터 분산 분해를 수행해 결과의 진위를 데이터 통계로 증명. 결과는 진짜이며, 그 의미는 si는 보간, oxide는 진짜 VM이라는 것.
