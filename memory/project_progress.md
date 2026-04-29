---
name: BOSCH Plasma Etching Project Progress
description: Phase-by-phase progress log for the virtual metrology research project
type: project
---

## 현재까지 완료된 것 (2026-04-29 기준)

### Phase 1 — 데이터 전처리 + 캐시 빌드 ✅ 완료

**핵심 데이터 사실:**
- 96 wafer NetCDF, 그 중 측정값 있는 88 wafer만 사용 (Si_Oxide_etch_89_points.csv)
- 89 측정 포인트/wafer → 총 7,832 sample
- 10 Lot (날짜 기반), DAY_TO_LOT 매핑: 2024_07_02=1 ... 2024_08_22=10
- OES 채널: 3,648 wavelength, ~14,744 time steps (uint16)
- Process 채널: **전체 44개지만 31개만 모든 wafer에 공통** (13채널은 9/88 wafer만 기록)
  - 누락 채널 예: Gas6Flow, Heater5~8Temp, ThermoCouple1~4Temp, attenuatorRatio, moriOuterCurrent
- 사이클 구조: SF6(4.5s ON) + C4F8(1.5s ON) = 6s/cycle × 100 cycles + ignition 1s
  - 원시 edge 수: 100개(10 wafer) / 101개(83 wafer) / 102개(3 wafer) → end-anchored 100 trim
- 1개 wafer (`2024-08-07_08`) alignment offset 이상치 (+118s) — 현재 캐시에 이대로 저장됨, 추후 확인 필요

**생성 파일:**
- `cache/v1/wafers/*.npz` — 88 wafer 각각 (7.2 GB 총)
- `cache/v1/measurements.csv` — 7832×11 측정 테이블
- `cache/v1/splits/kfold5_wafer.npz` — 메인 K=5 StratifiedGroupKFold (seed=42)
- `cache/v1/splits/loo_lot.npz` — Leave-One-Lot-Out (10 folds)
- `cache/v1/features/baseline_xgb_v1.csv` — 88 wafer × 328 features (32채널 × 8 stats × OES/Process)

**스크립트:**
- `scripts/01_build_cache.py --version v1`
- `scripts/02_make_splits.py --version v1`

---

### Phase 2 — XGBoost Baseline ✅ 완료

**실험 결과:** `outputs/experiments/2026-04-29_04-21_baseline-xgb/`

| Target | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| si_etch (μm) | 0.3246 ± 0.0885 | 0.1672 ± 0.0211 | 0.9913 ± 0.0046 | 0.38% |
| oxide_etch (μm) | 0.0515 ± 0.0047 | 0.0308 ± 0.0030 | 0.5490 ± 0.0890 | 4.78% |

**분석:**
- `si_etch`: R²=0.99 — 시그널 매우 강함. fold std/mean = 28% → 안정성 기준(10%) 미달
- `oxide_etch`: R²=0.55 — 예측이 어려운 타겟. DL 개선 여지가 큰 핵심 타겟

**Feature 구성 (330 = 328 wafer-level + 2 spatial):**
- OES: 10 bands × 8 stats (mean/std/min/max/slope/early/late/drift) = 80
- Process: 31 공통 채널 × 8 stats = 248
- Spatial: X, Y

**스크립트/설정:**
- `scripts/03_train.py --config configs/exp_baseline_xgb.yaml`
- 모델 checkpoints: 10개 (2 targets × 5 folds), `outputs/experiments/*/checkpoints/*.json`

---

### Phase 3 — Cycle-Aware DL (다음 단계) ⏳ 미착수

**연구계획서 기준 달성 목표:**
- DL RMSE ≤ XGBoost 대비 -20% (si_etch: 0.32×0.8=0.26, oxide_etch: 0.052×0.8=0.041)
- K=5 fold 간 RMSE std ≤ mean × 10%

**계획 아키텍처 (지도교수 피드백 반영):**
- Per-cycle OES Encoder: 2D-CNN (time × wavelength)
- Per-cycle Process Encoder: 2D-CNN (time × channel) — 아키텍처 공유, 가중치 독립
- Cycle sequence: Bi-LSTM (100 사이클 순서)
- Head: wafer_repr + (X, Y) → si_etch / oxide_etch

**다음 할 것:**
1. `src/data/dataset.py` — PyTorch Dataset (cycle tensor 조립, 정규화)
2. `src/models/cycle_encoder.py` — 2D-CNN Encoder
3. `src/models/bilstm_vm.py` — Bi-LSTM + Regression Head
4. `configs/exp_dl_oes_only.yaml` — Exp 3 (OES only, 비교군)
5. `configs/exp_dl_multimodal.yaml` — Exp 4 (★ 제안 방법)

---

## Why: si_etch fold variance 원인 (확인 필요)
- fold 3,4 RMSE ~0.43 vs fold 0,1,2 ~0.25 — Lot 구성 차이인지, 특정 wafer의 이상치인지 미분석
