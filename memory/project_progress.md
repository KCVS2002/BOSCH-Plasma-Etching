---
name: BOSCH Plasma Etching Project Progress
description: Phase-by-phase progress log — current state as of 2026-05-27 (fold 4 collapse root cause confirmed as encoder bottleneck, aux-loss fix pending)
type: project
---

## 한눈에 보기 (2026-05-27 업데이트)

**fold 4 진단 라운드 완료. encoder representational 문제로 확정. wafer-mean aux loss 구현 끝, 실행 대기.**

- **fold 4는 데이터적으로 어렵지 않음** (XGB R²=0.62, 동일 split). DL 고유의 표현력 한계 문제.
- **fold 4 seed sweep (5 seeds, fold 4 only):** [outputs/experiments/2026-05-27_00-58~01-30_dl-multimodal-oes-corr-topk-5fold-seed{0,1,2,42,100}-fold4/](outputs/experiments/)
  - 모든 seed R² = 0.30~0.41 좁은 구간, std=0.046
  - **residual 상관행렬 평균 off-diagonal = 0.94** → 5 seed의 error가 거의 동일
  - **5-seed simple mean ensemble**: R² 0.39 (최선 single 0.41보다 *나쁨*) → ensemble 우회로 차단
  - **결론: optimization-induced가 아니라 structural encoder collapse**
- **OES wavelength selection 인프라 구현됨** ([src/features/oes_selection.py](../src/features/oes_selection.py)): per-fold train-only correlation 기반 top-k. `stat ∈ {mean, late_mean, drift}`. drift는 fold 4 R²=0.165으로 더 나빠짐, late_mean이 안정. top_k=256/128 둘 다 fold 4는 비슷 (~0.38).
- **multi-stat pool ([mean, std, max, slope]) 시도 → 실패**: [outputs/experiments/2026-05-27_02-31_dl-multimodal-oes-multistat-pool-5fold/](outputs/experiments/) — fold 4 R²=0.33 (불변), aggregate 0.572 (오히려 -0.024). 11 wafer가 4자리까지 동일 pred=0.677 → **wafer_repr 자체가 동일, pool은 병목 아님**.
- **wafer-mean auxiliary loss 구현 완료** ([src/models/bilstm_vm.py](../src/models/bilstm_vm.py), [scripts/04_train_dl.py](../scripts/04_train_dl.py)): `aux_wafer_mean: true`, `aux_wafer_loss_weight: 0.3`. wafer_repr → Linear(d, 1)로 wafer mean 직접 예측, combined loss로 encoder에 mean 보존 압력. **실행 대기 중**: [configs/exp_dl_multimodal_oes_aux_wafer_mean_5fold.yaml](../configs/exp_dl_multimodal_oes_aux_wafer_mean_5fold.yaml)

**Why fold 4 collapses (2026-05-27 확정 진단):**
- Bimodal 타깃 분포 (low 0.57-0.61 / high 0.65-0.71, within-wafer std도 다름) → 모델이 두 클러스터 binary 분류 + 클러스터 평균 출력하는 local minimum에 안정적으로 빠짐
- fold 4 val 18 wafer 중 11개 high-mode가 모두 동일 wafer_repr → 동일 출력 (~0.6770)
- 2024-07-11_01, _02 (extreme high oxide지만 July OES 시그너처) → low-mode로 오분류
- proc-only DL fold 4 R²=0.134이나 XGB proc-stats는 0.62 — **DL의 process encoder가 XGB의 long-horizon stats (mean/std/slope across 100 cycles)를 추출 못함**

**How to apply now:** 다음 agent는 fold 4 개선 시도 시 (1) optimization 변경 (lr/epoch/seed) 절대 시도 금지, (2) ensemble도 무효, (3) 다음 우선순위는 wafer-mean aux loss 실험 결과 확인 → 실패하면 cycle-aggregated stats를 head에 직접 주입 (option B) 또는 XGB feature 주입 또는 residual hybrid.

상세 실행 로그는 [outputs/experiments/2026-05-27_*/](outputs/experiments/) 폴더들.

---

## 한눈에 보기 (2026-05-26 업데이트)

- **연구 초점은 `oxide_etch`로 좁힘.** `si_etch`는 spatial pattern 만으로도 대부분 포화되어 새 DL 실험에서는 학습 시간 절약을 위해 제외한다.
- **DL 5-fold 검증 완료:** `outputs/experiments/2026-05-21_02-09_dl-multimodal-5fold/`
  - 기본 multimodal mean-pool DL oxide 결과: RMSE `0.0507±0.0133`, R² `0.5426±0.2316`
  - fold 0/1/3은 양호, fold 2 약함, **fold 4가 심하게 붕괴** (RMSE `0.0730`, R² `0.149`)
- **fold 4 이슈가 현재 최우선 진단 대상.**
  - XGBoost fold 4는 RMSE `0.0487`, R² `0.621`로 괜찮음 → fold 4가 예측 불가능한 것은 아님.
  - DL fold 4는 prediction std와 wafer-mean slope가 낮아지는 **wafer-level calibration / mean-shrinkage 실패**가 반복됨.
  - fold 4는 평균 target shift가 크지는 않지만 low/high oxide wafer 조합을 DL이 잘 구분하지 못함.
- **Method 4 (mean+late+drift pooling) 결과 병합됨:** `configs/exp_dl_late_drift_oxide_5fold.yaml`, 비교 폴더 `outputs/experiments/2026-05-25_18-00_late-drift-vs-mean-compare/`
  - fold 2/4는 약간 개선하지만 aggregate는 mean pool보다 나쁨. fold 4 붕괴는 해결 못함.
- **Method 5 (Process-conditioned OES FiLM) 구현 및 실행됨:** `configs/exp_dl_multimodal_procfilm_singlefold.yaml`, 결과 `outputs/experiments/2026-05-25_17-38_dl-multimodal-procfilm-singlefold/`
  - oxide RMSE `0.0513±0.0121`, R² `0.538±0.215`; fold 4는 여전히 RMSE `0.0731`, R² `0.147`
  - 따라서 fold 4 실패는 procfilm 모듈 고유 문제가 아니라 DL 공통 문제에 가까움.
- **학습 스크립트 수정:** `scripts/04_train_dl.py`
  - target/fold마다 RNG seed 재설정: `fold_seed = base_seed + target_idx * 1000 + fold`
  - `--folds 4` 같은 선택 fold 실행 옵션 추가
  - fold 4 seed-reset 재실험: `outputs/experiments/2026-05-25_23-27_dl-multimodal-5fold/`
    - RMSE `0.0677`, R² `0.268`, prediction std `0.050`, wafer-mean slope `0.312`
    - 기존보다 개선됐지만 다른 fold보다 여전히 낮음 → seed/optimization 문제와 fold-specific DL representation failure가 섞여 있음.
- **seed sweep 진단 스크립트 추가:** `scripts/08_diagnose_seed_sweep.py`
  - fold 4를 seed 여러 개로 반복 실행하고 `seed_sweep_summary.csv`, `worst_wafers.csv`를 저장한다.
  - 권장 명령:
    ```powershell
    .\.venv\Scripts\python.exe -m scripts.08_diagnose_seed_sweep --config configs/exp_dl_multimodal_5fold.yaml --fold 4 --seeds 42,43,44,45,46 --run
    ```
- 상세 기록은 `docs/progress_update_2026-05-26.md` 참조.

**How to apply now:** 다음 agent는 새 DL 실험을 돌릴 때 기본적으로 oxide-only config를 사용한다. 성능 개선보다 먼저 fold 4 seed sweep을 통해 "운 나쁜 seed"인지 "fold-specific representation failure"인지 확인한다. fold 4가 여러 seed에서 계속 낮으면 residual correction 또는 explicit process temporal feature injection 쪽이 다음 우선순위다.

---

## 한눈에 보기 (2026-05-08 기준)

- **Phase 1 (전처리/캐시) ✅ 완료**
- **Phase 2 (XGBoost baseline) ✅ 완료**
- **Phase 3 (Cycle-Aware DL) ✅ single-fold 완료** — 최선 모델은 multimodal(FiLM+Fourier xy, mean pool). 5-fold 확장 미완.
- **Phase 4 (해석) ◐ 일부** — XGB SHAP, DL cycle attribution 각각 fold0 만 생성.
- **중간 발표 마침** — `docs/46분반_6조_종설_중간발표_최종.pptx`. 발표용 figure 4종(`outputs/figures/pres_01~04.png`) 빌드.

**Why:** 지도교수 피드백 (2026-04-24, 4-26 발표 후) 반영 → "PCA 쓰지 말고 2D-CNN 으로 통합 인코딩, OES/Process 동일 인코더 패밀리, BiLSTM 기반". 본 모델은 그 설계 원칙대로 구현 완료.

**How to apply:** 아래 결과표·이슈를 보고 Phase 3 의 5-fold 확장과 Phase 4 의 해석 확장이 다음 작업이라는 것을 기준으로 삼는다. NOTES.md 자동 시드는 비어 있는 경우가 많아 metrics.json 과 config.yaml 을 1차 truth 로 본다.

---

## Phase 1 — 데이터 전처리 + 캐시 빌드 ✅

**핵심 데이터 사실:**
- 96 wafer NetCDF, 측정값 있는 88 wafer 사용 (Si_Oxide_etch_89_points.csv)
- 89 측정 포인트/wafer → 총 7,832 sample
- 10 Lot (날짜 기반), DAY_TO_LOT 매핑: 2024_07_02=1 ... 2024_08_22=10
- OES: 3,648 wavelength × ~14,744 time steps (uint16)
- Process: 44 raw 채널, 31 채널만 모든 wafer 공통 (13개는 9/88 wafer 만 기록 — 누락 채널 예: Gas6Flow, Heater5~8Temp, ThermoCouple1~4Temp, attenuatorRatio, moriOuterCurrent)
- Cycle 구조: SF6(4.5s ON) + C4F8(1.5s ON) = 6s/cycle × 100 cycles + ignition 1s
- Edge 수: 100/101/102 → end-anchored 100 으로 trim
- 1개 wafer (`2024-08-07_08`) alignment offset +118s — 이대로 캐시됨, 추후 점검 필요

**생성 파일 (`cache/v1/`):**
- `wafers/*.npz` — 88 wafer (총 7.2 GB)
- `measurements.csv` — 7832 × 11
- `splits/kfold5_wafer.npz` — StratifiedGroupKFold (seed=42)
- `splits/loo_lot.npz` — Leave-One-Lot-Out (10 fold)
- `features/baseline_xgb_v1.csv` — 88 wafer × 330 features (32채널 × 8 stat × OES/Proc + X,Y)

**실행 스크립트:** `scripts/01_build_cache.py --version v1`, `scripts/02_make_splits.py --version v1`

---

## Phase 2 — XGBoost Baseline ✅

**최신 실험:** `outputs/experiments/2026-04-30_15-32_baseline-xgb/` (5-fold, K=5)

| Target | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| si_etch (μm)    | 0.3285 ± 0.0872 | 0.1710 ± 0.0213 | 0.9911 ± 0.0046 | 0.39% |
| oxide_etch (μm) | 0.0514 ± 0.0043 | 0.0308 ± 0.0029 | 0.5512 ± 0.0825 | 4.77% |

**관찰:**
- si_etch fold std/mean = 26.5% — 안정성 기준(10%) 미달. fold 3,4 RMSE ~0.43 vs fold 0,1,2 ~0.25. Lot 구성 차이로 추정 (미분석).
- oxide_etch 가 어려운 타겟 — DL 개선 여지 큼 (=실제로 큰 폭 개선됨, 아래 Phase 3).
- SHAP 결과: `figures/shap_oxide_etch_fold0.png` 생성됨. 1위 피처 `proc_Heater2Temp_late` (후반 히터 온도). 그룹별: Process Temporal ≈ XY > Process Static > OES.

---

## Phase 3 — Cycle-Aware DL ✅ (single-fold 완료, 5-fold 확장 미완)

### 아키텍처 (구현 완료, [src/models/bilstm_vm.py](../src/models/bilstm_vm.py))

```
OES (B, 100, 128, 3648)         Process (B, 100, 30, 31)
        |                                |
   2D-CNN encoder              2D-CNN encoder       (X, Y) per point
   (CycleSeriesEncoder)        (shared family,           |
        |                       independent weights)     |
   (B, 100, 128)               (B, 100, 64)          Fourier(X,Y) → MLP
        |                                |             (B, n_pts, 64)
        +------- concat -------+                          |
                          ↓                              |
              cycle_fusion FC: 192 → 128                 |
                          ↓                              |
              BiLSTM (1 layer, hidden 128) → (B, 256)    |
                          ↓                              |
                      FiLM 변조 ← ← ← ← ← ← ← ← ← ← ← ←┘
                          ↓
                   Regression head
                          ↓
                  si_etch / oxide_etch
```

**주요 설계 결정:**
- **mean pool** 이 cycle aggregation 의 정답 — attention pool 은 oxide_v2 에서 시도 후 오히려 악화 (R² 0.734 → 0.643). 다음 ablation 에서도 mean 유지.
- **FiLM + Fourier xy** 가 핵심. 89 점이 같은 wafer_repr 을 공유하는 구조에서, 점별 차별화 능력을 만들어줌. FiLM 없으면 si_etch RMSE 가 1.91 까지 폭발 (`2026-04-30_23-14_dl-multimodal-singlefold` 사례).

### 실험 결과 (모두 single-fold = fold 0)

| 실험 폴더 | Modality | 풀링 | oxide RMSE / R² | si RMSE / R² | 비고 |
|---|---|---|---|---|---|
| `2026-05-01_00-56_dl-multimodal-singlefold` ★ | OES + Proc | mean | **0.0399 / 0.734** | 0.232 / 0.996 | **현재 best, 발표용 결과** |
| `2026-05-01_03-16_dl-oxide-v2-attn-singlefold` | OES + Proc | attention | 0.0463 / 0.643 | 0.239 / 0.996 | attention 실패. 롤백 |
| `2026-05-01_13-00_dl-oes-only-singlefold` | OES only | mean | 0.0626 / 0.346 | 0.290 / 0.994 | OES 단독은 약함 |
| `2026-05-04_11-18_dl-proc-only-singlefold` | Proc only | mean | 0.0464 / 0.640 | 0.264 / 0.995 | **Proc 단독이 OES 단독보다 훨씬 강함** |

### 핵심 발견 (★ 발표·논문 핵심 메시지)

1. **Multimodal DL 이 XGB 대비 oxide RMSE 22% 감소** (0.0514 → 0.0399, R² 0.55 → 0.73). 졸업논문 종료 기준 "RMSE -20%" 달성.
2. **Process channel 이 oxide 신호의 대부분을 가짐.** Proc-only R² 0.64 ≈ multimodal R² 0.73 의 87%. OES 만으로는 0.35.
3. **OES + Process 가 합쳐지면 +R² 0.094 의 추가 이득.** Cross-modal interaction(예: 특정 cycle 의 RF/pressure 변화 ↔ OES 발광선 변화) 이 진짜 신호 → DL 의 cross-modal 학습 가치를 직접 증명.
4. **Attention pool 은 oxide 에 해롭다.** "특정 cycle 가 더 중요하다" 는 가설이 데이터에 안 맞음. 100 cycle 평균이 더 잘 일반화.
5. **si_etch 는 모든 모델에서 R² ≈ 0.99 로 saturate.** spatial baseline 이 이미 R²=0.985 라서 모델 비교의 변별력이 없음. **모든 contribution 주장은 oxide 결과로 만든다.**

### 다음 할 것 (Phase 3 마무리)

1. **5-fold 확장** (지금까지 모두 single-fold). 종료 기준 2번 (fold std/mean ≤ 10%) 검증 필요.
   - 가장 먼저 `2026-05-01_00-56_dl-multimodal-singlefold` 의 config 그대로 5-fold 실행.
2. **Sequence model ablation** (연구계획서 Exp 5) — BiLSTM vs uni-LSTM vs GRU vs 1D-Temporal-CNN.
3. **Process encoder ablation** (Exp 6) — 2D-CNN vs channel-wise 1D-CNN.
4. (Optional, 종료 기준 미달 시) Phase 5 — Seed Ensemble (k=3~5).

---

## Phase 4 — 해석 ◐ 일부 진행

**완료:**
- XGBoost SHAP, oxide_etch fold 0: `outputs/experiments/2026-04-30_15-32_baseline-xgb/figures/shap_oxide_etch_fold0.png`
  - 1위: `proc_Heater2Temp_late` / 2위: `X` 좌표 / 그룹별 Process Temporal ≈ XY > Process Static > OES
- DL cycle attribution, oxide_etch fold 0: `outputs/experiments/2026-05-01_00-56_dl-multimodal-singlefold/figures/cycle_attribution_oxide_etch_fold0.png`
  - 후반 사이클 (80~100, 특히 95~100 피크) 이 importance 높음. 후반 std 리본 넓음 → wafer 별 후반 drift 다양.
- 두 결과의 결론 수렴: "공정 후반의 열적/공정 drift 가 oxide etch 의 핵심 결정 인자"
- 스크립트: `scripts/05_interpret.py` (XGB SHAP + DL gradient attribution 통합)

**미완:**
- si_etch 해석 (현재 oxide 만)
- 5-fold 결과의 SHAP/attribution 안정성
- Cycle embedding 시각화 (t-SNE/UMAP) — 연구계획서 §4.5
- DL attribution vs SHAP 정량 비교

---

## 발표/문서 (2026-05-08 시점)

**중간발표 완료:**
- `docs/46분반_6조_종설_중간발표.pptx` (초안), `..._최종.pptx` (최종본)
- `docs/Modeling_발표.pptx` (모델링 세션용)
- `docs/연구계획서_초안.md` — 연구계획서 (지도교수 피드백 반영, 분산 분해 섹션 포함)

**발표용 figure (`outputs/figures/pres_*.png`):**
- pres_01: 데이터셋 overview (wafer 수, OES/Proc shape, 대표 spectrum/trace)
- pres_02: OES cycle x wavelength heatmap (cycle 진행에 따른 발광 변화)
- pres_03: 89-point wafer map + target 분포
- pres_04: XGB & DL pred-vs-true scatter (si, oxide)
- 생성 스크립트: `scripts/07_make_presentation_figures.py`

**아키텍처 다이어그램:** `outputs/figures/arch_xgboost.png`, `arch_dl_multimodal.png` (`scripts/06_draw_architecture.py`)
