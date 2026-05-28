---
name: BOSCH Plasma Etching Project Progress
description: 2026-05-28 기준. aux+mixup+EMA+120ep no-early-stop 5-fold 완료 — new best agg R²=0.666. 다음 우선순위는 fold 2 overfitting + LOO-Lot 재검증.
metadata:
  type: project
---

## 한눈에 보기 (2026-05-28 최종 업데이트)

**aux + mixup + EMA + 120ep no-early-stop → new best agg R²=0.666.** fold 4 0.59 (XGB 0.62에 근접), fold 0/1/3 모두 ≥ 0.72. **fold 2가 새 bottleneck (0.49, overfitting)** — distribution-outlier 문제로 mixup/aux도 못 풀음.

### 현재 best 모델 (5-fold wafer CV, oxide_etch)

| 실험 | f0 | f1 | f2 | f3 | f4 | agg R²±std | RMSE±std |
|---|---|---|---|---|---|---|---|
| XGB baseline | 0.56 | 0.49 | 0.43 | 0.65 | 0.62 | 0.551±0.082 | 0.0514±0.004 |
| DL baseline longrun | 0.71 | 0.70 | 0.50 | 0.75 | 0.32 | 0.596±0.163 | 0.0482±0.010 |
| DL aux-loss | 0.75 | 0.71 | 0.52 | 0.73 | 0.40 | 0.621±0.138 | 0.0468±0.009 |
| DL aux+mixup | 0.73 | 0.69 | 0.52 | 0.73 | 0.56 | 0.644±0.087 | 0.0458±0.006 |
| **DL aux+mixup+EMA 120ep ★** | **0.75** | **0.72** | **0.49** | **0.78** | **0.59** | **0.666±0.110** | **0.0440±0.007** |

**Best 폴더**: `outputs/experiments/2026-05-28_04-19_dl-multimodal-oes-aux-mixup-ema-longrun-5fold/`
**Best config**: `configs/exp_dl_multimodal_oes_aux_mixup_ema_longrun_5fold.yaml`

### 2026-05-28 실험 타임라인 (개선 chain)

1. **aux+mixup 80ep** (`03-16`): mixup이 fold 4 mode collapse를 깨뜨려 R² 0.40→0.56. agg std 0.138→0.087로 안정성 큰 폭 개선.
2. **aux+mixup+EMA 120ep no-early-stop** (`04-19`): EMA가 late-stage 진동 흡수 + 120ep로 fold 4 추가 학습 여유. fold 4 0.56→0.59, fold 3 0.725→0.781, agg 0.644→0.666.

### 핵심 기술 stack (현 best 모델 구성)

1. **OES wavelength selection**: per-fold train-only correlation top-k=256, stat=late_mean
2. **Multimodal early fusion**: OES + Process 2D-CNN → BiLSTM → mean-pool → wafer_repr (256-dim)
3. **FiLM + Fourier(X,Y, n_freqs=6)** head — 89 point 차별화 필수
4. **Wafer-mean auxiliary loss** (λ=0.3): wafer_repr → Linear(d, 1)로 wafer mean 직접 예측, mode collapse 압력 차단
5. **Wafer-level mixup** (Beta(0.2, 0.2), prob=1.0): bimodal collapse attractor 파괴 + 데이터 augmentation
6. **EMA weight averaging** (decay=0.999, ramp-up): val/inference shadow weights — mixup noise 흡수
7. **120 epochs, no early stopping**: best_state는 매 epoch 추적, EMA val_rmse 기준

### 진단 지표 변화 (fold 4)

- wafer_mean_corr: 0.63 (aux) → 0.82 (mixup) → **0.84** (EMA)
- std_ratio: 0.66 → 0.65 → 0.67
- 11개 high-mode wafer "동일 0.677" collapse 패턴 → 0.67-0.69 사이로 spread
- 잔존 문제: 2024-07-11_01, _02 (extreme high oxide, July OES) bias -0.03 (이전 -0.07)

### 현재 약점 (다음에 풀어야 할 것)

#### 1. fold 2 overfitting (새 bottleneck)
- best_ep=39 (120 중) — 일찍 best 찍고 점진 하락 (val_r2 0.49 → 0.38)
- **train_rmse 0.075→0.030**, val_rmse plateau at ~0.055 → 전형적 overfitting
- 핵심 원인: **2024-08-22_04 (y_true=0.6165, bimodal 사이 unique mid-range)** — train에 비슷한 wafer 없음, distribution outlier
- mixup/aux/EMA 모두 못 풀음 → fold 2엔 별도 접근 필요

#### 2. LOO-Lot 미검증 (이전 aux-loss 모델 R²=0.323이었음)
- 현 best 모델 (aux+mixup+EMA)에서 LOO-Lot 재실행 안 됨
- mixup이 lot-invariance에 효과 있는지 핵심 검증 필요
- 졸업논문 lot-robustness 주장의 근거가 될 핵심 실험

### 다음 우선순위 (이어서 작업할 agent를 위해)

#### A. LOO-Lot 재검증 (최우선, 1시간)
**현 best 모델 (aux+mixup+EMA)을 LOO-Lot으로 평가.** 이전 aux-only는 R²=0.323이었음. mixup이 lot-invariance에 효과 있다면 0.40+로 올라야 함.
- config 복제: `configs/exp_dl_lot_validation.yaml` 기반으로 mixup + EMA 옵션 추가
- 또는 best config의 split만 `splits/loo_lot.npz`로 바꿔서 실행

#### B. fold 2 overfitting 대응
fold 2 best_ep=39이고 이후 하락 → **fold 2만 별도로** 더 강한 regularization 필요. 후보:
- **Dropout 상향**: head_dropout 0.2 → 0.3, encoder dropout 0.1 → 0.2 (모든 fold에 적용. 단점: fold 0/1/3 성능 회귀 위험)
- **Weight decay 상향**: 1e-4 → 5e-4 또는 1e-3
- **mixup α 상향**: 0.2 → 0.4 (mixup이 fold 2에 효과가 있다면 강도 ↑로 추가 이득)
- **Cycle dropout (random cycle masking)**: 100 cycle 중 일부 마스킹 → 더 robust한 표현
- **2024-08-22_04 단독 분석**: 이 wafer가 정말 outlier인지, OES/proc 특성 분석. fold 2 train에 비슷한 wafer 1개라도 있는지 확인

#### C. 졸업논문 결과 정리
- 5-fold + LOO-Lot ablation table 완성 (baseline → aux → mixup → EMA → 120ep 누적 효과)
- agg std/mean = 16.5% (현재) → 10% 목표 미달. fold 2가 발목.
- 해석 분석 (SHAP, attribution)을 5개 fold 모두로 확장

### 확정된 설계 결정 (반복 시도 금지)

- `pool=attention`, `pool=multi_stat`, `pool=mean_late_drift` 모두 fold 4에 효과 없음 → **mean pool 유지**
- `use_film=true`, `xy_n_freqs=6` **필수**
- OES wavelength selection: **top_k=256, stat=late_mean** (drift 실패, top_k=128 동등)
- **per-wafer norm + InstanceNorm**: fold 4 악화 — 절대 시도 금지
- **seed ensemble**: residual corr=0.94로 무효
- **Optimization 변경 (lr/scheduler/epoch만)**: fold 4 collapse 해결 불가
- **EMA + mixup + aux-loss + 120ep no-early-stop**: 현 best 조합. 분리해서 ablation 가능하나 main config로 유지

---

## 한눈에 보기 (2026-05-27 최종 업데이트)

**aux-loss → pwnorm+instnorm → LOO-Lot 검증까지 완료. 모델 정확도는 개선되었으나 lot-level 일반화가 약점으로 확인됨.**

### 현재 best 모델 (5-fold wafer CV)

| 모델 | oxide R² | oxide RMSE | 비고 |
|---|---|---|---|
| XGB baseline (5-fold) | 0.551±0.082 | 0.0514±0.004 | 기준선 |
| DL aux-loss ★ | **0.621±0.138** | **0.0468±0.009** | best aggregate. fold 4 R²=0.397 |
| DL pwnorm+instnorm | 0.561±0.209 | 0.0499±0.012 | fold 4 R²=0.202. pwnorm은 fold 0/1/3 유지하나 fold 4 악화 |

**Best = aux-loss 모델** (`2026-05-27_04-16_dl-multimodal-oes-aux-wafer-mean-5fold`).

### LOO-Lot 검증 결과 (aux-loss 모델, oxide_etch)

- **aggregate R² = 0.323±0.294** (5-fold wafer CV의 0.621 대비 큰 하락)
- best lots: Lot 1 (R²=0.81), Lot 2 (R²=0.68)
- worst lots: Lot 4 (R²=-0.19, best_ep=0), Lot 6 (R²=-0.01, best_ep=0) — 학습 자체가 안 됨
- **결론: 새로운 lot에 대한 일반화 능력 부족.** 모델이 lot-specific 패턴에 의존.
- 실험 폴더: `outputs/experiments/2026-05-27_16-06_dl-lot-validation-oxide-aux/`
- config: `configs/exp_dl_lot_validation.yaml` (split: `splits/loo_lot.npz`)

### 2026-05-27 실험 타임라인

1. **aux-loss 5-fold** (`04-16`): 구현 의도대로 fold 4 R² 0.32→0.40 개선. aggregate 0.596→0.621.
2. **pwnorm+instnorm 5-fold** (`19-02`): per-wafer norm + InstanceNorm2d 시도. fold 4 오히려 악화 (0.40→0.20). aggregate 0.561.
3. **XGB baseline 재실행** (`14-00`): 비교용.
4. **LOO-Lot 검증** (`16-06`): aux-loss 모델의 lot-level 일반화 테스트. R²=0.323 — 취약점 확인.

### 확정된 설계 결정 (반복 시도 금지)

아래 결정들은 ablation으로 검증 완료. 다른 agent가 다시 시도하지 말 것:
- `pool=attention` 은 oxide에서 악화 → **mean pool 유지**
- `use_film=true`, `xy_n_freqs=6` **필수** (없으면 si RMSE 폭발)
- OES-only, Proc-only ablation 완료. **multimodal이 둘보다 +0.094 R² 우월**
- **OES wavelength selection**: top_k=256, stat=late_mean 확정. drift는 실패.
- **fold 4 collapse**: optimization(lr/seed/epoch) 변경으로는 불가. seed ensemble도 무효. structural encoder 문제.
- **multi-stat pool**: fold 4에 효과 없음. wafer_repr 자체가 동일한 것이 원인.
- **per-wafer norm + InstanceNorm**: fold 4 악화 (0.40→0.20). absolute offset 제거가 오히려 정보 손실.
- **aux-loss**: fold 4를 0.32→0.40으로 개선한 유일한 방법. 유지.

---

### 다음에 시도할 것 (우선순위순)

#### A. Lot-level 강건성 개선 (최우선)

LOO-Lot R²=0.323은 실용 수준 이하. 새 lot에 대한 일반화가 핵심 과제.

1. **Lot-aware augmentation / domain adaptation**
   - 학습 시 lot을 domain으로 취급하고, domain-invariant representation 학습 (gradient reversal layer 등)
   - 또는 lot-level batch balancing: 각 batch에 다양한 lot의 wafer가 포함되도록 샘플링

2. **Process temporal statistics를 head에 직접 주입** (설계 결정 §8의 option B)
   - XGB가 잘 잡는 long-horizon stats (cycle-mean/std/slope across 100 cycles)를 DL head에 auxiliary feature로 주입
   - LOO-Lot에서도 이런 통계가 lot-invariant signal을 제공할 가능성
   - 구현: `WaferDataset`에서 cycle-aggregated stats 계산 → head에 concat

3. **Residual hybrid** (option C)
   - XGB 예측을 base로, DL이 잔차만 학습
   - XGB가 lot-level mean을 잘 잡으므로 (fold 4도 R²=0.62), DL은 within-lot 패턴에 집중

4. **XGB feature injection** (option D)
   - XGB의 상위 feature들을 DL head에 직접 주입
   - 가장 단순하지만 "DL의 end-to-end 학습" 논문 서사가 약해짐

#### B. 논문 작성 관련

5. **5-fold 해석 분석 확장** — 현재 fold 0만. 5개 fold의 attribution 안정성 검증 필요.
6. **Sequence model ablation** (연구계획서 Exp 5) — BiLSTM vs GRU vs 1D-CNN. 아직 미수행.
7. **최종 결과표 정리** — XGB vs DL (5-fold) vs ablation 결과 종합. LOO-Lot 결과도 포함.

#### C. 실험 인프라

8. **`scripts/09_lot_validation.py`** 생성 완료 — LOO-Lot 실험 결과의 사후 분석 (lot별 bar chart, scatter, drift, heatmap). 다음 LOO 실험 후에도 재사용 가능.
   ```
   .venv\python.exe -m scripts.09_lot_validation --exp-dir <실험폴더>
   ```
   (주의: 현재 이 스크립트는 기존 sample_predictions.csv 사후 분석용. LOO-Lot에서도 동작하지만, fold=lot 매핑 활용은 추가 필요.)

---

## 이전 기록 (접어두기)

### 2026-05-27 이전 (fold 4 진단)

- **fold 4는 데이터적으로 어렵지 않음** (XGB R²=0.62, 동일 split). DL 고유의 표현력 한계 문제.
- **fold 4 seed sweep (5 seeds):** R² = 0.30~0.41 좁은 구간, residual 상관 0.94 → structural
- **OES wavelength selection 인프라** 구현됨. per-fold train-only correlation 기반 top-k.
- **multi-stat pool 시도 → 실패**: fold 4 불변. 11 wafer pred 동일 → wafer_repr collapse.
- **wafer-mean aux loss 구현 → 성공**: fold 4 R² 0.32→0.40.

### fold 4 collapse 확정 진단 (2026-05-27)

- Bimodal 타깃 분포에서 encoder가 high-mode wafer를 동일 repr로 collapse
- proc-only DL fold 4 R²=0.134이나 XGB proc-stats는 0.62 — DL process encoder가 long-horizon stats 추출 못함
- 해결 후보 중 aux-loss만 부분 성공, 나머지(pool/norm/optimization) 모두 실패

### 2026-05-26 이전

- 연구 초점을 oxide_etch로 좁힘. si_etch는 R²≈0.99로 포화.
- DL 5-fold 첫 확장 (`2026-05-21_02-09`): fold 4 collapse 최초 발견 (R²=0.149).
- late-drift pool, procfilm 등 시도 → 모두 fold 4 미해결.

### Phase 1–4 기존 기록

- **Phase 1 (전처리/캐시) ✅** — 88 wafer, 7832 sample, cache/v1/
- **Phase 2 (XGBoost baseline) ✅** — oxide R²=0.551, si R²=0.991
- **Phase 3 (Cycle-Aware DL) ✅** — single-fold best: oxide R²=0.734, 5-fold best: R²=0.621
- **Phase 4 (해석) ◐** — XGB SHAP + DL attribution fold 0만 완료
- **중간발표 완료** — `docs/46분반_6조_종설_중간발표_최종.pptx`
