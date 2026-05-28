---
name: DL Architecture Design Decisions
description: 확정된 DL 설계 결정 (FiLM, mean pool, multimodal, topk256, aux-loss). pwnorm/instnorm 실패 추가. 2026-05-27 기준.
metadata:
  type: project
---

DL 모델 설계에서 ablation으로 확정된 결정. 다른 agent가 "이거 한번 시도해볼까" 하는 함정에 빠지지 않게 기록.

## 1. xy 인코딩: FiLM + Fourier 필수

**Why:** 89 측정 점이 모두 같은 wafer_repr 공유. FiLM 없이 raw xy만 head로 흘리면 si_etch RMSE 1.91 폭발.
**How to apply:** `xy_n_freqs≥4` (실험은 6), `use_film: true` 절대 끄지 말 것.

## 2. Cycle aggregation: mean pool 정답

**Why:** attention pool oxide R²=0.643 (mean 0.734 대비 -0.09). multi-stat pool ([mean,std,max,slope])도 fold 4 불변.
**How to apply:** `pool: "mean"` 유지. 변경 시 fold 0/1/3 회귀 위험만 있음.

## 3. Multi-modal early fusion: cross-modal +0.094 R²

**Why:** OES-only 0.346, Proc-only 0.640, Multimodal 0.734.
**How to apply:** main 모델은 multimodal 유지.

## 4. 인코더: 2D-CNN 동일 패밀리

**Why:** 지도교수 피드백 — PCA 금지, OES/Proc 동일 인코더 패밀리, 가중치 독립.
**How to apply:** OES `n_blocks=4, kernel_chan=9`, Proc `n_blocks=3, kernel_chan=3`. 둘 다 `CycleSeriesEncoder`.

## 5. Sequence model: BiLSTM

**Why:** 88 wafer로 Transformer 과적합 위험. `lstm_hidden=128, lstm_layers=1`.
**미완:** 연구계획서 Exp 5 ablation (BiLSTM vs GRU vs 1D-CNN) 아직 미수행.

## 6. OES wavelength selection: top_k=256, stat=late_mean 확정

**Why:** 원본 3648 채널 → 256으로 축소. fold 4 R²: 0.149→0.376. drift stat은 모든 fold 악화.
**How to apply:** `oes_band_selection: {method: correlation, top_k: 256, stat: late_mean, late_start_cycle: 80}`.

## 7. fold 4 collapse: optimization 변경 불가 확정

**Why:** 5-seed sweep R²=0.30~0.41, residual corr=0.94. ensemble R²=0.39. lr/epoch/scheduler 모두 무효.
**금지 목록:** (1) lr/epoch/scheduler 변경, (2) seed 변경, (3) seed ensemble, (4) dropout/wd 강화.

## 8. Pool 변경은 fold 4를 못 고침 확정

**Why:** multi-stat pool fold 4 R²=0.33→0.33. late-drift pool 0.147. attention 더 나쁨. wafer_repr 자체가 동일.

## 9. aux-loss: fold 4 개선한 유일한 방법 (2026-05-27 확정)

**Why:** wafer_repr → Linear(d,1)로 wafer mean 직접 예측. combined_loss = point_mse + 0.3 × aux_mse.
- fold 4 R²: 0.32→0.40 (+0.08)
- aggregate R²: 0.596→0.621 (+0.025)
- fold 0/1/3 회귀 없음 (오히려 fold 1 개선)
**How to apply:** `aux_wafer_mean: true`, `aux_wafer_loss_weight: 0.3` 유지. 모든 새 실험에 포함.

## 10. per-wafer norm + InstanceNorm: 실패 (2026-05-27 확정)

**Why:** fold 4 입력-출력 상관 방향이 다른 fold와 반대인 것을 발견 → global normalizer의 absolute offset이 원인으로 추정 → per-wafer re-center + InstanceNorm2d 시도.
- 결과: fold 4 R²: 0.40→0.20 (**악화**). aggregate 0.621→0.561.
- absolute offset을 제거하면 lot-level mean 정보까지 손실 — 오히려 정보 감소.
**How to apply:** `per_wafer_norm: true` + `norm_type: "instance"` 조합은 **사용 금지**.

## 11. LOO-Lot 검증: lot-level 일반화가 현재 최대 약점 (2026-05-27 확인, 재검증 대기)

**Why (2026-05-27):** Leave-One-Lot-Out 10-fold CV 실행 (aux-loss 모델).
- aggregate R²: **0.323±0.294** (5-fold wafer CV의 0.621 대비 절반 이하)
- Lot 4 R²=-0.19 (best_ep=0), Lot 6 R²=-0.01 (best_ep=0): 학습 자체 안 됨
- Lot 1 (R²=0.81), Lot 2 (R²=0.68): 초기 lot만 양호
- 모델이 lot-specific 패턴에 의존. 새 lot에 대한 일반화 부족.

**How to apply:** 다음 개선 시도 시 LOO-Lot R²를 반드시 함께 검증. 5-fold R²만 보면 과대평가 위험.

**대기 중인 재검증:** 현 best 모델 (aux+mixup+EMA 120ep)으로 LOO-Lot 재실행 필수. mixup이 cross-lot interpolation을 만드므로 lot-invariance 효과 기대. config: `configs/exp_dl_lot_validation.yaml` 베이스로 mixup/EMA/aux 옵션 추가하여 실행.

## 12. Wafer-level mixup: fold 4 mode collapse를 푸는 정공법 (2026-05-28 확정)

**Why:** aux-loss 단독으로 fold 4 R² 0.32→0.40까지만 됨. 11개 high-mode wafer가 동일 wafer_repr → 동일 pred 0.677 문제 잔존. wafer-level mixup으로 cross-wafer 선형 보간 도입 후:
- fold 4 R²: 0.40 → **0.56** (+0.16). 동일 0.677 패턴 깨짐.
- aggregate R²: 0.621 → 0.644.
- **aggregate std: 0.138 → 0.087 (-37%)** — fold 간 안정성 큰 폭 개선.

**구현 ([scripts/04_train_dl.py](../scripts/04_train_dl.py) `_apply_mixup`):**
- 배치 내 모든 텐서(oes/proc/xy/target/xgb_feat)를 **단일 λ ~ Beta(α, α), 단일 permutation**으로 선형 보간
- α=0.2 (U-shaped lite mixup), prob=1.0
- aux-loss와 자동 호환: `mean(λ·t_A + (1-λ)·t_B) = λ·mean(t_A) + (1-λ)·mean(t_B)` (선형성)
- val 단계 절대 적용 안 함

**How to apply:**
- 새 multimodal 실험 기본 `training.mixup: {enabled: true, alpha: 0.2, prob: 1.0}`
- α=0.4/1.0 sweep은 미수행 — α=0.2가 fold 0/1/3 회귀 작고 fold 4 개선 큼

**작동 안 한 fold**: fold 2. 별도 진단 §14 참조.

## 13. EMA (weight averaging) + 120ep no-early-stop: late-stage 진동 흡수 (2026-05-28 확정)

**Why:** aux+mixup 학습 곡선이 후반부에 진동하며 상승 (mixup gradient noise + bimodal landscape). 단순 cosine annealing만으로는 부족 — `WeightEMA(decay=0.999, ramp-up)` shadow 모델 도입 후:
- fold 4 R²: 0.556 → **0.588**. 120 epoch까지 학습 계속 (best_ep=116).
- fold 3 R²: 0.725 → **0.781**. fold 0/1도 회복 (aux+mixup에서 -0.02 회귀했던 부분).
- aggregate R²: 0.644 → **0.666**. RMSE 0.0458 → 0.0440.
- 마지막 30 epoch val_r2 std: fold 1=0.002 (거의 평탄), fold 4=0.027 (이전 큰 swing).

**EMA 구현 ([scripts/04_train_dl.py](../scripts/04_train_dl.py) `WeightEMA`):**
- `deepcopy(model)` 후 매 `optim.step()` 직후 `ema_p ← d_eff·ema_p + (1-d_eff)·model_p`
- `d_eff = min(0.999, (1+step)/(10+step))` — 초기 ramp-up으로 init bias 회피
- **deepcopy 후 즉시 `flatten_parameters()` 호출 필수** (LSTM 경고 + 성능 손실 방지). `load_state_dict()` 후에도 동일.
- val/inference는 shadow model로, best_state도 shadow의 state_dict로 저장 → checkpoint에 EMA weights 그대로 저장됨.

**early stop 끄기:**
- `early_stop_patience: 0` → `early_stop_active=False` → break 안 됨, best_state는 매 epoch 추적
- 120ep × 5 fold × ~5min ≈ 4시간 (overnight). 잘 학습되는 fold도 stop 안 시키고 끝까지 학습.

**How to apply:**
- 새 multimodal 실험 기본 `training.ema: {enabled: true, decay: 0.999}` + `training.early_stop_patience: 0` + epochs 120 이상.
- early stop 끄기에 대한 안전망: `best_state`가 매 epoch 갱신되므로 overfitting 후 떨어져도 best는 보존됨.

## 14. fold 2 overfitting: distribution outlier 문제 (2026-05-28 신규 진단)

**Why:** aux+mixup+EMA 120ep에서 fold 2만 0.49로 하락 (다른 fold ≥0.72). best_ep=39이고 이후 plateau→하락. train_rmse 0.075→0.030, val plateau at 0.055 → **전형적 overfitting**, but mixup/aux/EMA 모두 안 풀음.

**진단:**
- fold 2 val에 **2024-08-22_04 (y_true=0.6165, bimodal 사이 unique mid-range)** 포함
- 전체 88 wafer 중 0.61~0.65 사이는 이 wafer 거의 단독 → fold 2 train에 비슷 wafer 없음
- mixup이 mid-range 합성을 만들어도 그게 **특정 OES/proc 시그너처와 연결되지 않음** → 우연 합성으로 학습 안 됨
- distribution outlier 문제. fold 4와 다른 종류의 어려움.

**How to apply:** fold 2를 끌어올리려면 mixup/aux/EMA로는 부족. 별도 접근 필요:
- Dropout/weight_decay 상향 (모든 fold에 영향)
- Cycle dropout (random cycle masking)
- 또는 2024-08-22_04를 outlier로 보고 별도 분석/제외 검토 (졸업논문엔 한계로 보고)

## 15. EMA + flatten_parameters: LSTM weight 메모리 정렬 (2026-05-28 확정)

**Why:** `copy.deepcopy(model)`이 LSTM weight를 비연속 메모리로 분리시킴 → 매 forward에서 cuDNN이 weight를 재압축 (`UserWarning: RNN module weights are not part of single contiguous chunk of memory`). 결과는 정확하지만 val/inference 단계 느려지고 peak GPU 메모리 ↑.

**How to apply:**
- `WeightEMA.__init__`의 deepcopy 직후 `for m in self.ema_model.modules(): if isinstance(m, nn.RNNBase): m.flatten_parameters()` 호출
- `model.load_state_dict(best_state)` 직후에도 동일
- 두 위치 모두 이미 적용됨 ([scripts/04_train_dl.py](../scripts/04_train_dl.py))
- in-place `mul_/add_` (EMA update)는 layout 유지하므로 단일 호출로 충분
