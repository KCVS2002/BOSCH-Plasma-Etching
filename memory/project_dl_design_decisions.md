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

## 11. LOO-Lot 검증: lot-level 일반화가 현재 최대 약점 (2026-05-27 확인)

**Why:** Leave-One-Lot-Out 10-fold CV 실행 (aux-loss 모델).
- aggregate R²: **0.323±0.294** (5-fold wafer CV의 0.621 대비 절반 이하)
- Lot 4 R²=-0.19 (best_ep=0), Lot 6 R²=-0.01 (best_ep=0): 학습 자체 안 됨
- Lot 1 (R²=0.81), Lot 2 (R²=0.68): 초기 lot만 양호
- 모델이 lot-specific 패턴에 의존. 새 lot에 대한 일반화 부족.
**How to apply:** 다음 개선 시도 시 LOO-Lot R²를 반드시 함께 검증. 5-fold R²만 보면 과대평가 위험.
**다음 시도 후보:**
- (A) process temporal stats를 head에 직접 주입 — XGB가 잘 잡는 lot-invariant signal
- (B) residual hybrid (XGB base + DL residual) — XGB가 lot mean을 잡으므로
- (C) domain adaptation / lot-aware training — gradient reversal 등
- (D) XGB feature injection — 가장 단순하지만 end-to-end 서사 약화
