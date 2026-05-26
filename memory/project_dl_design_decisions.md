---
name: DL Architecture Design Decisions
description: Settled DL design choices (FiLM+Fourier xy, mean pool, multimodal early fusion) — what was tried, what won, and why
type: project
---

DL 모델 설계에서 ablation 으로 확정된 결정. 다른 agent 가 "이거 한번 시도해볼까" 하는 함정에 빠지지 않게 기록.

## 1. xy 인코딩: FiLM + Fourier 가 필수 (raw xy 절대 안 됨)

**Why:** 89 측정 점이 모두 같은 wafer_repr 을 공유. 점별 차별 신호는 (X, Y) 2 개 raw scalar 뿐. FiLM 없이 raw xy 만 head 로 흘리면 si_etch RMSE 가 1.91 까지 폭발 (`2026-04-30_23-14_dl-multimodal-singlefold`).

**How to apply:**
- `model.params.xy_n_freqs` ≥ 4 (실험은 6 사용)
- `model.params.use_film: true`
- 새 multimodal 실험 시 이 두 값 절대 끄지 말 것. ablation 으로 끄는 건 OK 지만 main config 에서는 필수.

## 2. Cycle aggregation: mean pool 이 정답, attention 은 oxide 에서 악화

**Why:** `2026-05-01_03-16_dl-oxide-v2-attn-singlefold` 에서 검증. mean pool oxide R²=0.734 → attention pool R²=0.643. "특정 cycle 이 더 중요하다" 는 가설이 데이터에 안 맞음. 100 cycle 의 통계적 안정성이 학습된 가중치보다 강함.

**How to apply:**
- `model.params.pool: "mean"` 유지.
- attention pool 은 ablation 비교군으로만 (논문에 "시도했고 실패" 로 보고).
- 논문 서사: "uniform mean pool 이 oxide 에 더 강한 신호" — counterintuitive 하지만 데이터로 증명됨.

## 3. Multi-modal early fusion: OES + Process 가 cross-modal 이득 +0.094 R²

**Why (2026-05-04 검증):**
- OES-only oxide R² = 0.346
- Proc-only oxide R² = 0.640
- Multimodal oxide R² = 0.734 → cross-modal 이득 +0.094

Process 가 oxide 신호의 대부분을 가지지만, OES 와 합치면 추가 이득이 명확. "그냥 process 만 쓰면 되는 거 아닌가" 반박을 방어하는 데이터.

**How to apply:**
- 본 연구의 main 모델은 multimodal 유지.
- Modality ablation 결과는 발표/논문에서 "각 modality 의 기여" 로 명시 보고.
- Process 채널은 31 (88 wafer 공통) 만 사용. 누락 13 채널 추가 시도는 wafer 손실 trade-off 검토 필요.

## 4. 인코더: 2D-CNN 동일 패밀리 (지도교수 P1, P2 원칙)

**Why:** 지도교수 피드백 (2026-04-24) — PCA 같은 선형 축약 금지, OES/Proc 동일 인코더 패밀리, 가중치는 독립.

**How to apply:**
- OES encoder: `n_blocks=4`, time stride 2, channel stride 4, kernel_chan=9 (발광선 폭 고려)
- Proc encoder: `n_blocks=3`, time stride 2, channel stride 2, kernel_chan=3
- 두 인코더 모두 [src/models/cycle_encoder.py](../src/models/cycle_encoder.py) 의 `CycleSeriesEncoder` 클래스 재사용 (=동일 패밀리).
- 변경 시 연구계획서 §4.4.2 와 일관성 유지 필수.

## 5. Sequence model: BiLSTM (Transformer 시도 안 함)

**Why:** wafer 88 개로 Transformer 는 과적합 위험 큼. 100 cycle 길이는 LSTM 에 적합. 양방향은 ignition+endpoint 양쪽이 결과에 기여하므로 채택.

**How to apply:**
- `lstm_hidden=128, lstm_layers=1` 로 충분히 작은 모델. 키울 필요 없음.
- 연구계획서 Exp 5 ablation: BiLSTM vs uni-LSTM vs GRU vs 1D-Temporal-CNN — 아직 미수행.

## 6. Single-fold → 5-fold 확장 시 주의

- 모든 현재 결과는 fold 0 single-fold. 5-fold 평균이 single-fold 보다 나빠질 가능성 있음 (XGB 의 si fold 3,4 처럼).
- DL 5-fold 1회 ≈ 4시간 × 2 target × 5 fold = 40시간 — GPU 시간 큼. 가장 중요한 실험 (multimodal best) 부터.
- 5-fold 확장 시 종료 기준 2번 (fold std/mean ≤ 10%) 검증 필수. 미달 시 Phase 5 (seed ensemble) 발동.

## 7. OES wavelength selection: top-k=256, stat=late_mean 가 정답 (2026-05-27 확정)

**Why:**
- 원본 3648 채널 그대로 학습 시 fold 4 R²=0.149. wavelength reduction 후 0.376까지 회복.
- top_k=256과 128 결과 동일 (~0.376) → 더 줄여도 무의미.
- `stat=drift` (late_mean - early_mean) 는 late_mean 보다 모든 fold 악화 — 88 wafer × ~70 train 에서 차분은 노이즈 증폭.

**How to apply:**
- 새 multimodal 실험은 `data.oes_band_selection: {method: correlation, top_k: 256, stat: late_mean, late_start_cycle: 80}` 로 시작.
- top_k=128 도 backup으로 사용 가능 (성능 거의 동일, 메모리 절감).
- drift 는 ablation 비교군으로만 (논문에 "시도했고 실패" 보고).

## 8. fold 4 collapse: optimization 변경으로는 해결 불가 (2026-05-27 확정)

**Why:**
- 5-seed sweep on fold 4 (config: topk256 corr): R²=0.30~0.41, std=0.046.
- 5 seed의 잔차 상관 평균 off-diagonal = **0.94**. 다른 init이 같은 wafer에서 같은 방향으로 틀림.
- 5-seed simple mean ensemble R²=0.39 (best single 0.41보다 *나쁨*).
- lr 1.5e-3 → 5e-4, epochs 40 → 80, patience 8 → 15 다 시도 — fold 4 그대로 0.32.
- **결론**: optimization-induced local minima가 아니라 **encoder representational ceiling**.

**How to apply:**
- 다음 agent는 fold 4 개선 시도 시 **다음 방법 절대 시도 금지**: (1) lr/epoch/scheduler 변경, (2) seed 변경, (3) seed ensemble, (4) 더 강한 정규화 (dropout/wd).
- 본질적 후보: (A) wafer-mean aux loss, (B) cycle-aggregated stats를 head에 직접 주입, (C) residual hybrid (XGB base + DL residual), (D) XGB feature 주입.
- 우선순위는 진단을 직접 검증할 수 있는 A 부터.

## 9. Pool 변경은 fold 4를 못 고친다 (2026-05-27 확정)

**Why:**
- multi-stat pool ([mean, std, max, slope] concat + projection): fold 4 R²=0.33 → 0.33 (불변). aggregate 오히려 -0.024.
- 11개 high-mode wafer가 baseline과 multi-stat 모두에서 동일 출력 (~0.677) — wafer_repr 자체가 동일.
- mean_late_drift pool: fold 4 0.149 → 0.147 (불변). attention pool: 더 나쁨.
- **결론**: pool 변경으로는 encoder가 같은 입력에 같은 출력을 내는 collapse를 못 깬다.

**How to apply:**
- `pool: "mean"` 유지. 변경 시 fold 0/1/3 회귀 위험만 있고 fold 4 이득 없음.
- 본질적 해결책은 pool 상류 (encoder, loss, 또는 feature) 에서 찾아야 함.
