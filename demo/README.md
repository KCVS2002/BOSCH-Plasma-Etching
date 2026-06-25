# 경진대회 Live 시연 — Virtual Metrology Dashboard

플라즈마 식각 가상계측 모델을 발표장에서 **실시간으로 시연**하기 위한 Streamlit 대시보드.

테스트(held-out) 웨이퍼를 고르면 → 그 웨이퍼의 OES + Process 센서 신호를 **딥러닝 모델에
직접 통과시켜** 89개 측정 지점의 식각량을 예측하고, 예측 / 실측 / 오차 **웨이퍼 맵**과
XGBoost · Spatial-mean baseline 대비 정확도를 같은 화면에서 보여준다.

**held-out 정직성:** 5-fold wafer GroupKFold 라서 모든 웨이퍼를 시연하되, 각 웨이퍼는
**자기가 검증셋이었던 fold 의 체크포인트**로 추론한다(`oxide_etch_fold{0..4}.pt` 전부 번들에 포함).
따라서 fold 0뿐 아니라 **전 fold 웨이퍼**가 학습에 안 쓰인 진짜 held-out 예측이다.
`si_etch` 는 단일-fold 모델이라 **fold-0 웨이퍼만** 시연 가능하며, 사이드바 토글로 켠다(기본 off, oxide 전용).

```
센서(OES + Process)  ──►  Cycle-Aware DL 모델  ──►  89-point 웨이퍼 식각 맵
                                                     └► XGBoost / Spatial baseline 비교
```

---

## 빠른 시작 (2단계)

```bat
:: 1) 시연용 번들 생성 (1회). 전 fold held-out 웨이퍼(~90개)를 압축 패키징.
::    각 웨이퍼는 자기 fold 체크포인트로 추론되어 모두 정직한 held-out 예측.
.venv\Scripts\python.exe -m demo.build_bundle

:: 2) 대시보드 실행
.venv\Scripts\python.exe -m streamlit run demo/app.py --server.fileWatcherType none
```

> fold 0만 빠르게 만들려면(예전 동작): `.venv\Scripts\python.exe -m demo.build_bundle --folds 0`

브라우저가 자동으로 열린다(기본 http://localhost:8501). 발표 시 전체화면 권장.

---

## 무대에서의 흐름

1. 사이드바에서 **테스트 웨이퍼**를 고른다 (심사위원이 직접 고르게 해도 됨 — cherry-pick 아님을 보여줌).
2. **타겟 = oxide_etch** (본 연구 핵심) 선택. "모델이 보는 입력" OES 스펙트럼이 먼저 뜬다.
3. **🚀 모델 추론 실행** 버튼 → 센서 신호가 모델을 통과(수십 ms)하며 예측 맵이 드러난다.
4. 예측 / 실측 / 오차 맵 + 이 웨이퍼 R²·RMSE, 그리고 **DL vs XGBoost vs Spatial** 비교 막대.
5. **타겟 = si_etch**로 전환 → R²≈0.98 (거의 완벽). "이건 공간 패턴이 지배적이라 쉬움 →
   그래서 진짜 도전은 oxide" 라는 본 연구의 논리를 시각적으로 전달.

> 상단 KPI(전체 검증 R²: DL 0.75 vs XGB 0.56 vs Spatial 0.16)가 논문 헤드라인 수치다.
> 개별 웨이퍼 R²는 난이도에 따라 출렁이며, 이를 숨기지 않는 것이 학술적 신뢰를 준다.

---

## 동작 방식 / 구성

| 파일 | 역할 |
|---|---|
| [src/demo/inference.py](../src/demo/inference.py) | `DLPredictor` — 체크포인트 로드 → 단일 웨이퍼 forward → μm 역정규화 (라이브러리) |
| [demo/build_bundle.py](build_bundle.py) | `demo/bundle/` 생성: 모델 입력 텐서 + 실측 + 3개 모델 예측 + 메타 |
| [demo/app.py](app.py) | Streamlit 대시보드 |
| `demo/bundle/` | 산출물 (gitignored, 재생성 가능). ~214 MB |

- **oxide = 라이브 추론.** 앱이 `demo/bundle/checkpoints/oxide_etch_fold0.pt` 를 로드해
  무대에서 실제 `model.forward` 를 돌린다. 입력 텐서는 정규화까지만 미리 계산해 번들에 담겨 있어
  `Dataset/` 나 `cache/` 없이 **어떤 노트북에서도** 돈다 (GPU 있으면 ~16 ms, CPU도 1초 내).
- **si = 사전계산.** si 모델은 전체 3648밴드 OES(웨이퍼당 ~187 MB)가 필요해 라이브용으로는 너무 커서,
  예측값을 번들에 구워 넣어 표시만 한다 (값 자체는 모델의 진짜 출력).
- 두 모델 모두 **동일 split(`kfold5_wafer.npz`) fold 0** → 번들의 모든 웨이퍼가 si·oxide 양쪽 모두
  진짜 held-out 검증 웨이퍼.

### 모델 출처 (memory/reference_experiments.md)
- oxide: `2026-05-28_04-19_…aux-mixup-ema-longrun-5fold` (best, fold-0 R²=0.75)
- si: `2026-05-01_00-56_dl-multimodal-singlefold` (fold-0)
- XGBoost baseline: `2026-04-30_15-32_baseline-xgb`
- Spatial-mean: 학습 웨이퍼의 (X,Y) 위치별 평균 lookup

---

## 옵션

```bat
:: 특정 웨이퍼만 번들에 (가볍게)
.venv\python.exe -m demo.build_bundle --wafers 2024-07-02_06,2024-08-22_02

:: 포트 지정 실행
.venv\python.exe -m streamlit run demo/app.py --server.port 8501
```

추천 시연 웨이퍼:
- `2024-07-02_06` — oxide DL R²=0.99 (DL이 baseline 압도, "wow")
- `2024-08-22_02` — oxide DL R²=0.28 < XGB 0.63 (정직한 한계 사례, 질문 대비)

---

## 발표장 대비 (안정성)

- 번들만 있으면 오프라인 동작 (네트워크 불필요). USB로 복사해도 됨.
- 라이브가 꼬일 때를 대비해 **화면 녹화본**을 미리 1개 준비해 둘 것.
- 첫 추론은 CUDA 워밍업이 포함되므로 앱이 내부적으로 1회 warm-up 후 시간을 측정한다.
