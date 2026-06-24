# RunPod DL Cache Workflow

이 문서는 클라우드 GPU 실행 전에 CPU-heavy 전처리를 미리 끝내는 절차를 정리한다.

## 1. 캐시 만들기

현재 config에 맞춰 RunPod 업로드용 캐시를 만든다. 업로드 비용과 시간을 줄이려면 `normalizers` 레벨을 기본으로 쓴다.

```powershell
.venv\Scripts\python.exe -m scripts.10_prepare_dl_cache --config configs/exp_dl_multimodal_oes_aux_mixup_ema_longrun_5fold.yaml --level normalizers
```

생성되는 주요 폴더:

- `cache/v1/dl_tensors/to128_tp30_proc-common/`
  - raw wafer NPZ에서 100-cycle 고정 OES/Process tensor로 resample한 결과
- `cache/v1/dl_normalizers/to128_tp30_proc-common/<split>/<xgb-tag>/`
  - fold별 train-only normalizer stats
- `cache/v1/dl_normalized/to128_tp30_proc-common/<split>/pwnorm0/<xgb-tag>/`
  - fold별 normalized OES/Process/XY/target tensor
- `cache/v1/features/oes_scores/`
  - fold별 OES wavelength correlation score

`--level normalizers`는 normalized tensor까지 저장하지 않고 stats까지만 저장한다. `dl_tensors`는 split seed와 무관하게 1회만 공유되고, seed별로 추가되는 normalizer/score 캐시는 작다.

`--level normalized`는 학습 시작이 가장 빠르지만 full OES tensor를 fold별로 복제하므로 seed당 수십~100GB까지 커질 수 있다. 로컬 SSD 검증용으로만 쓰고, RunPod 업로드 대상에서는 보통 제외한다.

## 2. 학습 config에서 캐시 사용

`scripts/04_train_dl.py`는 기본적으로 `auto` 모드로 동작한다. 즉 config에 `data.dl_cache`가 없어도 사용 가능한 DL 캐시가 있으면 자동으로 가장 빠른 캐시를 선택한다.

우선순위:

1. `normalized`
2. `normalizers`
3. `tensors`
4. 캐시 없음: 기존 계산 경로

RunPod에는 보통 `cache/v1/dl_normalized/`를 업로드하지 않는다. 그러면 auto 모드는 `normalizers` 또는 `tensors`를 선택한다.

RunPod에서 캐시 누락을 바로 잡고 싶을 때만 config의 `data` 아래에 명시한다.

```yaml
data:
  dl_cache:
    mode: "normalized"
    require: true
```

모드 의미:

- `auto`: 기본값. 사용 가능한 캐시를 자동 선택하고, 없으면 기존 방식으로 fallback
- `tensors`: resampled cycle tensor만 디스크에서 로드하고, normalizer fit/apply는 실행 때 수행
- `normalizers`: resampled tensor와 fold별 normalizer stats를 로드하고, normalize apply만 수행
- `normalized`: fold별 normalized tensor를 바로 로드해 fit/apply를 모두 건너뜀
- `none`: 캐시 가속을 끄고 기존 방식으로 실행

## 3. 재생성해야 하는 경우

다음이 바뀌면 캐시를 다시 만든다.

- `cache_version`
- `t_o`, `t_p`
- `split_file`
- `per_wafer_norm`
- `xgb_feat_names`
- 공통 process channel 선택/정렬 로직
- OES/Process cycle tensor resampling 로직
- OES/Process/XY/target normalization 방식
- OES wavelength selection 설정 또는 score 계산 로직

학습 전용 설정만 바뀐 경우 (`epochs`, `lr`, `batch_size`, `dropout`, `mixup`, `ema`, hidden size 등)에는 일반적으로 재생성하지 않아도 된다.
