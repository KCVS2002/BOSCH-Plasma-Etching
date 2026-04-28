# 프로젝트 규칙 — BOSCH Plasma-Etching VM

이 문서는 Claude Code가 매 세션 자동 로드하는 프로젝트 헌법이다.
새 작업을 시작하기 전에 반드시 이 규칙을 참고한다.

---

## 1. 프로젝트 개요 (한 줄 요약)

BOSCH DRIE 플라즈마 에칭 공정의 Virtual Metrology — OES + Process 센서 → 웨이퍼 측정값 (si_etch, oxide_etch) 을 Cycle-Aware Deep Learning 으로 예측하는 학부 졸업논문 프로젝트.

자세한 도메인·전략은 [docs/연구계획서_초안.md](docs/연구계획서_초안.md) 참조.

---

## 2. 폴더 구조와 각 폴더의 역할

```
BOSCH Plasma-Etching/
├── CLAUDE.md                  # 이 파일 (프로젝트 규칙)
├── requirements.txt           # pip pinned (torch는 cu124 인덱스)
├── .gitignore
│
├── Dataset/                   # 원본 데이터. 절대 수정 금지. read-only.
│
├── src/                       # 재사용 라이브러리 코드. IMPORT 전용.
│   ├── data/                  #   로더, cycle 세그멘테이션, 캐시 I/O
│   ├── features/              #   cycle tensor 조립, 정규화, 피처 엔지니어링
│   ├── models/                #   모델 아키텍처 (CNN, LSTM, Transformer, baseline)
│   ├── training/              #   학습 루프, loss, optimizer, scheduler
│   ├── evaluation/            #   metrics, GroupKFold split
│   └── utils/                 #   make_experiment_dir, set_seed, config IO 등
│
├── configs/                   # 실험 config YAML. 1 실험 = 1 파일.
│
├── scripts/                   # 실행 엔트리. `python -m scripts.NN_name` 으로 실행.
│   ├── 01_build_cache.py      #   전처리: raw → cache/vN/
│   ├── 02_make_splits.py      #   GroupKFold 분할 저장
│   ├── 03_train.py            #   학습
│   └── 04_evaluate.py         #   평가
│
├── notebooks/
│   ├── eda/                   # 정식 EDA 스크립트 (재사용, 보존)
│   └── scratch/               # 일회성 탐색 (주기적 정리·삭제 OK)
│
├── cache/                     # 전처리 산출물. gitignored. 재생성 가능.
│   └── vN/                    #   전처리 파이프라인 바뀔 때 v2, v3 로 버전업
│
├── outputs/
│   ├── figures/               # EDA·분석용 독립 그림 (특정 실험 소속 아님)
│   └── experiments/           # ★ 모든 학습/평가 실행 결과 (규칙 3 참조)
│
├── docs/                      # 보고서, 계획서, 발표자료, 다이어그램
│
└── memory/                    # Claude auto-memory (건드리지 말 것)
```

### 폴더별 원칙

- **`src/`는 라이브러리, `scripts/`는 엔트리포인트.** 섞지 말 것. `src/` 모듈은 import-time 부작용 금지 (print, 파일쓰기, CUDA 초기화 등). 실행은 반드시 `scripts/`를 거친다.
- **`Dataset/`에는 절대 쓰지 않는다.** 원본은 수정도 파생파일 저장도 금지. 모든 산출물은 `cache/` 또는 `outputs/` 로 간다.
- **`notebooks/scratch/`는 일회용 쓰레기통.** 가설 검증·디버깅용. 보존 가치가 있으면 `notebooks/eda/`로 승격하거나 `src/`로 흡수한다.

---

## 3. 실험 결과 저장 규칙 ★ (최우선)

> **모든 학습/평가 실행은 `outputs/experiments/` 아래에 새 폴더를 만들어 그 안에 결과를 저장한다.**

### 3.1. 폴더 이름

```
outputs/experiments/<YYYY-MM-DD_HH-MM>_<slug>/
```

- **앞부분은 실행 시작 시각** (분 단위까지). 이름순 정렬 = 시간순 정렬이 되도록 한다.
- **뒷부분은 실험 제목 슬러그** (영문 소문자, 하이픈 구분). 예: `baseline-xgb`, `cnn-lstm-v1`, `oes-only-ablation`.
- 예시: `outputs/experiments/2026-04-17_15-30_baseline-xgb/`

### 3.2. 폴더 내부 구조

```
<experiment-dir>/
├── config.yaml        # 실행에 사용된 config 복사본 (필수)
├── metrics.json       # 최종 성능 지표 (필수)
├── NOTES.md           # 실험 목적·설정요약·결과·배운점 (필수, 자동 생성)
├── logs/              # stdout 로그, train/val 로스 곡선 csv
├── checkpoints/       # 모델 가중치 (best, last)
└── figures/           # 이 실험에서 나온 그림만
```

### 3.3. 폴더 생성은 반드시 `make_experiment_dir` 사용

```python
from src.utils import make_experiment_dir
exp_dir = make_experiment_dir("baseline xgb")
# → outputs/experiments/2026-04-17_15-30_baseline-xgb/ 생성 + 하위 폴더 + NOTES.md 시드
```

손으로 mkdir 하지 말 것. 타임스탬프 형식이 틀리면 정렬이 깨진다.

### 3.4. 기존 폴더 재사용 금지

실험을 다시 돌리면 **새 폴더**를 만든다. 덮어쓰기·이어쓰기는 선후관계를 파괴한다.
(예외: 중간에 죽은 학습을 체크포인트에서 이어갈 때만 같은 폴더 사용 가능. NOTES.md 에 기록.)

### 3.5. `outputs/figures/` vs 실험 폴더의 `figures/`

- **특정 실험에 속하는 그림** → `<experiment-dir>/figures/`
- **실험 독립적인 그림** (EDA, 데이터셋 개요, 전처리 검증) → `outputs/figures/`

EDA 그림은 숫자 prefix로 구분: `01_oes_cycle_overview.png`, `08_gasflow_cycles.png` 등.

---

## 4. 코드 규칙

### 4.1. 실행 방식

프로젝트 루트에서 모듈로 실행한다:

```bash
python -m scripts.01_build_cache --config configs/cache_v1.yaml
python -m scripts.03_train --config configs/exp_baseline.yaml
```

(`python scripts/01_build_cache.py` 직접 실행도 되게 `sys.path` 조작은 하지 않는다 — `-m` 로 충분.)

### 4.2. Config 기반 재현성

- 하이퍼파라미터·경로·seed 등은 `configs/*.yaml`에 둔다.
- 스크립트는 `--config` 인자로 YAML 을 받는다.
- 실험 시작 시 **config를 experiment 폴더로 복사**해서 "무슨 설정으로 돌렸나"를 고정한다.

### 4.3. Seed

- `src.utils.set_seed(seed)` 를 모든 랜덤성 있는 스크립트 최상단에 호출.
- seed 는 config에 명시.

### 4.4. 타입·스타일

- Python 3.11+, `from __future__ import annotations`.
- `@dataclass` 선호, 튜플/딕셔너리 반환보다 타입이 있는 컨테이너.
- docstring 은 모듈·공개 함수에만. 내부 함수·명백한 코드에는 주석 달지 말 것.
- 주석은 "WHY"가 비자명할 때만.

### 4.5. 의존성

- 새 패키지 추가시 반드시 `requirements.txt` 에 pinned version 추가.
- torch 는 cu124 index 유지 (`--index-url https://download.pytorch.org/whl/cu124`).
- jupyter 계열은 설치하지 않는다 (사용자 선호 — .py 스크립트로 작업).

---

## 5. 데이터 & 캐시 규칙

### 5.1. Dataset 경로

- 절대 경로: `Dataset/` (프로젝트 루트 기준). 로더가 `DATASET_DIR` 상수로 관리.
- Windows 한글 경로 이슈: `netCDF4` 는 한글 경로를 열지 못한다. [src/data/loader.py](src/data/loader.py) 의 `_cwd()` 컨텍스트매니저가 chdir로 우회한다 — 새로 netCDF 를 열 땐 이 패턴을 따를 것.

### 5.2. Cache 버전 규칙

- `cache/v1/`, `cache/v2/`, ... — 전처리 파이프라인이 바뀌면 버전 올림.
- 각 버전 디렉터리 루트에 `README.md` 넣어 "v1 과 다른 점", "생성 스크립트·커밋 해시" 기록.
- **캐시는 재생성 가능해야 한다.** 캐시만 있고 생성 로직이 사라지면 안 된다 — 생성 스크립트 + config 를 명시.

### 5.3. CV split

- 반드시 **wafer 단위 GroupKFold** (같은 웨이퍼의 89 포인트가 train/val 에 분산되면 누수).
- split 인덱스는 `cache/vN/splits/kfold_K.npz` 로 저장, 학습 스크립트는 이걸 로드만 한다.

---

## 6. 작업 습관 (Claude 행동 규칙)

- **새 작업 시작 전에 이 파일을 읽는다.** 특히 "실험 결과 저장 규칙"은 매번 확인.
- **질문·수정 요청이 들어오면 먼저 관련 파일을 `Read`** 해서 현재 상태를 확인한 뒤 응답.
- **파일·폴더 새로 만들기 전에 이 문서의 구조와 일치하는지 확인.** 애매하면 사용자에게 묻는다.
- **실험 결과 폴더를 수동 mkdir 하지 말 것.** `make_experiment_dir` 경유.
- **한글 파일·폴더 이름 피하기.** Windows netCDF·일부 툴에서 경로 문제 유발. 문서(`docs/*.md`)는 예외.
- **커밋·push 같은 공유 변경은 사용자 명시 요청 시에만.** 로컬 파일 작업은 자유.

---

## 7. 워크플로 요약 (새 실험 실행 절차)

1. `configs/exp_xxx.yaml` 작성
2. (필요시) `scripts/01_build_cache.py` 로 캐시 생성
3. `python -m scripts.03_train --config configs/exp_xxx.yaml` 실행
4. 스크립트 내부에서 `make_experiment_dir("exp xxx")` 호출 → 실험 폴더 자동 생성
5. config 복사 + 학습 진행 + metrics.json + checkpoint 저장
6. 실험 종료 후 `NOTES.md` 업데이트 (관찰·다음 할 것)

---

## 8. 참고 문서

- 연구 계획: [docs/연구계획서_초안.md](docs/연구계획서_초안.md)
- 아키텍처 다이어그램: [docs/architecture_diagram.png](docs/architecture_diagram.png)
- 파이프라인 다이어그램: [docs/pipeline_diagram.png](docs/pipeline_diagram.png)
