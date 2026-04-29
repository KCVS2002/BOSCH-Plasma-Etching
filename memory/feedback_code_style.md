---
name: Code Style Feedback
description: Coding conventions and workflow preferences confirmed during sessions
type: feedback
---

Python 스크립트는 sys.path bootstrap을 스크립트 최상단에 추가해서 `python scripts/NN_name.py`로 직접 실행 가능하게 한다.

**Why:** 스크립트 파일명이 숫자로 시작(`01_build_cache.py`)해서 `python -m scripts.01_...`이 Python 식별자 규칙상 불가능함. CLAUDE.md의 `-m` 방식은 이 점 때문에 사실상 직접 실행으로 대체.

**How to apply:** 모든 `scripts/*.py` 파일 최상단에 아래 패턴 포함:
```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```
