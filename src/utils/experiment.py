"""Experiment directory helper — enforces the project's naming rule.

Every training/evaluation run MUST create its output folder via
`make_experiment_dir`. The resulting layout is:

    outputs/experiments/YYYY-MM-DD_HH-MM_<slug>/
        config.yaml        # frozen config (caller writes)
        metrics.json       # final metrics (caller writes)
        logs/
        checkpoints/
        figures/
        NOTES.md           # seeded with title + timestamp

See CLAUDE.md → "실험 결과 저장 규칙" for the full rule.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"


def slugify(title: str) -> str:
    """Lower-case, hyphen-separated, ASCII-only slug.

    Korean / non-ASCII characters are dropped — keep slugs short and English
    so the folder name reads cleanly in Windows Explorer / git.
    """
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if not s:
        raise ValueError(f"title {title!r} produced empty slug; use ASCII letters")
    return s


def make_experiment_dir(title: str, root: Path = EXPERIMENTS_DIR) -> Path:
    """Create `outputs/experiments/YYYY-MM-DD_HH-MM_<slug>/` and seed it.

    Returns the created directory path. The timestamp is captured at call
    time so the folder name pins "when this run started".
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    slug = slugify(title)
    exp_dir = root / f"{ts}_{slug}"
    exp_dir.mkdir(parents=True, exist_ok=False)  # fail if collision — rerun a minute later

    for sub in ("logs", "checkpoints", "figures"):
        (exp_dir / sub).mkdir()

    notes = exp_dir / "NOTES.md"
    notes.write_text(
        f"# {title}\n\n"
        f"- Started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- Slug: `{slug}`\n\n"
        f"## 목적\n\n(무엇을 검증/개선하려는 실험인지)\n\n"
        f"## 설정 요약\n\n(핵심 하이퍼파라미터, 데이터 버전)\n\n"
        f"## 결과\n\n(metrics.json 참조, 주요 관찰)\n\n"
        f"## 배운 점 / 다음 할 것\n\n",
        encoding="utf-8",
    )
    return exp_dir
