"""Path helpers for reusable DL preprocessing caches."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence


def _slug(text: str) -> str:
    text = text.replace("\\", "/")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = text.strip("-")
    return text or "default"


def split_cache_tag(split_file: str | Path) -> str:
    path = Path(str(split_file).replace("\\", "/"))
    stem = path.with_suffix("").as_posix()
    return _slug(stem.replace("/", "-"))


def xgb_feature_tag(xgb_feat_names: Sequence[str] | None) -> str:
    if not xgb_feat_names:
        return "xgb-none"
    payload = json.dumps(list(xgb_feat_names), ensure_ascii=True, sort_keys=False)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"xgb-{len(xgb_feat_names)}-{digest}"


def tensor_cache_tag(t_o: int, t_p: int) -> str:
    return f"to{int(t_o)}_tp{int(t_p)}_proc-common"


def dl_tensor_cache_root(cache_root: Path, t_o: int, t_p: int) -> Path:
    return Path(cache_root) / "dl_tensors" / tensor_cache_tag(t_o, t_p)


def dl_normalizer_cache_root(
    cache_root: Path,
    t_o: int,
    t_p: int,
    split_file: str | Path,
    xgb_feat_names: Sequence[str] | None = None,
) -> Path:
    return (
        Path(cache_root)
        / "dl_normalizers"
        / tensor_cache_tag(t_o, t_p)
        / split_cache_tag(split_file)
        / xgb_feature_tag(xgb_feat_names)
    )


def dl_normalized_cache_root(
    cache_root: Path,
    t_o: int,
    t_p: int,
    split_file: str | Path,
    per_wafer_norm: bool,
    xgb_feat_names: Sequence[str] | None = None,
) -> Path:
    return (
        Path(cache_root)
        / "dl_normalized"
        / tensor_cache_tag(t_o, t_p)
        / split_cache_tag(split_file)
        / f"pwnorm{int(bool(per_wafer_norm))}"
        / xgb_feature_tag(xgb_feat_names)
    )
