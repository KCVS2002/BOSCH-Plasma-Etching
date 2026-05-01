"""Model registry. New models register a factory in `_FACTORIES`.

Keeping a string-keyed registry lets configs name models without the
training script importing every backend.
"""
from __future__ import annotations

from typing import Any, Callable


def _xgboost_regressor(params: dict[str, Any]):
    from xgboost import XGBRegressor
    return XGBRegressor(**params)


def _cycle_aware_bilstm(params: dict[str, Any]):
    from .bilstm_vm import build_cycle_aware_bilstm
    return build_cycle_aware_bilstm(params)


_FACTORIES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "xgboost_regressor": _xgboost_regressor,
    "cycle_aware_bilstm": _cycle_aware_bilstm,
}


def make_model(name: str, params: dict[str, Any] | None = None):
    """Build a model from a registry key + params dict."""
    if name not in _FACTORIES:
        raise ValueError(
            f"unknown model {name!r}. Known: {sorted(_FACTORIES)}"
        )
    return _FACTORIES[name](params or {})


__all__ = ["make_model"]
