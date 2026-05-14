"""Helper for selecting XGB wafer-level features to inject into the DL model.

Usage (run once to see the candidate list):
    python src/features/xgb_inject.py --cache cache/v1

The output is the YAML snippet to paste into the config's `data.xgb_feat_names`.

Selection criteria (중요도 + 적절성):
  1. Process-only  — OES features are redundant (DL already sees raw 3648-ch OES).
  2. No X, Y       — already handled by FiLM + Fourier encoding.
  3. Temporal stats first — late / slope / drift capture global-cycle trends that
     BiLSTM struggles to extract from raw tensors with only 88 training wafers.
     Avoid mean/std/min/max unless the channel has strong SHAP importance.
  4. Per-channel dedup — keep at most one stat per channel (highest-priority stat).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


# Priority order for stat suffixes. Features earlier in this list are preferred
# when multiple stats of the same channel are candidates.
_STAT_PRIORITY = ("late", "slope", "drift", "early", "mean", "std", "min", "max")

# Process channels ranked by physics relevance for oxide_etch prediction.
# Order: thermal drift > RF power > DC bias > pressure > gas > other.
_CHANNEL_PRIORITY = [
    "Heater1Temp", "Heater2Temp", "Heater3Temp", "Heater4Temp",
    "PlatenRFLoadPower", "PlatenRFPeakToPeak", "PlatenRFReflectedPower",
    "SourceRFLoadPower", "SourceRF2LoadPower",
    "PlatenDcBias",
    "Pressure", "ForeLinePressure",
    "Gas1Flow", "Gas2Flow", "Gas3Flow", "Gas4Flow", "Gas5Flow",
    "Gas7Flow", "Gas8Flow",
    "EpdIntensity",
    "HeliumBPFlow", "HeliumBPPressure",
    "PlatenRFLoadCapacitor", "PlatenRFTuningCapacitor",
    "SourceRFLoadCapacitor", "SourceRFTuningCapacitor",
    "SourceRF2TuningCapacitor",
    "moriInnerCurrent",
]


def select_injection_features(
    feat_csv: Path,
    stat_priority: Sequence[str] | None = None,
    max_stats_per_channel: int = 1,
    top_k: int | None = None,
    interleave_stats: bool = False,
) -> list[str]:
    """Return an ordered list of proc feature names appropriate for DL injection.

    Parameters
    ----------
    feat_csv:
        Path to `cache/vN/features/baseline_xgb_v1.csv` (or .parquet).
    stat_priority:
        Override the default stat priority order.
    max_stats_per_channel:
        How many stats to keep per channel (default 1 = highest-priority only).
    top_k:
        Truncate to this many features after ranking. None = return all.
    interleave_stats:
        If True, cycle round-robin through stat types so the top-K includes
        diverse stats (e.g. late, slope, drift) rather than only the highest-
        priority stat from the top channels. Useful when mean-pool BiLSTM
        already captures average behavior — slope/drift features then add
        non-redundant signal.
    """
    import pandas as pd

    p = Path(feat_csv)
    if not p.exists():
        pq = p.with_suffix(".parquet")
        if pq.exists():
            p = pq
        else:
            raise FileNotFoundError(f"feature table not found: {feat_csv}")

    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    all_cols = [c for c in df.columns if c != "experiment_key"]

    stat_order = list(stat_priority or _STAT_PRIORITY)

    # Build (channel, stat, col_name) tuples for proc features only
    candidates: list[tuple[int, int, str]] = []
    for col in all_cols:
        if not col.startswith("proc_"):
            continue
        parts = col.split("_", 1)[1].rsplit("_", 1)   # strip "proc_", split off stat
        if len(parts) != 2:
            continue
        channel, stat = parts
        if stat not in stat_order:
            continue
        ch_rank = _CHANNEL_PRIORITY.index(channel) if channel in _CHANNEL_PRIORITY else len(_CHANNEL_PRIORITY)
        stat_rank = stat_order.index(stat)
        candidates.append((ch_rank, stat_rank, col))

    candidates.sort(key=lambda t: (t[0], t[1]))

    if interleave_stats:
        # Round-robin by stat type, preserving channel priority within each stat.
        # Top-K then contains diverse stat types from diverse channels.
        per_stat: dict[int, list[str]] = {i: [] for i in range(len(stat_order))}
        for ch_rank, st_rank, col in candidates:
            per_stat[st_rank].append((ch_rank, col))
        for buckets in per_stat.values():
            buckets.sort(key=lambda x: x[0])

        selected: list[str] = []
        seen_channels: dict[str, int] = {}
        idx_per_stat = {i: 0 for i in range(len(stat_order))}
        # Cycle until we hit top_k or all buckets exhausted
        while True:
            advanced = False
            for st_rank in range(len(stat_order)):
                while idx_per_stat[st_rank] < len(per_stat[st_rank]):
                    _, col = per_stat[st_rank][idx_per_stat[st_rank]]
                    idx_per_stat[st_rank] += 1
                    channel = col.split("_", 1)[1].rsplit("_", 1)[0]
                    if seen_channels.get(channel, 0) >= max_stats_per_channel:
                        continue
                    selected.append(col)
                    seen_channels[channel] = seen_channels.get(channel, 0) + 1
                    advanced = True
                    if top_k is not None and len(selected) >= top_k:
                        return selected
                    break
            if not advanced:
                break
        return selected

    # Default: dedup by channel, return in priority order
    seen_channels: dict[str, int] = {}
    selected: list[str] = []
    for _, _, col in candidates:
        channel = col.split("_", 1)[1].rsplit("_", 1)[0]
        count = seen_channels.get(channel, 0)
        if count < max_stats_per_channel:
            selected.append(col)
            seen_channels[channel] = count + 1

    return selected[:top_k] if top_k is not None else selected


# ---------------------------------------------------------------------------
# CLI helper: prints YAML snippet + feature list
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="List XGB features appropriate for DL injection (prints YAML snippet)"
    )
    parser.add_argument("--cache", type=Path, default=Path("cache/v1"),
                        help="cache version directory (default: cache/v1)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="return top K features (default: 20)")
    parser.add_argument("--max-stats", type=int, default=2,
                        help="max stats per channel (default: 2)")
    parser.add_argument("--interleave", action="store_true",
                        help="cycle through stat types (late→slope→drift→…) for diversity")
    parser.add_argument("--stat-priority", nargs="+",
                        default=["late", "slope", "drift"],
                        help="stat types to consider in priority order")
    args = parser.parse_args()

    feat_csv = args.cache / "features" / "baseline_xgb_v1.csv"
    feats = select_injection_features(
        feat_csv,
        stat_priority=args.stat_priority,
        top_k=args.top_k,
        max_stats_per_channel=args.max_stats,
        interleave_stats=args.interleave,
    )

    print(f"\n# Top-{len(feats)} injection candidates (paste into config)")
    print(f"# model.params.xgb_feat_dim must equal {len(feats)}")
    print("data:")
    print("  xgb_feat_names:")
    for f in feats:
        print(f"    - {f}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _cli()
