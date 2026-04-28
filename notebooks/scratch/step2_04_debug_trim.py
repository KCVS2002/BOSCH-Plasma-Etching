"""Debug: print ALL rising edges for one n=101 wafer to understand structure."""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from src.data import list_wafers, load_process_wafer, segment_cycles_by_sf6


def main() -> None:
    wafers = {(w.day, w.wafer_num): w for w in list_wafers()}

    # One of each class
    for key in [("2024_07_09", 5), ("2024_07_02", 1), ("2024_07_09", 10)]:
        wk = wafers[key]
        pw = load_process_wafer(wk)
        seg = segment_cycles_by_sf6(pw)
        t = seg.cycle_starts_s
        iv = np.diff(t)

        print(f"\n{wk.process_group} (n={seg.n_cycles})")
        print(f"  duration={pw.t_rel[-1]:.1f}s")
        print(f"  first 12 edge times: {t[:12]}")
        print(f"  first 11 intervals:  {iv[:11]}")
        print(f"  last  5 edge times:  {t[-5:]}")
        print(f"  last  4 intervals:   {iv[-4:]}")
        # Count how many intervals are in [5.5, 6.5]
        valid = (iv >= 5.5) & (iv <= 6.5)
        print(f"  total valid intervals: {valid.sum()} / {len(valid)}")
        print(f"  bad interval positions: {np.where(~valid)[0].tolist()}")
        print(f"  bad interval values:    {iv[~valid]}")


if __name__ == "__main__":
    main()
