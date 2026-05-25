from pathlib import Path
import numpy as np
import pandas as pd
import sys

cache_root = Path('cache') / 'v1_2024_07_02'
meas = pd.read_csv(cache_root / 'measurements.csv')
wafer_keys = sorted(meas['experiment_key'].astype(str).unique())

n_wafers = len(wafer_keys)
# sample_fold_id: one per sample (89 per wafer)
sample_fold_id = np.zeros(n_wafers * 89, dtype=np.int32)
wafer_fold_id = np.zeros(n_wafers, dtype=np.int32)

out = cache_root / 'splits'
out.mkdir(parents=True, exist_ok=True)
np.savez(out / 'kfold_perday.npz',
         sample_fold_id=sample_fold_id,
         wafer_fold_id=wafer_fold_id,
         wafer_keys=np.array(wafer_keys, dtype=object),
         n_folds=1,
         method='perday-1fold')
print('Wrote', out / 'kfold_perday.npz')
