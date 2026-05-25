import shutil
from pathlib import Path
src = Path('cache') / 'v1' / 'splits' / 'kfold5_wafer.npz'
dst_dir = Path('cache') / 'v1_2024_07_02' / 'splits'
dst_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst_dir / src.name)
print('COPIED', src, '->', dst_dir / src.name)
