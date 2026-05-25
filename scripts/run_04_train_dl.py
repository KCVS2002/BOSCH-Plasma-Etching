"""Wrapper to run the numeric-prefixed script `04_train_dl.py` as a module.

Some scripts in this repo are named with a leading digit (e.g. `04_train_dl.py`),
which is not a valid Python identifier for `import`/`-m` module paths. This small
wrapper lets you run the script with `python -m scripts.run_04_train_dl ...`
while preserving the original script behaviour and CLI args.
"""
from pathlib import Path
import runpy
import sys

SCRIPT_PATH = Path(__file__).parent / "04_train_dl.py"
if not SCRIPT_PATH.exists():
    raise FileNotFoundError(f"Expected script at {SCRIPT_PATH}")

# Forward execution to the original script. `sys.argv` is preserved so CLI
# flags are handled by the underlying script's argparse.
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
