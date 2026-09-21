# Makes `edp_samples` importable without requiring the folder to already be
# an installed package or on PYTHONPATH, and forces a headless matplotlib
# backend before cartopy/matplotlib get imported by the module under test.
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
