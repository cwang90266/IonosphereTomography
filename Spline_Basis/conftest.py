# Makes the Spline_Basis modules importable as top-level modules (no
# __init__.py), mirroring Parameterization/conftest.py.
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

for _p in (
    _REPO_ROOT, _THIS_DIR, _REPO_ROOT / "Parameterization", _REPO_ROOT / "EDPSamples",
    _REPO_ROOT / "IRI_Sample_Inputs",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
