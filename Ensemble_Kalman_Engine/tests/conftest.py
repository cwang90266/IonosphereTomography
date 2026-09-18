# Makes `Ensemble_Kalman_Engine` importable as a package, and resolves the
# `Parameterization`/`edp_samples` imports needed by the (optional)
# integration tests that exercise real EDP_Parameterization styles --
# same shim pattern already used by Parameterization/conftest.py and
# EDPSamples/conftest.py (neither folder is an installed package).
#
# Run with:  /opt/anaconda3/bin/python3 -m pytest Ensemble_Kalman_Engine/ -q
# (project's numpy/scipy stack lives in the base conda environment).
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_REPO_ROOT = _PACKAGE_DIR.parent

for _p in (_REPO_ROOT, _REPO_ROOT / "Parameterization", _REPO_ROOT / "EDPSamples"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
