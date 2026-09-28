# Makes `Assimilation_Cycle` importable as a package, and resolves the
# `Parameterization`/`edp_samples`/`IRI_Sample_inputs` imports needed by
# the (optional) integration tests that exercise the real modules --
# same shim pattern already used by Ensemble_Kalman_Engine/tests/conftest.py,
# Parameterization/conftest.py, EDPSamples/conftest.py, and
# IRI_Sample_Inputs/conftest.py (none of those folders are installed
# packages).
#
# Run with:  /opt/anaconda3/bin/python3 -m pytest Assimilation_Cycle/ -q
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _THIS_DIR.parent
_REPO_ROOT = _PACKAGE_DIR.parent

for _p in (
    _REPO_ROOT,
    _REPO_ROOT / "Parameterization",
    _REPO_ROOT / "EDPSamples",
    _REPO_ROOT / "IRI_Sample_Inputs",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
