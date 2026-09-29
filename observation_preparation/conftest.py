# Makes `edp_samples` importable as a top-level module for
# test_diagnostics.py's plot_tec_comparison check -- same shim pattern
# already used by EDPSamples/conftest.py, Parameterization/conftest.py,
# and Ensemble_Kalman_Engine/tests/conftest.py (none of these folders are
# installed packages).
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

for _p in (_REPO_ROOT / "EDPSamples",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass
