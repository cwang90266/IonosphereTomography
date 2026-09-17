# Makes `Parameterization` importable as `import Parameterization` and
# resolves its `from edp_samples import EDPSamples` dependency, without
# requiring either folder to already be an installed package or on
# PYTHONPATH. Neither IonosphereTomography/Parameterization nor
# IonosphereTomography/EDPSamples has an __init__.py (see the module
# verification plan, section 7.1) -- this is the minimal shim to make the
# test suite runnable today; a proper package layout would remove the need
# for it.
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: plotting tests must not require a display

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

for _p in (_THIS_DIR, _REPO_ROOT / "EDPSamples"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
