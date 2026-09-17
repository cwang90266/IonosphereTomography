# Makes `IRI_Sample_inputs` importable without requiring the folder to
# already be an installed package or on PYTHONPATH.
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
