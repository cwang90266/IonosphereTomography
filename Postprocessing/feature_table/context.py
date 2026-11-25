"""Discovery of staged hour folders and the per-hour context shared by all extractors.

HourContext loads inputs lazily and caches them, so several extractors that need the same expensive
input (e.g. the decoded forecast/analysis ensemble at the ISR) share one decode.
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "Parameterization", REPO / "EDPSamples", REPO / "IRI_Sample_Inputs",
           REPO / "Spline_Basis", REPO / "Run_Scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

HOUR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{8}$")
CONDITIONS = ("boosted", "no_boost")


def is_run_tree(d: Path) -> bool:
    return d.is_dir() and any(HOUR_RE.match(c.name) and c.is_dir() for c in d.iterdir())


def find_run_trees(roots: list[Path]) -> list[Path]:
    """Each root may be a run tree itself or hold several run trees side by side."""
    trees = []
    for r in roots:
        r = Path(r)
        if is_run_tree(r):
            trees.append(r)
        else:
            trees += sorted(d for d in r.iterdir() if is_run_tree(d))
    return trees


def hour_dirs(tree: Path, hours: list[str] | None = None) -> list[Path]:
    out = sorted(c for c in tree.iterdir() if c.is_dir() and HOUR_RE.match(c.name))
    return [h for h in out if hours is None or h.name in hours]


@dataclass
class Shared:
    """Inputs shared across all hours of a session (loaded once)."""
    isr: dict = field(default_factory=dict)          # code -> IsrDataset
    ionosonde: dict = field(default_factory=dict)    # code -> IonosondeDataset
    registry: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)      # extractor options, e.g. {"peaks.f_top_km": 450}
    center_tolerance_deg: float = 0.5


class HourContext:
    def __init__(self, hour_dir: Path, shared: Shared):
        self.hour_dir = Path(hour_dir)
        self.tree = self.hour_dir.parent
        self.run_label = self.tree.name
        self.hour = self.hour_dir.name
        self.shared = shared
        self.registry_entry = shared.registry.get(self.run_label)
        self._recon = {}

    # ---- cheap metadata ------------------------------------------------------------------
    @cached_property
    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        from Postprocessing.tabulate_nmf2_hmf2 import parse_hour_dir
        return parse_hour_dir(self.hour)

    @cached_property
    def input_signature(self) -> str:
        """Changes when the hour folder is (re)written: size+mtime of its summary files and condition dirs."""
        parts = []
        for name in ("hourly_summary.csv", "mode_timing.csv", *CONDITIONS):
            p = self.hour_dir / name
            if p.exists():
                st = p.stat()
                parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}" if p.is_file() else
                             f"{name}:{len(list(p.iterdir()))}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]

    def read_csv(self, name: str) -> pd.DataFrame | None:
        p = self.hour_dir / name
        return pd.read_csv(p) if p.exists() else None

    @cached_property
    def conditions(self) -> list[str]:
        return [c for c in CONDITIONS if (self.hour_dir / c).is_dir()]

    def edp_path(self, condition: str) -> Path | None:
        hits = sorted((self.hour_dir / condition).glob("*_edp_samples.nc"))
        return hits[0] if hits else None

    def prefix(self, condition: str) -> str | None:
        p = self.edp_path(condition)
        return p.name[: -len("_edp_samples.nc")] if p else None

    def styles(self, condition: str, complete_only: bool = True) -> list[str]:
        """Styles with a *_parameterized.nc; complete_only also requires saved ensemble batches."""
        pre = self.prefix(condition)
        if pre is None:
            return []
        d = self.hour_dir / condition
        out = []
        for p in sorted(d.glob(f"{pre}_*_parameterized.nc")):
            s = p.name[len(pre) + 1: -len("_parameterized.nc")]
            if not complete_only or list((d / f"{pre}_{s}_ensembles").glob("ensemble_batch_*.nc")):
                out.append(s)
        return out

    @cached_property
    def center(self) -> tuple[float | None, float | None]:
        from Postprocessing.tabulate_nmf2_hmf2 import _edp_attrs
        for c in self.conditions:
            p = self.edp_path(c)
            if p is not None:
                a = _edp_attrs(p)
                return a["center_lat"], a["center_lon"]
        return None, None

    def _match(self, datasets: dict):
        lat, lon = self.center
        if lat is None:
            return None, None
        tol = self.shared.center_tolerance_deg
        for code, ds in datasets.items():
            if abs(ds.station.lat - lat) <= tol and abs(ds.station.lon - lon) <= tol:
                return code, ds
        return None, None

    @cached_property
    def isr(self):
        """(code, IsrDataset) of the ISR at this run's centre, or (None, None)."""
        return self._match(self.shared.isr)

    @cached_property
    def ionosonde(self):
        return self._match(self.shared.ionosonde)

    # ---- expensive, cached ---------------------------------------------------------------
    def reconstruction(self, condition: str, style: str):
        """Forecast/analysis ensembles decoded at the ISR station (or run centre), per batch;
        Assimilation_Cycle.isr_comparison._build_batch_isr_matches_from_saved, cached."""
        key = (condition, style)
        if key not in self._recon:
            from Assimilation_Cycle.isr_comparison import _build_batch_isr_matches_from_saved
            from Postprocessing.tabulate_nmf2_hmf2 import _empty_isr_dataset
            code, isr = self.isr
            if isr is not None:
                lat, lon, ds = isr.station.lat, isr.station.lon, isr
            else:
                lat, lon = self.center
                ds = _empty_isr_dataset(lat, lon)
            self._recon[key] = _build_batch_isr_matches_from_saved(
                self.hour_dir / condition / f"{self.prefix(condition)}_{style}", ds, lat, lon,
                ionosonde=self.ionosonde[1])
        return self._recon[key]

    def option(self, name: str, default):
        return self.shared.options.get(name, default)
