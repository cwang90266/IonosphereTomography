"""Append-only per-extractor tables.

<store>/<extractor>.csv holds every row ever written by that extractor. Rows are keyed by
(run_label, hour) — an extractor processes a whole hour folder at once — and carry
extractor_version, input_signature and extracted_utc. Re-processing an hour appends new rows;
readers keep only the most recent extraction of each (run_label, hour). Appending (instead of
rewriting) means an interrupted session never loses earlier results, and the table survives
the run folders being unstaged. A lock file serialises concurrent writers.
"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

KEY = ["run_label", "hour"]
META = ["extractor_version", "input_signature", "extracted_utc", "status", "error"]


class FeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, extractor: str) -> Path:
        return self.root / f"{extractor}.csv"

    @contextmanager
    def _lock(self, extractor: str):
        with open(self.root / f".{extractor}.lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def read(self, extractor: str, latest_only: bool = True) -> pd.DataFrame:
        p = self.path(extractor)
        if not p.exists():
            return pd.DataFrame(columns=KEY + META)
        df = pd.read_csv(p, low_memory=False)
        if latest_only and len(df):
            last = df.groupby(KEY).extracted_utc.transform("max")
            df = df[df.extracted_utc == last].reset_index(drop=True)
        return df

    def done(self, extractor: str) -> dict[tuple[str, str], tuple[str, str, str]]:
        """{(run_label, hour): (extractor_version, input_signature, status)} of the latest extraction."""
        df = self.read(extractor)
        if df.empty:
            return {}
        g = df.groupby(KEY).first()
        return {k: (str(r.extractor_version), str(r.input_signature), str(r.status)) for k, r in g.iterrows()}

    def append(self, extractor: str, rows: list[dict]) -> None:
        if not rows:
            return
        new = pd.DataFrame(rows)
        p = self.path(extractor)
        with self._lock(extractor):
            if p.exists():
                cols = list(pd.read_csv(p, nrows=0).columns)
                extra = [c for c in new.columns if c not in cols]
                if extra:                         # new columns (e.g. extractor grew): rewrite with union header
                    old = pd.read_csv(p, low_memory=False)
                    tmp = p.with_suffix(".tmp")
                    pd.concat([old, new], ignore_index=True).to_csv(tmp, index=False)
                    os.replace(tmp, p)
                    return
                new.reindex(columns=cols).to_csv(p, mode="a", header=False, index=False)
            else:
                new.to_csv(p, index=False)

    def compact(self, extractor: str) -> int:
        """Drop superseded extractions from the file; returns rows removed."""
        p = self.path(extractor)
        if not p.exists():
            return 0
        with self._lock(extractor):
            full = pd.read_csv(p, low_memory=False)
            latest = self.read(extractor)
            tmp = p.with_suffix(".tmp")
            latest.to_csv(tmp, index=False)
            os.replace(tmp, p)
        return len(full) - len(latest)
