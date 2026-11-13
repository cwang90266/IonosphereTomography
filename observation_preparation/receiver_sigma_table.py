"""Per-receiver, per-tangent-altitude-bin TEC measurement-noise table.

Some RO receivers (e.g. YM08) have been found to carry substantially
higher TEC uncertainty than others at a given tangent altitude, so a
single project-wide ``obs_sigma`` constant is too coarse. This module
reads a user-supplied CSV lookup table and resolves a per-ray sigma for
RO entries, replacing the constant wherever the table covers that
ray's receiver and tangent altitude. IGS entries have no tangent-point
concept, and RO rays with a receiver or altitude the table doesn't
cover, always fall back to the caller-supplied constant.

CSV layout (positional, not by exact header name): the first two
columns give each row's tangent-altitude bin as ``[alt_min_km,
alt_max_km]`` (inclusive on both ends); every column after that is one
receiver, named by its header (e.g. ``GN04``, ``GN05``, ``YM08``), with
TEC-sigma values (TECU) in its rows. A blank cell is treated as
"not covered" for that bin/receiver, falling back to the constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .schema import ObservationEntry


@dataclass(frozen=True)
class ReceiverSigmaTable:
    alt_min_km: np.ndarray  # (n_bins,)
    alt_max_km: np.ndarray  # (n_bins,)
    receivers: list[str]
    sigma: np.ndarray  # (n_bins, n_receivers), TECU; NaN = not covered

    def lookup(self, receiver: str, tangent_alt_km: np.ndarray) -> np.ndarray:
        """Per-altitude sigma for ``receiver``, NaN where no bin covers it
        (unknown receiver, altitude outside every bin, or a blank cell)."""
        tangent_alt_km = np.asarray(tangent_alt_km, dtype=float)
        out = np.full(tangent_alt_km.shape, np.nan)
        if receiver not in self.receivers:
            return out
        col = self.receivers.index(receiver)
        for i in range(len(self.alt_min_km)):
            mask = (
                np.isnan(out)
                & (tangent_alt_km >= self.alt_min_km[i])
                & (tangent_alt_km <= self.alt_max_km[i])
            )
            out[mask] = self.sigma[i, col]
        return out


def load_receiver_sigma_table(path: str | Path) -> ReceiverSigmaTable:
    df = pd.read_csv(path)
    if df.shape[1] < 3:
        raise ValueError(
            f"receiver sigma table {path} needs at least 3 columns "
            f"(alt_min_km, alt_max_km, and >=1 receiver column); got {df.shape[1]}"
        )
    alt_min_km = df.iloc[:, 0].to_numpy(dtype=float)
    alt_max_km = df.iloc[:, 1].to_numpy(dtype=float)
    receivers = [str(c) for c in df.columns[2:]]
    sigma = df.iloc[:, 2:].to_numpy(dtype=float)
    return ReceiverSigmaTable(
        alt_min_km=alt_min_km, alt_max_km=alt_max_km, receivers=receivers, sigma=sigma,
    )


def match_receiver(entry: "ObservationEntry", receivers: list[str]) -> str | None:
    """Identify which of ``receivers`` an entry came from by substring
    search over its label/rec_id/obs_source (RO receiver codes like
    YM08/GN04 show up verbatim in the podTc2 filename carried in
    ``label``, which is more reliable than ``rec_id`` -- some file
    vintages store a bare digit there instead of the full code).
    Case/``-``/``_``-insensitive; the longest matching receiver name
    wins if more than one substring-matches."""
    text = " ".join(
        str(v) for v in (entry.label, entry.rec_id, entry.obs_source) if v
    ).upper().replace("-", "").replace("_", "")
    matches = [r for r in receivers if r.upper().replace("-", "").replace("_", "") in text]
    if not matches:
        return None
    return max(matches, key=len)


def sigma_for_entry(
    entry: "ObservationEntry", table: ReceiverSigmaTable | None, default_sigma: float,
) -> np.ndarray:
    """Per-ray TEC sigma (TECU) for one entry: ``default_sigma`` for IGS
    (no tangent altitude) and whenever no table is supplied; for RO,
    the table's value wherever its receiver/altitude is covered, and
    ``default_sigma`` elsewhere (unmatched receiver, out-of-table
    altitude, or a blank cell)."""
    n = entry.n_rays
    if table is None or entry.tangent_alt_km is None:
        return np.full(n, default_sigma, dtype=float)
    receiver = match_receiver(entry, table.receivers)
    if receiver is None:
        return np.full(n, default_sigma, dtype=float)
    looked_up = table.lookup(receiver, entry.tangent_alt_km)
    return np.where(np.isnan(looked_up), default_sigma, looked_up)
