from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Fields promoted to explicit, typed attributes because they are either the
# core 7-component contract (tec, gnss_ecef_km, rec_ecef_km) or common enough
# across RO/IGS to be worth a fixed netCDF variable name and dtype (Plan
# Section 6.1). Everything else a source attaches (arc timing, station
# lat/lon, tangent-point lat/lon, ...) is carried through unmodified via
# ``extra`` so nothing is silently dropped, without hand-coding every
# source-specific field name here.
_CORE_KEYS = {"tec", "rec_ecef_km", "gnss_ecef_km"}
_KNOWN_KEYS = _CORE_KEYS | {
    "obs_type", "tangent_alt_km", "pierce_lat", "pierce_lon", "elev_deg",
    "snr_l1", "snr_l2", "date", "label", "rec_id", "prn_id", "tec_type",
    "obs_source", "occ_type", "abel",
}

# Legacy dict keys (today's prepare_ro_observations.py / prepare_igs_observations.py
# entry shape) that map onto this schema's field names. `LEO`/`GNSS` are the
# rename this schema exists to make (Plan Section 6.1/8a); the others are
# just naming differences already present in the two sources today.
_LEGACY_KEY_MAP = {
    "LEO": "rec_ecef_km",
    "GNSS": "gnss_ecef_km",
    "tangent_km": "tangent_alt_km",
    "ipp_lat": "pierce_lat",
    "ipp_lon": "pierce_lon",
    "leo_id": "rec_id",
    "rx_id": "rec_id",
}


@dataclass
class ObservationEntry:
    """One RO occultation or IGS arc, in the schema Plan Section 6.1 defines.

    ``tec``, ``gnss_ecef_km`` (transmitter), and ``rec_ecef_km`` (receiver --
    the LEO for RO, the ground station for IGS) are the required "7
    component" core. Everything else is optional and source-dependent;
    unmodeled fields are kept in ``extra`` rather than dropped.
    """

    obs_type: str  # "RO" | "IGS"
    tec: np.ndarray  # (n_rays,)
    gnss_ecef_km: np.ndarray  # (3, n_rays)
    rec_ecef_km: np.ndarray  # (3, n_rays)
    tangent_alt_km: np.ndarray | None = None
    pierce_lat: np.ndarray | None = None
    pierce_lon: np.ndarray | None = None
    elev_deg: np.ndarray | None = None
    snr_l1: np.ndarray | None = None
    snr_l2: np.ndarray | None = None
    date: pd.Timestamp | None = None
    label: str = ""
    rec_id: str = ""
    prn_id: str = ""
    tec_type: str = ""
    obs_source: str = ""
    occ_type: str | None = None
    abel: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rays(self) -> int:
        return int(np.asarray(self.tec).shape[0])

    @classmethod
    def from_dict(cls, d: dict, obs_type: str) -> "ObservationEntry":
        """Build an entry from today's RO/IGS clean-entry dict shape.

        Renames the legacy ``LEO``/``GNSS`` (and other legacy-named) keys
        onto this schema's field names (Plan Section 8a); everything not
        recognized is preserved verbatim in ``extra``.
        """
        mapped: dict[str, Any] = {}
        extra: dict[str, Any] = {}

        for raw_key, value in d.items():
            key = _LEGACY_KEY_MAP.get(raw_key, raw_key)
            if key in _KNOWN_KEYS:
                mapped[key] = value
            else:
                extra[raw_key] = value

        for required in ("tec", "rec_ecef_km", "gnss_ecef_km"):
            if required not in mapped:
                raise KeyError(
                    f"ObservationEntry.from_dict: missing required field "
                    f"'{required}' (obs_type={obs_type!r}, label={d.get('label')!r})"
                )

        return cls(obs_type=obs_type, extra=extra, **mapped)

    def to_operator_dict(self) -> dict[str, np.ndarray]:
        """The minimal dict ``EDPSamples.get_observation_operator`` needs."""
        return {
            "rec_ecef_km": np.asarray(self.rec_ecef_km, dtype=np.float64),
            "gnss_ecef_km": np.asarray(self.gnss_ecef_km, dtype=np.float64),
        }
