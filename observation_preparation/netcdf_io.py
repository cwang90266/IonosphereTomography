from __future__ import annotations

from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pandas as pd

from .schema import ObservationEntry

# Fixed, explicitly-modeled per-ray variables (Plan Section 6.2). Only
# created in the file if at least one entry actually sets the field --
# RO-only/IGS-only fields don't pollute the other source's file.
_OPTIONAL_RAY_FIELDS = (
    "tangent_alt_km", "pierce_lat", "pierce_lon",
    "elev_deg", "snr_l1", "snr_l2",
)
_ABEL_FIELDS = ("Ne", "alt_km", "TEC_cal", "TEC_forward")
_ABEL_VAR_NAMES = {
    "Ne": "Abel_Ne",
    "alt_km": "Abel_alt_km",
    "TEC_cal": "Abel_TEC_cal_TECU",
    "TEC_forward": "Abel_TEC_forward_TECU",
}
_STR_SCALAR_FIELDS = ("obs_type", "label", "rec_id", "prn_id", "tec_type", "obs_source", "occ_type")

_ROI_ATTRS = (
    "roi_center_lat", "roi_center_lon", "roi_radius_km", "roi_mode",
    "roi_alt_limit_km", "roi_fraction_required",
)

_EXTRA_PREFIX = "extra_"


def _str_var(nc: netCDF4.Dataset, name: str, dims: tuple[str, ...]) -> netCDF4.Variable:
    return nc.createVariable(name, str, dims)


def _pad_1d(arr: np.ndarray | None, n_ray: int) -> np.ndarray:
    out = np.full(n_ray, np.nan, dtype=np.float64)
    if arr is not None:
        arr = np.asarray(arr, dtype=np.float64).ravel()
        out[: len(arr)] = arr
    return out


def _pad_2d_xyz(arr: np.ndarray, n_ray: int) -> np.ndarray:
    """(3, n) -> (n_ray, 3), NaN-padded."""
    out = np.full((n_ray, 3), np.nan, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    out[: arr.shape[1], :] = arr.T
    return out


def _epoch_seconds(ts) -> float:
    if ts is None or pd.isna(ts):
        return np.nan
    return float(pd.Timestamp(ts).timestamp())


def _classify_extra(entries: list[ObservationEntry], key: str) -> str:
    """'ray' if any entry's value for `key` is an array matching that
    entry's own n_rays (including the n_rays==1 case -- e.g. IGS arcs all
    collapsed to a single epoch), else 'scalar'."""
    for e in entries:
        value = e.extra.get(key)
        if value is None:
            continue
        if hasattr(value, "__len__") and not isinstance(value, str):
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == e.n_rays:
                return "ray"
    return "scalar"


def write_observations(
    entries: list[ObservationEntry],
    path: str | Path,
    *,
    roi: dict[str, Any] | None = None,
) -> Path:
    """Write ``entries`` to one netCDF file, padded ``(obs, ray)`` layout
    (Plan Section 6.2). Replaces the old CSV exporters outright (Section 8.7).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_obs = len(entries)
    n_ray = max((e.n_rays for e in entries), default=0)

    have_abel = [e for e in entries if e.abel]
    n_abel = 0
    if have_abel:
        n_abel = max(
            max(len(np.asarray(e.abel.get(f, [])).ravel()) for f in _ABEL_FIELDS)
            for e in have_abel
        )

    optional_ray_fields = [
        f for f in _OPTIONAL_RAY_FIELDS
        if any(getattr(e, f) is not None for e in entries)
    ]

    extra_keys = sorted({k for e in entries for k in e.extra})
    extra_kind = {k: _classify_extra(entries, k) for k in extra_keys}

    with netCDF4.Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("obs", n_obs)
        nc.createDimension("ray", n_ray)
        nc.createDimension("xyz", 3)
        if n_abel:
            nc.createDimension("abel_level", n_abel)

        if roi:
            for k in _ROI_ATTRS:
                if k in roi and roi[k] is not None:
                    setattr(nc, k, roi[k])

        v_n_rays = nc.createVariable("n_rays", "i4", ("obs",))
        v_date = nc.createVariable("date_epoch_s", "f8", ("obs",))
        v_tec = nc.createVariable("tec", "f4", ("obs", "ray"), fill_value=np.nan)
        v_gnss = nc.createVariable("gnss_ecef_km", "f8", ("obs", "ray", "xyz"), fill_value=np.nan)
        v_rec = nc.createVariable("rec_ecef_km", "f8", ("obs", "ray", "xyz"), fill_value=np.nan)

        str_vars = {name: _str_var(nc, name, ("obs",)) for name in _STR_SCALAR_FIELDS}

        opt_vars = {
            f: nc.createVariable(f, "f4", ("obs", "ray"), fill_value=np.nan)
            for f in optional_ray_fields
        }

        abel_vars = {}
        v_n_abel = None
        if n_abel:
            v_n_abel = nc.createVariable("n_abel_levels", "i4", ("obs",))
            abel_vars = {
                f: nc.createVariable(_ABEL_VAR_NAMES[f], "f4", ("obs", "abel_level"), fill_value=np.nan)
                for f in _ABEL_FIELDS
            }

        extra_vars = {}
        for k in extra_keys:
            var_name = _EXTRA_PREFIX + k
            if extra_kind[k] == "ray":
                extra_vars[k] = nc.createVariable(var_name, "f4", ("obs", "ray"), fill_value=np.nan)
            else:
                sample = next((e.extra[k] for e in entries if k in e.extra), None)
                if isinstance(sample, str):
                    extra_vars[k] = _str_var(nc, var_name, ("obs",))
                else:
                    extra_vars[k] = nc.createVariable(var_name, "f8", ("obs",), fill_value=np.nan)

        for i, e in enumerate(entries):
            v_n_rays[i] = e.n_rays
            v_date[i] = _epoch_seconds(e.date)
            v_tec[i, :] = _pad_1d(e.tec, n_ray)
            v_gnss[i, :, :] = _pad_2d_xyz(e.gnss_ecef_km, n_ray)
            v_rec[i, :, :] = _pad_2d_xyz(e.rec_ecef_km, n_ray)

            for name, var in str_vars.items():
                value = getattr(e, name)
                var[i] = "" if value is None else str(value)

            for f, var in opt_vars.items():
                var[i, :] = _pad_1d(getattr(e, f), n_ray)

            if v_n_abel is not None:
                abel = e.abel or {}
                lens = [len(np.asarray(abel.get(f, [])).ravel()) for f in _ABEL_FIELDS]
                v_n_abel[i] = max(lens) if lens else 0
                for f, var in abel_vars.items():
                    var[i, :] = _pad_1d(abel.get(f), n_abel)

            for k, var in extra_vars.items():
                value = e.extra.get(k)
                if extra_kind[k] == "ray":
                    var[i, :] = _pad_1d(value, n_ray)
                elif isinstance(var.dtype, type) and var.dtype is str:
                    var[i] = "" if value is None else str(value)
                elif value is None:
                    var[i] = np.nan
                else:
                    arr = np.asarray(value)
                    if arr.ndim > 0 and arr.size != 1:
                        # extra fields are an open-ended, per-source catch-all
                        # (schema.py), so _classify_extra's "scalar" call is a
                        # heuristic over entries it may not all agree on. Found
                        # via real-data testing (Plan Section 10 step 5): an
                        # upstream per-epoch field not collapsed consistently
                        # (a missing entry in collapse_igs_arc_to_central_epoch's
                        # field list) reached here as a multi-element array.
                        # Surface that plainly instead of a cryptic TypeError.
                        raise ValueError(
                            f"write_observations: extra field {k!r} was classified "
                            f"scalar but entry {i} (label={e.label!r}) holds a "
                            f"length-{arr.size} array, not a scalar -- check "
                            f"whatever upstream step attached this field for a "
                            f"per-entry-inconsistent shape."
                        )
                    var[i] = float(arr) if arr.ndim > 0 else float(value)

    return path


def read_observations(path: str | Path) -> list[ObservationEntry]:
    """Read a netCDF file written by :func:`write_observations` back into
    ``ObservationEntry`` objects (Plan Section 6.2)."""
    path = Path(path)
    entries: list[ObservationEntry] = []

    with netCDF4.Dataset(path, "r") as nc:
        n_obs = nc.dimensions["obs"].size
        n_rays_all = nc.variables["n_rays"][:]
        dates = nc.variables["date_epoch_s"][:]
        tec_all = nc.variables["tec"][:]
        gnss_all = nc.variables["gnss_ecef_km"][:]
        rec_all = nc.variables["rec_ecef_km"][:]

        str_vals = {
            name: nc.variables[name][:] if name in nc.variables else None
            for name in _STR_SCALAR_FIELDS
        }
        opt_vals = {
            f: nc.variables[f][:] for f in _OPTIONAL_RAY_FIELDS if f in nc.variables
        }

        have_abel = "n_abel_levels" in nc.variables
        n_abel_all = nc.variables["n_abel_levels"][:] if have_abel else None
        abel_vals = {
            f: nc.variables[_ABEL_VAR_NAMES[f]][:]
            for f in _ABEL_FIELDS
            if _ABEL_VAR_NAMES[f] in nc.variables
        }

        extra_names = [
            name[len(_EXTRA_PREFIX):] for name in nc.variables
            if name.startswith(_EXTRA_PREFIX)
        ]
        extra_vals = {name: nc.variables[_EXTRA_PREFIX + name][:] for name in extra_names}

        for i in range(n_obs):
            n = int(n_rays_all[i])
            date_s = float(dates[i])
            date = None if np.isnan(date_s) else pd.Timestamp(date_s, unit="s")

            kwargs: dict[str, Any] = {
                "tec": np.asarray(tec_all[i, :n], dtype=np.float64),
                "gnss_ecef_km": np.asarray(gnss_all[i, :n, :], dtype=np.float64).T,
                "rec_ecef_km": np.asarray(rec_all[i, :n, :], dtype=np.float64).T,
                "date": date,
            }
            for name, values in str_vals.items():
                if name == "obs_type":
                    continue
                if values is not None:
                    v = str(values[i])
                    kwargs[name] = v if v else (None if name == "occ_type" else "")
            for f, values in opt_vals.items():
                kwargs[f] = np.asarray(values[i, :n], dtype=np.float64)

            abel = None
            if have_abel and int(n_abel_all[i]) > 0:
                n_ab = int(n_abel_all[i])
                abel = {f: np.asarray(v[i, :n_ab], dtype=np.float64) for f, v in abel_vals.items()}

            extra: dict[str, Any] = {}
            for name, values in extra_vals.items():
                if values.ndim == 2:
                    extra[name] = np.asarray(values[i, :n], dtype=np.float64)
                else:
                    v = values[i]
                    extra[name] = v.item() if hasattr(v, "item") else v

            obs_type = str(str_vals["obs_type"][i]) if str_vals["obs_type"] is not None else ""
            entries.append(ObservationEntry(obs_type=obs_type, abel=abel, extra=extra, **kwargs))

    return entries
