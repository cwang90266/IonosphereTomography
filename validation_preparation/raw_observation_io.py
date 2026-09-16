"""Direct NetCDF and Madrigal HDF5 readers; never reads/writes a cache.

Supported: NetCDF groups with timestamps, flat/grouped HDF5 timestamp arrays,
Madrigal Data/Array Layout (including split groups), and Data/Table Layout.
Unknown layouts, ambiguous time axes, and missing station metadata fail loudly.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


def observation_files(root):
    root = Path(root).expanduser()
    suffixes = {".nc", ".nc4", ".h5", ".hdf5", ".hdf"}
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix.lower() in suffixes)
    if not files:
        raise FileNotFoundError(f"No NetCDF/HDF5 observations below {root}")
    return files


def text(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").strip()
    array = np.asarray(value)
    if array.size == 1 and array.ndim:
        return text(array.ravel()[0])
    return str(value).strip()


def key(value):
    return re.sub(r"[^a-z0-9]", "", text(value).lower())


class Variable:
    def __init__(self, data, attrs=None, dimensions=()):
        self.data = np.ma.asarray(data, dtype=float).filled(np.nan).copy()
        attrs = attrs or {}
        self.units = text(attrs.get("units", ""))
        self.dimensions = tuple(dimensions)
        self.shape = self.data.shape
        for attr in ("_FillValue", "missing_value"):
            for fill in np.asarray(attrs.get(attr, [])).ravel():
                try:
                    self.data[self.data == float(fill)] = np.nan
                except (ValueError, TypeError):
                    pass
        # Madrigal missing/assumed/knownbad sentinels can be extremely large.
        self.data[np.abs(self.data) >= 1e30] = np.nan
        self.calendar = text(attrs.get("calendar", "standard"))

    def __getitem__(self, index):
        return np.atleast_1d(self.data)[index]


class Product:
    def __init__(self, variables, attrs, label, madrigal=False):
        self.variables = variables
        self.attrs = {key(k): v for k, v in attrs.items()}
        self.label = label
        self.madrigal = madrigal

    def getncattr(self, name):
        aliases = {
            "instrumentlatitude": ("instrumentlatitude", "instrumentlat", "stationlatitude"),
            "instrumentlongitude": ("instrumentlongitude", "instrumentlon", "stationlongitude"),
            "kindatcode": ("kindatcode", "kindat", "kindofdatacode"),
        }
        for candidate in aliases.get(key(name), (key(name),)):
            if candidate in self.attrs:
                return text(self.attrs[candidate])
        raise AttributeError(f"{self.label}: missing metadata {name}")


# Default units only for recognized standard Madrigal containers/mnemonics.
MADRIGAL_UNITS = {
    "gdalt": "km", "ne": "m-3", "fof2": "MHz", "foe": "MHz",
    "kmztrf": "km", "kmztre": "km", "timestamps": "Unix seconds",
    "ut1_unix": "Unix seconds", "ut2_unix": "Unix seconds",
}


def time_values(product):
    variable = product.variables["timestamps"]
    values = variable.data.ravel()
    units = variable.units
    if "unix" in units.lower():
        return list(pd.to_datetime(values, unit="s", utc=True, errors="coerce"))
    match = re.match(r"(seconds?|minutes?|hours?|days?)\s+since\s+(.+)", units, re.I)
    if not match or variable.calendar not in ("standard", "gregorian", "proleptic_gregorian"):
        raise ValueError(f"{product.label}: unsupported time units/calendar {units!r}")
    scale = {"s": "s", "m": "m", "h": "h", "d": "D"}[match[1][0].lower()]
    epoch = pd.Timestamp(match[2])
    epoch = epoch.tz_localize("UTC") if epoch.tzinfo is None else epoch.tz_convert("UTC")
    return [epoch + pd.to_timedelta(v, unit=scale) if np.isfinite(v) else pd.NaT
            for v in values]


def profile_at(product, name, index, n_time):
    """Use NetCDF dimension identity first; reject ambiguous HDF5 array shapes."""
    variable = product.variables[name]
    data = variable.data
    if data.ndim == 0:
        return data.reshape(1)
    time_dim = product.variables["timestamps"].dimensions
    if time_dim and time_dim[0] in variable.dimensions:
        axis = variable.dimensions.index(time_dim[0])
        return np.take(data, index, axis=axis).ravel()
    if variable.dimensions:
        return data.ravel()  # A static altitude coordinate with named dimensions.
    if data.ndim == 1:
        if name == "gdalt" or n_time == 1:
            return data.ravel()
        raise ValueError(f"{product.label}: ambiguous 1-D {name} versus time")
    axes = [a for a, size in enumerate(data.shape) if size == n_time]
    if len(axes) != 1:
        raise ValueError(
            f"{product.label}: ambiguous time axis for {name} shape {data.shape}; "
            "use a NetCDF export with named dimensions"
        )
    return np.take(data, index, axis=axes[0]).ravel()


def converted(values, variable, quantity):
    unit = variable.units.lower().replace(" ", "").replace("**", "").replace("^", "")
    unit = unit.replace("{", "").replace("}", "")
    units = {
        "height": {"km": 1., "m": .001},
        "density": {"m-3": 1., "1/m3": 1., "cm-3": 1e6, "1/cm3": 1e6},
    }
    if unit not in units[quantity]:
        raise ValueError(f"Unknown {quantity} units {variable.units!r}")
    return np.asarray(values, dtype=float) * units[quantity][unit]


def _nc_products(path, wanted):
    try:
        import netCDF4
    except ImportError as exc:
        raise ImportError("Install netCDF4 in your Python environment to read .nc files") from exc
    with netCDF4.Dataset(path) as root:
        def visit(group, inherited):
            attrs = dict(inherited)
            attrs.update({a: group.getncattr(a) for a in group.ncattrs()})
            if "timestamps" in group.variables:
                variables = {}
                for name in wanted | {"timestamps", "kindat"}:
                    variable = group.variables.get(name)
                    if variable is None:
                        continue
                    # netCDF4 already handles masked values and packed scale/offset.
                    var_attrs = {a: variable.getncattr(a) for a in variable.ncattrs()
                                 if a not in ("_FillValue", "missing_value")}
                    variables[name] = Variable(variable[:], var_attrs, variable.dimensions)
                yield Product(variables, attrs, f"{path}:{group.path}")
            for child in group.groups.values():
                yield from visit(child, attrs)
        products = list(visit(root, {}))
        if not products:
            raise ValueError(f"{path}: no NetCDF group with timestamps")
        yield from products


def _h5_metadata(root):
    attrs = dict(root.attrs)
    units = {}
    metadata = root.get("Metadata")
    if metadata is None:
        return attrs, units
    for obj in metadata.values():
        if not hasattr(obj, "dtype") or not obj.dtype.names:
            continue
        fields = {key(f): f for f in obj.dtype.names}
        if "name" in fields and "value" in fields:
            for row in obj[:]:
                attrs[text(row[fields["name"]])] = text(row[fields["value"]])
        if "mnemonic" in fields and "units" in fields:
            for row in obj[:]:
                units[text(row[fields["mnemonic"]]).lower()] = text(row[fields["units"]])
    return attrs, units


def _h5_products(path, wanted):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install h5py in your Python environment to read HDF5 files") from exc
    with h5py.File(path, "r") as root:
        base_attrs, unit_catalog = _h5_metadata(root)
        groups = []
        if "timestamps" in root:
            groups.append(root)
        def find_groups(_, obj):
            if isinstance(obj, h5py.Group) and "timestamps" in obj:
                groups.append(obj)
        root.visititems(find_groups)
        if groups:
            for group in groups:
                standard = "/Data/Array Layout" in group.name
                attrs = dict(base_attrs)
                ancestors, current = [], group
                while current.name != "/":
                    ancestors.append(current)
                    current = current.parent
                for ancestor in reversed(ancestors):
                    attrs.update(dict(ancestor.attrs))
                variables = {}
                containers = [group]
                containers.extend(group[k] for k in ("1D Parameters", "2D Parameters") if k in group)
                for container in containers:
                    for name in wanted | {"timestamps", "kindat"}:
                        if name not in container or not isinstance(container[name], h5py.Dataset):
                            continue
                        dataset = container[name]
                        var_attrs = dict(dataset.attrs)
                        if "units" not in var_attrs:
                            var_attrs["units"] = unit_catalog.get(
                                name.lower(), MADRIGAL_UNITS.get(name, "") if standard else ""
                            )
                        raw = Variable(dataset[()], var_attrs)
                        # Ordinary HDF5 packing is not decoded automatically.
                        raw.data = raw.data * float(var_attrs.get("scale_factor", 1)) + float(var_attrs.get("add_offset", 0))
                        variables[name] = raw
                yield Product(variables, attrs, f"{path}:{group.name}", standard)
            return

        table = root.get("Data/Table Layout")
        if table is None or not table.dtype.names:
            raise ValueError(
                f"{path}: unsupported HDF5 layout; expected timestamps arrays "
                "or Madrigal Data/Table Layout"
            )
        fields = set(table.dtype.names)
        requested = wanted | {"recno", "timestamps", "ut1_unix", "ut2_unix", "kindat"}
        columns = {name: table.fields(name)[:] for name in requested & fields}
        if "timestamps" in columns:
            stamps = np.asarray(columns["timestamps"], dtype=float)
        elif {"ut1_unix", "ut2_unix"} <= columns.keys():
            # Explicit midpoint of measurement start/end for Madrigal records.
            stamps = .5 * (columns["ut1_unix"] + columns["ut2_unix"])
        else:
            raise ValueError(f"{path}: table has no supported UTC timestamp fields")
        grouping = columns.get("recno", stamps)
        for record in np.unique(grouping):
            mask = grouping == record
            if not mask.any():
                continue
            record_times = np.unique(stamps[mask])
            if len(record_times) != 1:
                raise ValueError(f"{path}: record {record} contains conflicting times")
            variables = {"timestamps": Variable(record_times, {"units": "Unix seconds"}, ("time",))}
            for name in (wanted | {"kindat"}) & columns.keys():
                raw = np.asarray(columns[name][mask], dtype=float)
                var_units = unit_catalog.get(name, MADRIGAL_UNITS.get(name, ""))
                if name in ("gdalt", "ne"):
                    variables[name] = Variable(raw[None, :], {"units": var_units}, ("time", "gate"))
                else:
                    good = raw[np.isfinite(raw) & (np.abs(raw) < 1e30)]
                    if good.size and not np.allclose(good, good[0], rtol=1e-9, atol=0):
                        raise ValueError(f"{path}: {name} varies within record {record}")
                    variables[name] = Variable([good[0] if good.size else np.nan], {"units": var_units}, ("time",))
            yield Product(variables, base_attrs, f"{path}:record {record}", True)


def iter_products(path, wanted):
    """Yield lightweight records; never reads the ionosonde pf matrix."""
    path = Path(path)
    if path.suffix.lower() in (".nc", ".nc4"):
        yield from _nc_products(path, set(wanted))
    else:
        yield from _h5_products(path, set(wanted))


def read_isr_file(path):
    records = []
    for product in iter_products(path, {"gdalt", "ne"}):
        if not {"gdalt", "ne"} <= product.variables.keys():
            continue
        lat = float(product.getncattr("instrument_latitude"))
        lon = float(product.getncattr("instrument_longitude"))
        times = time_values(product)
        try:
            default_kindat = product.getncattr("kindat_code")
        except AttributeError:
            match = re.search(r"MAD(\d+)_", Path(path).name, re.I)
            default_kindat = match[1] if match else None
        for i, timestamp in enumerate(times):
            if pd.isna(timestamp):
                continue
            kindat = default_kindat
            if "kindat" in product.variables:
                codes = product.variables["kindat"].data.ravel()
                kindat = codes[0] if codes.size == 1 else codes[i]
            if kindat is None:
                raise ValueError(f"{product.label}: missing kindat; cannot identify fitted ISR data")
            alt = converted(profile_at(product, "gdalt", i, len(times)), product.variables["gdalt"], "height")
            ne = converted(profile_at(product, "ne", i, len(times)), product.variables["ne"], "density")
            if alt.size != ne.size:
                raise ValueError(f"{product.label}: gdalt and ne profile lengths differ")
            valid_alt = np.isfinite(alt)
            alt, ne = alt[valid_alt], ne[valid_alt]
            records.append(dict(time=timestamp, lat=lat, lon=lon, kindat=str(int(float(kindat))),
                                alt_km=alt, ne_m3=ne, source_file=str(path), source_group=product.label))
    return records
