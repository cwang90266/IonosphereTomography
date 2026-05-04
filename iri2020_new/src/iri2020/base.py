from __future__ import annotations
import subprocess
import logging
from dateutil.parser import parse
from datetime import datetime
import xarray
import io
import os
import numpy as np
import importlib.resources as impr
from .build import build

#from .build import build

SIMOUT = ["ne", "Tn", "Ti", "Te", "nO+", "nH+", "nHe+", "nO2+", "nNO+", "nCI", "nN+"]

__all__ = ["IRI"]


def IRI(time: str | datetime, altkmrange: list[float], glat: float, glon: float,
        foF2: float = None, hmF2: float = None, B0: float = None, B1: float = None,
        f107D: float=None, ap:float = None, IG12: float = None, Rz12: float = None) -> xarray.Dataset:
    if isinstance(time, str):
        time = parse(time)

    assert len(altkmrange) == 3, "altitude (km) min, max, step"
    assert isinstance(glat, (int, float)) and isinstance(
        glon, (int, float)
    ), "glat, glon is scalar"
    # %% build IRI executable if needed
    iri_name = "iri2020_driver"
    if os.name == "nt":
        iri_name += ".exe"

    # %% run IRI
    with (
        impr.as_file(impr.files(__package__).joinpath(iri_name)) as exe,
        impr.as_file(impr.files(__package__).joinpath("data")) as data_path,
    ):

        if not exe.is_file():
            build()

        if foF2 is not None and hmF2 is not None and B0 is not None and B1 is not None:
            #print(f"foF2= {foF2}, hmF2={hmF2}, B0={B0}, B1={B1}")
            cmd = [
                str(exe),
                str(time.year),
                str(time.month),
                str(time.day),
                str(time.hour),
                str(time.minute),
                str(time.second),
                str(glat),
                str(glon),
                str(altkmrange[0]),
                str(altkmrange[1]),
                str(altkmrange[2]),
                str(1),
                str(foF2),
                str(hmF2),
                str(B0),
                str(B1)
            ]
        elif f107D is not None and ap is not None and IG12 is not None and Rz12 is not None:
            #print(f"f107D= {f107D}, ap={ap}, IG12={IG12}, Rz12={Rz12}")
            cmd = [
                str(exe),
                str(time.year),
                str(time.month),
                str(time.day),
                str(time.hour),
                str(time.minute),
                str(time.second),
                str(glat),
                str(glon),
                str(altkmrange[0]),
                str(altkmrange[1]),
                str(altkmrange[2]),
                str(0),
                str(f107D),
                str(ap),
                str(IG12),
                str(Rz12)
            ]
        else:
            cmd = [
                str(exe),
                str(time.year),
                str(time.month),
                str(time.day),
                str(time.hour),
                str(time.minute),
                str(time.second),
                str(glat),
                str(glon),
                str(altkmrange[0]),
                str(altkmrange[1]),
                str(altkmrange[2])
            ]

        logging.info(" ".join(cmd))
        ret = subprocess.check_output(cmd, text=True, cwd=data_path)

    logging.debug(ret)
    if not ret:
        raise RuntimeError("IRI failed to run correctly--gave empty text output")
    # %% get altitude profile data
    Nalt = int((altkmrange[1] - altkmrange[0]) // altkmrange[2]) + 1

    arr = np.genfromtxt(io.StringIO(ret), max_rows=Nalt)
    arr = np.atleast_2d(arr)

    assert arr.ndim == 2 and arr.shape[1] == 12, f"bad text data output format, shape {arr.shape}"

    dsf = {k: (("alt_km"), v) for (k, v) in zip(SIMOUT, arr[:, 1:].T)}
    altkm = arr[:, 0]
    # %% get parameter data
    arr = np.genfromtxt(io.StringIO(ret), skip_header=Nalt)
    assert arr.ndim == 1 and arr.size == 100, "bad text data output format"
    # %% assemble output
    iono = xarray.Dataset(
        dsf,
        coords={"time": [time], "alt_km": altkm, "glat": glat, "glon": glon},
        attrs={"f107D": arr[40], "ap": arr[50], "f107_81": arr[45], "IG12": arr[38], "Rz12": arr[32]},
    )

    for i, p in enumerate(["NmF2", "hmF2", "NmF1", "hmF1", "NmE", "hmE"]):
        iono[p] = (("time"), [arr[i]])

    iono["TEC"] = (("time"), [arr[36]])
    iono["EqVertIonDrift"] = (("time"), [arr[43]])
    iono["foF2"] = (("time"), [arr[99]])
    iono["B0"] = (("time"), [arr[9]])
    iono["B1"] = (("time"), [arr[34]])
    #print(iono)
    return iono
