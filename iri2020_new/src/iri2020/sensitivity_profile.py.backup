from __future__ import annotations
from .base import IRI
import datetime
from dateutil.parser import parse
from argparse import ArgumentParser
import numpy as np
from .get_iri_inputs import read_apf107, read_ig_rz 
from statistics import median, mean

def main(time: str, alt_km: list[float], glat: float, glon: float,
    foF2: float = None, hmF2: float = None, B0: float = None, B1: float = None):
    """Height Profile Example"""
    if foF2 is not None and hmF2 is not None and B0 is not None and B1 is not None:
        iono = IRI(time, alt_km, glat, glon, foF2, hmF2, B0, B1)
    else:
        iono = IRI(time, alt_km, glat, glon)
 
    return iono


def cli():
    p = ArgumentParser(description="IRI altitude profile")
    p.add_argument("iParam_Switch", help="switch for parameter selection", type=int)
    p.add_argument("time", help="time of simulation")
    p.add_argument("latlon", help="geodetic latitude, longitude (degrees)", type=float, nargs=2)
    p.add_argument(
        "-alt_km",
        help="altitude START STOP STEP (km)",
        type=float,
        nargs=3,
        default=(80, 1000, 10),
    )
    P = p.parse_args()
    SimulationTime=parse(P.time)
    iono_initial = main(P.time, P.alt_km, *P.latlon)
    print(iono_initial)
    foF2 = iono_initial.foF2.item()
    hmF2 = iono_initial.hmF2.item()
    B0 = iono_initial.B0.item()
    B1 = iono_initial.B1.item()
    f107D = iono_initial.f107D.item()
    f107_81 = iono_initial.f107_81.item()
    IG12 = iono_initial.IG12.item()
    Rz12 = iono_initial.Rz12.item()
    if P.iParam_Switch == None or P.iParam_Switch == 0:
        num_steps = 10
        variation_factors = np.linspace(0.5, 2.0, num_steps)

        perturbed_inputs_foF2 = []
        for factor in variation_factors:
                perturbed_inputs_foF2.append({
                "time": P.time,
                "alt_km": P.alt_km,
                "glat": P.latlon[0],
                "glon": P.latlon[1],
                "foF2": foF2 * factor,
                "hmF2": hmF2,
                "B0": B0,
                "B1": B1
            })
    
        perturbed_inputs_hmF2 = []
        for factor in variation_factors:
            perturbed_inputs_hmF2.append({
                "time": P.time,
                "alt_km": P.alt_km,
                "glat": P.latlon[0],
                "glon": P.latlon[1],
                "foF2": foF2,
                "hmF2": hmF2 * factor,
                "B0": B0,
                "B1": B1
            })
    
        perturbed_inputs_B0 = []
        for factor in variation_factors:
            perturbed_inputs_B0.append({
                "time": P.time,
                "alt_km": P.alt_km,
                "glat": P.latlon[0],
                "glon": P.latlon[1],
                "foF2": foF2,
                "hmF2": hmF2,
                "B0": B0 * factor,
                "B1": B1
            })
    
        perturbed_inputs_B1 = []
        for factor in variation_factors:
            perturbed_inputs_B1.append({
                "time": P.time,
                "alt_km": P.alt_km,
                "glat": P.latlon[0],
                "glon": P.latlon[1],
                "foF2": foF2,
                "hmF2": hmF2,
                "B0": B0,
                "B1": B1 * factor
            })
    
        results_foF2 = []
        for params in perturbed_inputs_foF2:
            iono = IRI(
                params["time"],
                params["alt_km"],
                params["glat"],
                params["glon"],
                params["foF2"],
                params["hmF2"],
                params["B0"],
                params["B1"]
            )
            results_foF2.append(iono)
    
        results_hmF2 = []
        for params in perturbed_inputs_hmF2:
            iono = IRI(
                params["time"],
                params["alt_km"],
                params["glat"],
                params["glon"],
                params["foF2"],
                params["hmF2"],
                params["B0"],
                params["B1"]
            )
            results_hmF2.append(iono) 
    
        results_B0 = []
        for params in perturbed_inputs_B0:
            iono = IRI(
                params["time"],
                params["alt_km"],
                params["glat"],
                params["glon"],
                params["foF2"],
                params["hmF2"],
                params["B0"],
                params["B1"]
            )
            results_B0.append(iono)
    
        results_B1 = []
        for params in perturbed_inputs_B1:
            iono = IRI(
                params["time"],
                params["alt_km"],
                params["glat"],
                params["glon"],
                params["foF2"],
                params["hmF2"],
                params["B0"],
                params["B1"]
            )
            results_B1.append(iono)
    
        try:
            from matplotlib.pyplot import show
            import iri2020.plots as piri
    
            piri.altprofile_sensitivity(1,iono_initial,results_foF2, results_hmF2, results_B0, results_B1)
            show()
        except ImportError:
            pass
    else:
        apf107=read_apf107("/Users/cwang/Documents/Consulting/PlanetIQ/Code/iri2020/src/iri2020/data/")
        ig_rz=read_ig_rz("/Users/cwang/Documents/Consulting/PlanetIQ/Code/iri2020/src/iri2020/data/")
        datenum_f107 = [datetime.date(int(yy), int(mm), int(dd)) for yy, mm, dd in 
            zip(apf107["yr"], apf107["mn"], apf107["dy"])]
        datenum_sim=datetime.date(SimulationTime.year,SimulationTime.month,SimulationTime.day)
        idx_f107=datenum_f107.index(datenum_sim)
        #
        start_end_month=ig_rz["Start_end_month"]
        if start_end_month[0] == 1 :
            first_month =12
            first_year = start_end_month[1] -1
        else:
            first_month = start_end_month[0]-1
            first_year = start_end_month[1]
    
        if start_end_month[2] == 12 :
            last_month =1
            last_year = start_end_month[3] + 1
        else:
            last_month = start_end_month[2]+1
            last_year = start_end_month[3]
 
        month = []
        year = []
        y, m = first_year, first_month
        while (y < last_year) or (y == last_year and m <= last_month):
            month.append(m)
            year.append(y)
            m += 1
            if m > 12:
                m = 1
                y += 1
        day = [15] * len(year)
        datenum_igrz = [datetime.date(int(yy), int(mm), int(dd)) for yy, mm, dd in 
            zip(year, month, day)]
        datenum_sim=datetime.date(SimulationTime.year,SimulationTime.month,15)
        idx_igrz=datenum_igrz.index(datenum_sim)
        # Define range of parameters
        ap107_range=apf107["f107"]
        idx_start=max((0,idx_f107-81))
        idx_end=min((idx_f107+81,len(ap107_range)-1))
        ap107_range=ap107_range[idx_start:idx_end]
        f107D_range=[min(ap107_range),mean(ap107_range),median(ap107_range),max(ap107_range)]
        #
        ap107_range=apf107["f107_81"]
        idx_start=max((0,idx_f107-365))
        idx_end=min((idx_f107+365,len(ap107_range)-1))
        ap107_range=ap107_range[idx_start:idx_end]
        f107_81_range=[min(ap107_range),mean(ap107_range),median(ap107_range),max(ap107_range)]
        #
        ig12_range=ig_rz["ig"]
        idx_start=max((0,idx_igrz-12))
        idx_end=min((idx_igrz+12,len(ig12_range)-1))
        ig12_range=ig12_range[idx_start:idx_end]
        ig12_range=[min(ig12_range),mean(ig12_range),median(ig12_range),max(ig12_range)]
        #
        Rz12_range=ig_rz["rz"]
        idx_start=max((0,idx_igrz-12))
        idx_end=min((idx_igrz+12,len(Rz12_range)-1))
        Rz12_range=Rz12_range[idx_start:idx_end]
        Rz12_range=[min(Rz12_range),mean(Rz12_range),median(Rz12_range),max(Rz12_range)]
        
        
        perturbed_inputs_f107D = []
        for value in f107D_range:
                perturbed_inputs_f107D.append({
                "time": P.time,
                "alt_km": P.alt_km,
                "glat": P.latlon[0],
                "glon": P.latlon[1],
                "f107D": value,
                "f107_81": f107_81,
                "IG12": IG12,
                "Rz12": Rz12
            })
    
        perturbed_inputs_f107_81 = []
        for value in f107_81_range:
            perturbed_inputs_f107_81.append({
                "time": P.time,
                "alt_km": P.alt_km,
                "glat": P.latlon[0],
                "glon": P.latlon[1],
                "f107D": f107D,
                "f107_81": value,
                "IG12": IG12,
                "Rz12": Rz12
            })
    
        perturbed_inputs_IG12 = []
        for value in ig12_range:
            perturbed_inputs_IG12.append({
                "time": P.time,
                "alt_km": P.alt_km,
                "glat": P.latlon[0],
                "glon": P.latlon[1],
                "f107D": f107D,
                "f107_81": f107_81,
                "IG12": value,
                "Rz12": Rz12
            })
    
        perturbed_inputs_Rz12 = []
        for value in Rz12_range:
            perturbed_inputs_Rz12.append({
                "time": P.time,
                "alt_km": P.alt_km,
                "glat": P.latlon[0],
                "glon": P.latlon[1],
                "f107D": f107D,
                "f107_81": f107_81,
                "IG12": IG12,
                "Rz12": value
            })
    
        results_f107D = []
        for params in perturbed_inputs_f107D:
            iono = IRI(
                params["time"],
                params["alt_km"],
                params["glat"],
                params["glon"],
                f107D=params["f107D"],
                f107_81=params["f107_81"],
                IG12=params["IG12"],
                Rz12=params["Rz12"]
            )
            results_f107D.append(iono)
    
        results_f107_81 = []
        for params in perturbed_inputs_f107_81:
            iono = IRI(
                params["time"],
                params["alt_km"],
                params["glat"],
                params["glon"],
                f107D=params["f107D"],
                f107_81=params["f107_81"],
                IG12=params["IG12"],
                Rz12=params["Rz12"]
            )
            results_f107_81.append(iono) 
    
        results_IG12 = []
        for params in perturbed_inputs_IG12:
            iono = IRI(
                params["time"],
                params["alt_km"],
                params["glat"],
                params["glon"],
                f107D=params["f107D"],
                f107_81=params["f107_81"],
                IG12=params["IG12"],
                Rz12=params["Rz12"]
            )
            results_IG12.append(iono)
    
        results_Rz12 = []
        for params in perturbed_inputs_Rz12:
            iono = IRI(
                params["time"],
                params["alt_km"],
                params["glat"],
                params["glon"],
                f107D=params["f107D"],
                f107_81=params["f107_81"],
                IG12=params["IG12"],
                Rz12=params["Rz12"]
            )
            results_Rz12.append(iono)
    
        try:
            from matplotlib.pyplot import show
            import iri2020.plots as piri
    
            #piri.altprofile_sensitivity(2,iono_initial,results_f107D, results_f107_81, results_IG12, results_Rz12)
            piri.altprofile_sensitivity(2,iono_initial,results_f107D, results_f107D, results_IG12, results_Rz12)
            show()
        except ImportError:
            pass


if __name__ == "__main__":
    cli()
