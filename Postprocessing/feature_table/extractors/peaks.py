"""F2 and E peak parameters of forecast, analysis, ISR and ionosonde, per (condition, style, batch).

Same reconstruction and matching as tabulate_nmf2_hmf2.py (forecast/analysis decoded at the ISR station from
the saved ensembles; ISR profile nearest the batch midpoint; model profiles log10-interpolated to the ISR gates;
Run_Scoring.score_runs.retrieve_peak). Differences:
  * the F-peak search top is an option (peaks.f_top_km, default 450 km): with 700 km, spurious night-time
    topside ISR values (Millstone Hill 02-03 UT) are taken as the F peak;
  * E-region peak (NmE/hmE) added; ISR match time recorded.
"""
import numpy as np
import pandas as pd

NAME = "peaks"
VERSION = 1
NEEDS_DECODE = True
DEFAULTS = {"peaks.e_bottom_km": 90.0, "peaks.e_top_km": 150.0, "peaks.f_top_km": 450.0,
            "peaks.e_valley_top_km": 200.0, "peaks.e_valley_ratio": 1.0}


def _opt(ctx, k):
    return ctx.option(k, DEFAULTS[k])


def version(options):
    o = {**DEFAULTS, **{k: v for k, v in options.items() if k in DEFAULTS}}
    return f"{VERSION}|ftop={o['peaks.f_top_km']:g}|etop={o['peaks.e_top_km']:g}"


def _peaks(alt, ne, ctx):
    from score_runs import retrieve_peak
    e_bot, e_top, f_top = _opt(ctx, "peaks.e_bottom_km"), _opt(ctx, "peaks.e_top_km"), _opt(ctx, "peaks.f_top_km")
    nmf2, hmf2 = retrieve_peak(alt, ne, e_top, f_top)
    nme, hme = retrieve_peak(alt, ne, e_bot, e_top, valley_top=_opt(ctx, "peaks.e_valley_top_km"),
                             valley_ratio=_opt(ctx, "peaks.e_valley_ratio"))
    return nmf2, hmf2, nme, hme


def extract(ctx):
    from Assimilation_Cycle.isr_comparison import _interp_log10, nearest_ionosonde_point

    isr_code, isr = ctx.isr
    iono_code, iono = ctx.ionosonde
    start, end = ctx.window
    mid = start + (end - start) / 2
    rows = []
    for c in ctx.conditions:
        for s in ctx.styles(c):
            recon = ctx.reconstruction(c, s)
            for m in recon.matches:
                f_mean = np.nanmean(m.forecast_profile, axis=-1)
                a_mean = np.nanmean(m.analysis_profile, axis=-1)
                r = dict(condition=c, style=s, batch_index=m.batch_index, station_code=isr_code or iono_code or "")
                if isr is not None and m.isr_density is not None:
                    alt = isr.altitude
                    profs = {"forecast": _interp_log10(alt, recon.model_altitude, f_mean),
                             "analysis": _interp_log10(alt, recon.model_altitude, a_mean),
                             "isr": np.asarray(m.isr_density, dtype=float)}
                    r["isr_time"] = m.isr_time
                    r["isr_time_delta_min"] = abs((pd.Timestamp(m.isr_time) - mid).total_seconds()) / 60
                else:                              # no ISR: model peaks on the model grid
                    alt = recon.model_altitude
                    profs = {"forecast": f_mean, "analysis": a_mean}
                for tag, p in profs.items():
                    r[f"NmF2_{tag}"], r[f"hmF2_{tag}"], r[f"NmE_{tag}"], r[f"hmE_{tag}"] = _peaks(alt, p, ctx)
                ref_t = m.isr_time if m.isr_time is not None else mid
                pt = nearest_ionosonde_point(iono, ref_t) if iono is not None else None
                if pt is not None:
                    t_io, r["NmF2_ionosonde"], r["hmF2_ionosonde"] = pt
                    r["ionosonde_time_delta_min"] = abs((pd.Timestamp(t_io) - pd.Timestamp(ref_t)).total_seconds()) / 60
                rows.append(r)
    return rows
