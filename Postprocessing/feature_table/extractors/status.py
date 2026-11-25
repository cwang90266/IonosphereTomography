"""Run status, observation counts, state size and wall time per (condition, style); no decoding.

Sources: the hour folder's hourly_summary.csv (one row per condition, incl. failures and their message)
and mode_timing.csv (one row per condition x style), plus which output files exist on disk.
"""
import numpy as np

NAME = "status"
VERSION = 1
NEEDS_DECODE = False


def extract(ctx):
    hs = ctx.read_csv("hourly_summary.csv")
    mt = ctx.read_csv("mode_timing.csv")
    cond_info = {}
    if hs is not None:
        for _, r in hs.iterrows():
            cond_info[r["condition"]] = r
    styles = {}                                   # (condition, style) -> timing row or None
    if mt is not None:
        for _, r in mt.iterrows():
            styles[(r["condition"], r["style"])] = r
    for c in ctx.conditions:
        for s in ctx.styles(c, complete_only=False):
            styles.setdefault((c, s), None)

    rows = []
    for c in sorted(set(cond_info) | {c for c, _ in styles}):
        ci = cond_info.get(c)
        cs = [s for cc, s in styles if cc == c] or [""]      # a failed condition may have no style rows
        complete = set(ctx.styles(c, complete_only=True)) if c in ctx.conditions else set()
        for s in sorted(cs):
            t = styles.get((c, s))
            rows.append(dict(
                condition=c, style=s,
                run_status=None if ci is None else ci.get("status"),
                message=None if ci is None or (isinstance(ci.get("message"), float) and np.isnan(ci.get("message")))
                else str(ci.get("message"))[:300],
                outputs_complete=s in complete,
                n_ro_profiles=(t["n_ro_profiles"] if t is not None else (None if ci is None else ci.get("n_ro_profiles"))),
                n_tec_rays=(t["n_tec_rays"] if t is not None else (None if ci is None else ci.get("n_tec_rays"))),
                n_state=None if t is None else t["n_state"],
                assimilation_wall_time_min=None if t is None else t["assimilation_wall_time_minutes"],
                condition_elapsed_s=None if ci is None else ci.get("elapsed_seconds"),
                initial_edp_source=None if ci is None else ci.get("initial_edp_source"),
                boost_amplitude=None if ci is None else ci.get("boost_amplitude"),
                boost_log_space=None if ci is None else ci.get("boost_log_space"),
            ))
    return rows
