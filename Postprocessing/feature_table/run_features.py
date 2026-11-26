#!/usr/bin/env python3
"""Incremental feature extraction over whatever run folders are staged on local disk.

Subcommands
  extract  Process staged hour folders; for each extractor, only hours not yet in its table at the current
           extractor version (or whose folder changed since: input_signature) are processed and appended.
  report   Coverage per run x extractor: hours ok / failed / missing, and runs that still need staging.
  table    Join the extractor tables (latest extraction) with the run registry into one wide CSV.

Typical session on the server (repeat with different runs staged):

    python -m Postprocessing.feature_table.run_features extract \\
        --results-root /local/tomography_output --store /shared/feature_store \\
        --registry /shared/run_registry.yaml \\
        --isr MIL=/data/MILISR2025_nonan.nc --ionosonde MIL=/data/MILIONO_Aug2530_2025.nc
    python -m Postprocessing.feature_table.run_features report --store /shared/feature_store \\
        --registry /shared/run_registry.yaml

The store directory must live outside the staging area (it accumulates across sessions).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from Postprocessing.feature_table.context import HourContext, Shared, find_run_trees, hour_dirs  # noqa: E402
from Postprocessing.feature_table.extractors import ALL, version_of  # noqa: E402
from Postprocessing.feature_table.registry import config_table, load_registry  # noqa: E402
from Postprocessing.feature_table.store import KEY, FeatureStore  # noqa: E402


def _parse_options(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = float(v)
        except ValueError:
            pass
        out[k] = v
    return out


def cmd_extract(a):
    from Assimilation_Cycle.isr_comparison import load_ionosonde_dataset, load_isr_dataset

    store = FeatureStore(a.store)
    shared = Shared(
        isr={c: load_isr_dataset(p) for c, p in (x.split("=", 1) for x in a.isr)},
        ionosonde={c: load_ionosonde_dataset(p) for c, p in (x.split("=", 1) for x in a.ionosonde)},
        registry=load_registry(a.registry), options=_parse_options(a.option),
        center_tolerance_deg=a.center_tolerance_deg)
    names = a.extractors or list(ALL)
    mods = sorted((ALL[n] for n in names), key=lambda m: m.NEEDS_DECODE)   # cheap first
    versions = {m.NAME: version_of(m, shared.options) for m in mods}
    done = {m.NAME: store.done(m.NAME) for m in mods}

    trees = find_run_trees(a.results_root)
    if a.runs:
        trees = [t for t in trees if t.name in a.runs]
    unregistered = [t.name for t in trees if shared.registry and t.name not in shared.registry]
    if unregistered:
        print(f"[extract] WARNING: {len(unregistered)} staged run(s) not in the registry "
              f"(config will be label-parsed): {unregistered}", flush=True)

    todo = []
    for t in trees:
        for h in hour_dirs(t, a.hours):
            todo.append(h)
    print(f"[extract] {len(trees)} run trees, {len(todo)} hour folders staged; extractors: "
          + ", ".join(f"{n} v{versions[n]}" for n in versions), flush=True)

    n_new = {m.NAME: 0 for m in mods}
    for i, h in enumerate(todo, 1):
        ctx = HourContext(h, shared)
        for m in mods:
            prev = done[m.NAME].get((ctx.run_label, ctx.hour))
            if not a.force and prev is not None and prev[0] == versions[m.NAME] \
                    and prev[1] == ctx.input_signature and (prev[2] == "ok" or not a.retry_failed):
                continue
            t0 = time.time()
            meta = dict(run_label=ctx.run_label, hour=ctx.hour, extractor_version=versions[m.NAME],
                        input_signature=ctx.input_signature,
                        extracted_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f"))
            try:
                rows = [dict(meta, status="ok", error="", **r) for r in m.extract(ctx)]
                if not rows:                                  # record that the hour was looked at
                    rows = [dict(meta, status="empty", error="no rows (no complete outputs?)")]
            except Exception as exc:                           # keep going; the failure is recorded
                rows = [dict(meta, status="error", error=f"{type(exc).__name__}: {exc}"[:500])]
                if a.verbose:
                    traceback.print_exc()
            store.append(m.NAME, rows)
            n_new[m.NAME] += 1
            print(f"[extract] ({i}/{len(todo)}) {ctx.run_label}/{ctx.hour} {m.NAME}: "
                  f"{rows[0]['status']}, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    print("[extract] done; hours (re)processed: " + ", ".join(f"{k}={v}" for k, v in n_new.items()))


def coverage(store: FeatureStore, registry: dict, extractors) -> pd.DataFrame:
    rows = []
    for name in extractors:
        df = store.read(name)
        seen = df.groupby("run_label") if len(df) else None
        labels = sorted(set(registry) | (set(df.run_label) if len(df) else set()))
        for lab in labels:
            g = seen.get_group(lab) if seen is not None and lab in seen.groups else df.iloc[0:0]
            per_hour = g.groupby("hour").status.first() if len(g) else pd.Series(dtype=str)
            expected = (registry.get(lab) or {}).get("hours")
            n_exp = len(expected) if expected else None
            ok = int((per_hour == "ok").sum())
            rows.append(dict(extractor=name, run_label=lab, hours_expected=n_exp, hours_ok=ok,
                             hours_failed=int((per_hour == "error").sum()), hours_empty=int((per_hour == "empty").sum()),
                             hours_missing=(n_exp - len(set(per_hour.index) & set(expected))) if expected else None,
                             in_registry=lab in registry,
                             last_extracted=g.extracted_utc.max() if len(g) else None))
    return pd.DataFrame(rows)


def cmd_report(a):
    store = FeatureStore(a.store)
    reg = load_registry(a.registry)
    cov = coverage(store, reg, a.extractors or list(ALL))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 80)
    print(cov.to_string(index=False))
    need = cov[(cov.hours_ok == 0) | (cov.hours_missing.fillna(0) > 0)].run_label.unique()
    print(f"\nRuns still to stage (some extractor has no or missing hours): {len(need)}")
    for lab in need:
        print("  ", lab)
    if a.out:
        cov.to_csv(a.out, index=False)


def cmd_table(a):
    store = FeatureStore(a.store)
    reg = load_registry(a.registry)
    names = a.extractors or list(ALL)
    tables = []
    for n in names:
        df = store.read(n)
        df = df[df.status == "ok"].drop(columns=["status", "error", "input_signature"], errors="ignore")
        df = df.rename(columns={c: f"{n}.{c}" for c in ("extractor_version", "extracted_utc")})
        tables.append(df)
    keys = KEY + ["condition", "style"]
    wide = tables[0]
    for n, t in zip(names[1:], tables[1:]):
        k = [c for c in keys if c in wide.columns and c in t.columns]
        wide = wide.merge(t, on=k, how="outer", suffixes=("", f"_{n}"))
    cfg = config_table(reg, extra_labels=wide.run_label.unique() if len(wide) else ())
    wide = wide.merge(cfg, on="run_label", how="left")
    wide.to_csv(a.out, index=False)
    print(f"[table] {len(wide)} rows x {wide.shape[1]} columns -> {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--results-root", nargs="+", required=True, type=Path,
                   help="staging dir(s): a run tree, or a dir holding run trees")
    e.add_argument("--store", required=True, type=Path)
    e.add_argument("--registry", type=Path)
    e.add_argument("--isr", action="append", default=[], metavar="CODE=PATH")
    e.add_argument("--ionosonde", action="append", default=[], metavar="CODE=PATH")
    e.add_argument("--extractors", nargs="*", choices=list(ALL))
    e.add_argument("--runs", nargs="*", help="only these run labels")
    e.add_argument("--hours", nargs="*", help="only these hour folders")
    e.add_argument("--option", action="append", metavar="NAME=VALUE", help="extractor option, e.g. peaks.f_top_km=450")
    e.add_argument("--force", action="store_true", help="re-extract even if up to date")
    e.add_argument("--retry-failed", action="store_true", help="re-extract hours whose last extraction failed")
    e.add_argument("--center-tolerance-deg", type=float, default=0.5)
    e.add_argument("--verbose", action="store_true")
    e.set_defaults(func=cmd_extract)
    r = sub.add_parser("report")
    r.add_argument("--store", required=True, type=Path)
    r.add_argument("--registry", type=Path)
    r.add_argument("--extractors", nargs="*", choices=list(ALL))
    r.add_argument("--out", type=Path)
    r.set_defaults(func=cmd_report)
    t = sub.add_parser("table")
    t.add_argument("--store", required=True, type=Path)
    t.add_argument("--registry", type=Path)
    t.add_argument("--extractors", nargs="*", choices=list(ALL))
    t.add_argument("--out", required=True, type=Path)
    t.set_defaults(func=cmd_table)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
