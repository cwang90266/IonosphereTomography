# feature_table — incremental post-processing of hourly runs

Builds per-run/hour/condition/style feature tables from saved run outputs, a few runs at a time:
only part of the runs fits on the server's local disk, so each session processes whatever is staged and
appends to tables that live elsewhere (e.g. a shared folder).

## Pieces

| File | Role |
|---|---|
| `run_features.py` | CLI: `extract`, `report`, `table` |
| `registry.py` | Run registry (YAML, maintained by hand): run_label → configuration, expected hours/styles |
| `context.py` | Finds staged hour folders; `HourContext` lazily loads and caches inputs (decoded ensembles at the ISR are shared by all extractors) |
| `store.py` | One append-only CSV per extractor, keyed by (run_label, hour); latest extraction wins |
| `extractors/` | Feature extractors (`status`, `peaks`); add new ones here |

## Session on the server

```bash
# 1. stage some run folders under /local/tomography_output, then
python -m Postprocessing.feature_table.run_features extract \
    --results-root /local/tomography_output --store /shared/feature_store \
    --registry /shared/run_registry.yaml \
    --isr MIL=/data/MILISR2025_nonan.nc --ionosonde MIL=/data/MILIONO_Aug2530_2025.nc
# 2. what is done / still to stage
python -m Postprocessing.feature_table.run_features report --store /shared/feature_store --registry /shared/run_registry.yaml
# 3. unstage, stage the next runs, repeat 1-2.  Wide table for plotting:
python -m Postprocessing.feature_table.run_features table --store /shared/feature_store \
    --registry /shared/run_registry.yaml --out /shared/features_wide.csv
```

An hour is (re)processed by an extractor when it has no row yet, when the extractor's version changed
(`VERSION`, plus options that affect results such as `--option peaks.f_top_km=450`), or when the hour
folder changed since (input signature: sizes/mtimes of `hourly_summary.csv`, `mode_timing.csv`, condition
folders). Failures are stored as `status=error` rows with the message; `--retry-failed` retries them,
`--force` redoes everything selected (`--runs`, `--hours`, `--extractors`).

## Writing an extractor

`extractors/<name>.py` with `NAME`, `VERSION`, `NEEDS_DECODE`, `extract(ctx) -> list[dict]` (optionally
`version(options)`), then add it to `extractors/__init__.py:ALL`. Rows should carry `condition` and `style`
(and `batch_index` if per batch). Use `ctx.reconstruction(condition, style)` for the decoded forecast/analysis
at the ISR (cached), `ctx.isr`, `ctx.ionosonde`, `ctx.read_csv(name)`, `ctx.styles(condition)`, `ctx.window`.
Bump `VERSION` whenever the output changes.

## Notes

* Do not point `--store` inside the staging area.
* The registry is the authority for configuration; staged runs missing from it are still processed and
  reported, and `table` falls back to parsing their folder names (`config_source=label-parse`).
* `peaks` reproduces `tabulate_nmf2_hmf2.py`'s forecast/analysis values exactly, but searches the F peak
  only up to 450 km by default (700 km picks spurious night-time topside ISR values at Millstone Hill).
