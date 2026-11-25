"""Feature extractors.

An extractor is a module with
    NAME          table name (one CSV per extractor in the store)
    VERSION       bump when the extractor's output changes -> every hour is re-extracted
    NEEDS_DECODE  True if it uses ctx.reconstruction() (slow); cheap extractors run first
    extract(ctx)  -> list of row dicts for one hour folder (ctx: context.HourContext). Rows normally carry
                  'condition' and 'style' (and 'batch_index' if per batch); run_label/hour and bookkeeping
                  columns are added by the driver.
and optionally
    version(options) -> str   VERSION plus any options that change results (recorded per row).

To add one: write extractors/<name>.py and add it to ALL below.
"""
from . import peaks, status

ALL = {m.NAME: m for m in (status, peaks)}


def version_of(mod, options: dict) -> str:
    return mod.version(options) if hasattr(mod, "version") else str(mod.VERSION)
