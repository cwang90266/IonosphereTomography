"""Run registry: run_label -> configuration, maintained by hand in a YAML file.

The registry is the authority for run configuration (folder names have used several conventions over time).
Each entry is a mapping of parameters; any keys are allowed. Keys with special meaning:

  hours:        list of expected hour-folder names (e.g. "2025-08-27_14001500"), or omitted. Used by the
                coverage report to tell "not processed yet" from "not part of this run".
  conditions:   expected conditions, default [boosted, no_boost].
  styles:       expected styles.
  superseded_by / status_note: free text, carried into tables.

Runs found on disk but missing from the registry are still processed; their configuration falls back to
tabulate_nmf2_hmf2.parse_run_label_tokens and is marked config_source="label-parse".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

RESERVED = {"hours", "conditions", "styles"}


def load_registry(path: str | Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with open(path) as f:
        reg = yaml.safe_load(f) or {}
    if not isinstance(reg, dict):
        raise ValueError(f"{path}: top level must be a mapping run_label -> parameters")
    for label, entry in reg.items():
        if entry is None:
            reg[label] = {}
        elif not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {label!r} must be a mapping")
    return reg


def config_table(registry: dict[str, dict], extra_labels=()) -> pd.DataFrame:
    """One row per run_label with its configuration (registry entries, plus label-parsed fallbacks
    for labels not in the registry)."""
    from Postprocessing.tabulate_nmf2_hmf2 import parse_run_label_tokens

    rows = []
    for label, entry in registry.items():
        rows.append(dict(run_label=label, config_source="registry",
                         **{k: v for k, v in entry.items() if k not in RESERVED}))
    for label in sorted(set(extra_labels) - set(registry)):
        rows.append(dict(run_label=label, config_source="label-parse", **parse_run_label_tokens(label)))
    return pd.DataFrame(rows)
