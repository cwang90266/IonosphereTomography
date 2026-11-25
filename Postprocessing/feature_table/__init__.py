"""Incremental feature tables for hourly data-assimilation runs.

Run on whatever run folders are staged on local disk; each extractor adds rows only for hours it has not
yet processed (at its current VERSION), so repeated sessions with different staged runs fill the tables
up over time. See run_features.py for the command line and README.md for the design.
"""
