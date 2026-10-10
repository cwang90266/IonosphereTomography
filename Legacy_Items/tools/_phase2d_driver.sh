#!/usr/bin/env bash
# Phase 2d: validate NmF2-freeze across ISR groups + obs modes.
set -u
PY=/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
cd /home/austinhunter/IonosphereTomography
OUT=/tmp/phase2d_results.txt
: > "$OUT"

run() {
  local g="$1" m="$2"
  echo "======= GROUP=$g  OBS=$m  bin=5  alpha=0.25 =======" >> "$OUT"
  "$PY" tools/ekf_tune_harness.py --group "$g" --obs-mode "$m" --bin 5 \
      --alpha 0.25 --free-sets all "log10(NmF2)" 2>&1 \
    | grep -E "prior->post|0\.[0-9]+->|no co-located|FAILED|ValueError" >> "$OUT"
  echo "" >> "$OUT"
}

run 2025-08-27_1227 ro_only
run 2025-08-27_1227 ro_igs
run 2025-09-26_1227 ro_only
run 2025-09-26_1227 ro_igs
run 2025-10-18_1227 ro_only
run 2025-10-18_1227 ro_igs
echo "ALL PHASE2D RUNS DONE" >> "$OUT"
