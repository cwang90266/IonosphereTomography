#!/usr/bin/env bash
# Phase 2a: adaptive-alpha stabilization. Does it rescue all-free convergence,
# and does converged all-free beat frozen? Gate=40 (frozen floor ~40 converges).
set -u
PY=/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
cd /home/austinhunter/IonosphereTomography
OUT=/tmp/phase2a_results.txt
: > "$OUT"

run() {
  local g="$1" m="$2"
  echo "======= GROUP=$g OBS=$m  adapt_alpha alpha0=0.25 alpha_max=1.0 gate=40 =======" >> "$OUT"
  "$PY" tools/ekf_tune_harness.py --group "$g" --obs-mode "$m" --bin 5 \
      --alpha 0.25 --adapt-alpha --alpha-max 1.0 --tec-rmse-tol 40 \
      --free-sets all "log10(NmF2)" 2>&1 \
    | grep -E "prior->post|0\.[0-9]+->|no co-located|FAILED|converged at|max_iter" >> "$OUT"
  echo "" >> "$OUT"
}

run 2025-08-27_1227 ro_only
run 2025-08-27_1227 ro_igs
run 2025-10-18_1227 ro_only
run 2025-10-18_1227 ro_igs
echo "ALL PHASE2A RUNS DONE" >> "$OUT"
