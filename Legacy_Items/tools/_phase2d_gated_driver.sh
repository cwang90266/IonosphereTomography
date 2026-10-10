#!/usr/bin/env bash
# Phase 2d re-test WITH compound convergence gate (max_iter OR (dP/P<tol AND RMSE<tectol)).
set -u
PY=/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
cd /home/austinhunter/IonosphereTomography
OUT=/tmp/phase2d_gated_results.txt
TECTOL=35
: > "$OUT"

run() {
  local g="$1" m="$2"
  echo "======= GROUP=$g  OBS=$m  bin=5  alpha=0.25  tec_rmse_tol=$TECTOL =======" >> "$OUT"
  "$PY" tools/ekf_tune_harness.py --group "$g" --obs-mode "$m" --bin 5 \
      --alpha 0.25 --tec-rmse-tol "$TECTOL" --free-sets all "log10(NmF2)" 2>&1 \
    | grep -E "prior->post|0\.[0-9]+->|no co-located|FAILED|ValueError|converged at|max_iter" >> "$OUT"
  echo "" >> "$OUT"
}

run 2025-08-27_1227 ro_only
run 2025-08-27_1227 ro_igs
run 2025-10-18_1227 ro_only
run 2025-10-18_1227 ro_igs
echo "ALL PHASE2D GATED RUNS DONE" >> "$OUT"
