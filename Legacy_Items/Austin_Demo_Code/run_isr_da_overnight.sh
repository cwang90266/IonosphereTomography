#!/usr/bin/env bash
# Robust overnight supervisor for demo_isr_da_comparison.py.
#
# demo_isr_da_comparison.py caches progress per (group, obs_mode, filter_type)
# to disk, so re-launching it after a crash or kill just resumes where it
# left off -- no --force needed. This wrapper:
#   1. Self-detaches into its own session (setsid) on first launch, so it is
#      immune to SIGHUP even if you forget nohup and the terminal/SSH session
#      closes. (A prior overnight run died along with its supervisor when the
#      whole process tree got signalled at once -- this closes that gap.)
#   2. Restarts the run immediately if the python process crashes/exits non-zero.
#   3. Watches the log file; if there's no new output for HANG_TIMEOUT seconds
#      (process hung, e.g. on a network call), it kills and restarts the run.
#   4. Exits cleanly once the script finishes successfully (exit code 0).
#   5. Logs its own start/stop/signal events so a silent death overnight is
#      diagnosable instead of just trailing off.
#
# NOTE: none of this can survive a SIGKILL (e.g. the OOM killer) -- SIGKILL
# can't be trapped or ignored, by any process, ever. If the machine runs out
# of memory the whole tree dies instantly with no time to log anything. If
# runs keep dying with no trap message in the log, suspect memory pressure
# and consider an external watchdog (cron) that relaunches this script if
# the pidfile's process isn't alive -- ask before setting that up, since it's
# a persistent, standing change.
#
# Usage:
#   ./run_isr_da_overnight.sh [args passed through to demo_isr_da_comparison.py]
#
# Just run it directly -- it detaches itself. No need to prefix with nohup.

set -uo pipefail

PYTHON=/home/austinhunter/Downloads/PlanetiQ_Code/venv311/bin/python3.11
SCRIPT_DIR="/home/austinhunter/IonosphereTomography"
SCRIPT="demo_isr_da_comparison.py"
LOG_DIR="$SCRIPT_DIR/Data/DA_Cache/run_logs"
PID_FILE="$LOG_DIR/supervisor.pid"
HANG_TIMEOUT=3600   # seconds of no log output => assume hung, kill + restart
POLL_INTERVAL=60    # how often to check the log for staleness
RESTART_DELAY=30    # pause between restarts after a crash

mkdir -p "$LOG_DIR"

# --- Self-detach on first launch so a closed terminal/SSH session (SIGHUP)
# --- can't take this down. Re-exec under setsid in the background, then the
# --- original foreground invocation exits immediately.
if [ -z "${_ISR_DA_DETACHED:-}" ]; then
    BOOTSTRAP_LOG="$LOG_DIR/bootstrap_$(date +%Y%m%d_%H%M%S).log"
    _ISR_DA_DETACHED=1 setsid nohup "$0" "$@" </dev/null >"$BOOTSTRAP_LOG" 2>&1 &
    disown
    echo "[supervisor] detached as pid $! (own session, immune to this terminal closing)"
    echo "[supervisor] bootstrap output: $BOOTSTRAP_LOG"
    echo "[supervisor] pidfile: $PID_FILE"
    exit 0
fi

LOG_FILE="$LOG_DIR/isr_da_comparison_$(date +%Y%m%d_%H%M%S).log"
cd "$SCRIPT_DIR"

echo "$$" > "$PID_FILE"
trap '' HUP
trap 'echo "[supervisor] received SIGTERM, stopping (child pid ${PY_PID:-none}): $(date -Iseconds)" | tee -a "$LOG_FILE"; kill -TERM "${PY_PID:-0}" 2>/dev/null; rm -f "$PID_FILE"; exit 143' TERM
trap 'echo "[supervisor] received SIGINT, stopping (child pid ${PY_PID:-none}): $(date -Iseconds)" | tee -a "$LOG_FILE"; kill -TERM "${PY_PID:-0}" 2>/dev/null; rm -f "$PID_FILE"; exit 130' INT
trap 'rm -f "$PID_FILE"' EXIT

echo "[supervisor] starting $(date -Iseconds)  log=$LOG_FILE  supervisor_pid=$$" | tee -a "$LOG_FILE"

while true; do
    echo "[supervisor] launching run: $(date -Iseconds)" | tee -a "$LOG_FILE"
    "$PYTHON" "$SCRIPT" "$@" >>"$LOG_FILE" 2>&1 &
    PY_PID=$!

    # Watchdog: kill + restart if the log goes quiet for HANG_TIMEOUT seconds.
    while kill -0 "$PY_PID" 2>/dev/null; do
        sleep "$POLL_INTERVAL"
        if kill -0 "$PY_PID" 2>/dev/null; then
            LAST_MOD=$(stat -c %Y "$LOG_FILE")
            NOW=$(date +%s)
            if (( NOW - LAST_MOD > HANG_TIMEOUT )); then
                echo "[supervisor] no log output for ${HANG_TIMEOUT}s, treating as hung -- killing pid $PY_PID: $(date -Iseconds)" | tee -a "$LOG_FILE"
                kill -TERM "$PY_PID" 2>/dev/null
                sleep 10
                kill -KILL "$PY_PID" 2>/dev/null
                break
            fi
        fi
    done

    wait "$PY_PID" 2>/dev/null
    EXIT_CODE=$?
    echo "[supervisor] run exited with code $EXIT_CODE: $(date -Iseconds)" | tee -a "$LOG_FILE"

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[supervisor] completed successfully, exiting supervisor: $(date -Iseconds)" | tee -a "$LOG_FILE"
        break
    fi

    echo "[supervisor] restarting in ${RESTART_DELAY}s (will resume from on-disk cache) ..." | tee -a "$LOG_FILE"
    sleep "$RESTART_DELAY"
done
