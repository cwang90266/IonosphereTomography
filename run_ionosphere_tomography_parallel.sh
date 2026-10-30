#!/usr/bin/env bash
#
# Run a Python script in parallel across a range of dates and hour windows.
#
# Hour windows are START-END, where END is exclusive and must be greater than START.
# START is 00-23 and END is 01-24, so "23-24" is the last hour of the day and
# "00-24" is the full day. Windows never cross midnight, so --date stays constant.
#
# Example:
#   ./run_parallel.sh -s 2026-10-01 -e 2026-10-07 -i -j 4
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Defaults --------------------------------------------------------------
NICE=19
LOGDIR="parallel_jobs_logs"
JOBS=4
MAX_JOBS=64
SCRIPT="/home/knelson/src/IonosphereTomography/run_hourly_all4styles_boost_compare.py"
PYTHON="${PYTHON:-python3}"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
HOURS="00-24"
JOBLOG="ionosphere_tomography_parallel_jobs.log"
DRY_RUN=""
HOURLY=""
RETRIES=0
MAX_RETRIES=10
RESUME=""
LABEL=""
JOBLOG_SET=""
START_DATE=""
END_DATE=""

usage() {
    cat <<EOF
Usage: $(basename "$0") -s START_DATE [options]

Required:
  -s DATE     Start date (YYYY-MM-DD, UTC)

Options:
  -e DATE     End date, inclusive (YYYY-MM-DD, UTC). Defaults to start date.
  -H RANGES   Comma-separated hour windows START-END (HH-HH), END exclusive.
              START is 00-23, END is 01-24 and must be greater than START.
              Spaces are ignored. Default: "$HOURS" (the full day)
              e.g. "00-06,06-12,12-18,18-24"
  -j N        Max number of parallel processes, 1 to $MAX_JOBS. Default: $JOBS
  -p PATH     Python script to run. Default: $SCRIPT
  -i          Hourly mode: split each window into one-hour jobs
              (e.g. "21-24" runs 21-22, 22-23 and 23-24)
  -t LABEL    Label added to log file names (letters, digits, . _ - only).
              Per-job logs become ionosphere_tomography_ParallelRun_LABEL_d...log,
              and the default job log becomes ionosphere_tomography_parallel_jobs_LABEL.log
  -l FILE     GNU Parallel job log file. Default: $JOBLOG
              (with -t: ionosphere_tomography_parallel_jobs_LABEL.log)
  -L PATH     Directory to place logs from individual python script jobs
  -N N        Niceness, 0 to 19. Default: $NICE
  -r N        Retry each failed job up to N more times, 0 to $MAX_RETRIES. Default: $RETRIES
  -R          Resume: rerun only jobs that failed or never ran, per the job log
  -n          Dry run: print the commands without executing them
  -h          Show this help

Environment:
  PYTHON       Python interpreter name inside the pipenv virtualenv. Default: python3
  PROJECT_DIR  Directory holding the Pipfile and Driver_Scripts/.
               Default: this script's directory ($SCRIPT_DIR)
  OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS
               Threads per job for numeric libraries. Default: 1 each
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

# Make a path absolute relative to the directory the script was launched from
abspath() {
    if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}

# ---- Parse arguments -------------------------------------------------------
while getopts ":s:e:H:j:p:l:L:N:r:t:inRh" opt; do
    case "$opt" in
        s) START_DATE="$OPTARG" ;;
        e) END_DATE="$OPTARG" ;;
        H) HOURS="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        p) SCRIPT="$OPTARG" ;;
        l) JOBLOG="$OPTARG"; JOBLOG_SET=1 ;;
        t) LABEL="$OPTARG" ;;
        i) HOURLY=1 ;;
        n) DRY_RUN=1 ;;
        N) NICE="$OPTARG" ;;
        L) LOGDIR="$OPTARG" ;;
        r) RETRIES="$OPTARG" ;;
        R) RESUME=1 ;;
        h) usage; exit 0 ;;
        :) die "Option -$OPTARG requires a value" ;;
        \?) die "Unknown option -$OPTARG (use -h for help)" ;;
    esac
done
shift $((OPTIND - 1))
(( $# == 0 )) || die "Unexpected argument(s): $* (use -h for help)"

# ---- Validation ------------------------------------------------------------
[[ -n "$START_DATE" ]] || { usage >&2; exit 1; }
END_DATE="${END_DATE:-$START_DATE}"
command -v parallel >/dev/null 2>&1 || die "GNU parallel is not installed"
command -v pipenv   >/dev/null 2>&1 || die "pipenv is not installed"
[[ -f "$SCRIPT" ]] || die "Script not found: $SCRIPT"
# Digit-count limit stops huge values from overflowing the arithmetic check
[[ "$JOBS" =~ ^[1-9][0-9]{0,2}$ ]] && (( JOBS <= MAX_JOBS )) ||
    die "-j must be an integer from 1 to $MAX_JOBS"
[[ "$RETRIES" =~ ^(0|[1-9][0-9]?)$ ]] && (( RETRIES <= MAX_RETRIES )) ||
    die "-r must be an integer from 0 to $MAX_RETRIES"
if [[ -n "$LABEL" ]]; then
    [[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || die "-t label may contain only letters, digits, '.', '_' and '-'"
    # An explicit -l always wins; otherwise label the default job log too
    [[ -n "$JOBLOG_SET" ]] || JOBLOG="ionosphere_tomography_parallel_jobs_${LABEL}.log"
fi
[[ "$NICE" =~ ^([0-9]|1[0-9])$ ]] || die "-N must be 0-19"

# Resolve relative paths now, before we cd into PROJECT_DIR
SCRIPT="$(abspath "$SCRIPT")"
LOGDIR="$(abspath "$LOGDIR")"
JOBLOG="$(abspath "$JOBLOG")"

# Run a frozen copy of the Python script. Each job reads the script when it
# STARTS, so editing LABEL_OF_EACH_RUN or other settings while this run is
# queued would otherwise change jobs that have not started yet, and mix them
# into another run's output folder.
# (Modules imported from the repo are not frozen; don't edit them mid-run.)
if [[ -z "$DRY_RUN" ]]; then
    mkdir -p "$LOGDIR"
    SNAPSHOT="$LOGDIR/$(basename "$SCRIPT" .py)${LABEL:+_$LABEL}_$(date -u +%Y%m%dT%H%M%SZ)_$$.py"
    cp -p "$SCRIPT" "$SNAPSHOT"
    echo "Running frozen copy of the script: $SNAPSHOT" >&2
    SCRIPT="$SNAPSHOT"
fi

validate_date() {
    # Checks both the format and that it's a real calendar date (e.g. rejects 2026-02-30).
    # UTC avoids failures in time zones where local midnight doesn't exist on DST days.
    [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] &&
    [[ "$(date -u -d "$1" +%F 2>/dev/null)" == "$1" ]]
}

validate_date "$START_DATE" || die "Invalid start date: $START_DATE"
validate_date "$END_DATE"   || die "Invalid end date: $END_DATE"
[[ ! "$START_DATE" > "$END_DATE" ]] || die "Start date is after end date"

# ---- Build date list -------------------------------------------------------
dates=()
d="$START_DATE"
while [[ ! "$d" > "$END_DATE" ]]; do
    dates+=("$d")
    d="$(date -u -d "$d + 1 day" +%F)"
done

# ---- Build hour-window list ------------------------------------------------
# Each entry is "START_HOUR END_HOUR".
HOURS="${HOURS//[[:space:]]/}"
hour_re='^([01][0-9]|2[0-4])$'
windows=()
declare -A hour_owner=()   # hour -> window that already covers it
IFS=',' read -ra raw_ranges <<< "$HOURS"
for r in "${raw_ranges[@]}"; do
    IFS='-' read -r sh eh <<< "$r"
    [[ "$sh" =~ $hour_re && "$eh" =~ $hour_re ]] || die "Invalid hour window: '$r' (expect HH-HH 00-24)"
    s=$(( 10#$sh ))
    e=$(( 10#$eh ))
    (( s <= 23 )) || die "Start hour must be 00-23 in window: '$r'"
    (( e > s ))   || die "End hour must be greater than start hour in window: '$r'"

    # Without -i, overlapping windows (e.g. "00-12,10-14") become separate
    # jobs that run the same hours at the same time and write the same files.
    # 'sort -u' below only removes exact duplicates, so reject overlaps here.
    if [[ -z "$HOURLY" ]]; then
        for (( h = s; h < e; h++ )); do
            [[ -z "${hour_owner[$h]:-}" ]] ||
                die "Hour window '$r' overlaps '${hour_owner[$h]}' at hour $(printf '%02d' "$h")"
            hour_owner[$h]="$r"
        done
    fi

    if [[ -n "$HOURLY" ]]; then
        for (( h = s; h < e; h++ )); do
            windows+=("$(printf '%02d %02d' "$h" $(( h + 1 )))")
        done
    else
        windows+=("$sh $eh")
    fi
done
(( ${#windows[@]} > 0 )) || die "No hour windows given (-H)"

# ---- Build job list --------------------------------------------------------
# Each line is "DATE START_HOUR END_HOUR"; duplicates from overlapping windows are removed.
jobfile="$(mktemp)"
trap 'rm -f "$jobfile"' EXIT
for d in "${dates[@]}"; do
    for w in "${windows[@]}"; do
        printf '%s %s\n' "$d" "$w"
    done
done | LC_ALL=C sort -u > "$jobfile"

total=$(wc -l < "$jobfile")
echo "Running $total job(s) with up to $JOBS in parallel (log: $JOBLOG)" >&2

# ---- Environment -----------------------------------------------------------
[[ -d "$PROJECT_DIR" ]] || die "Project directory not found: $PROJECT_DIR"
cd "$PROJECT_DIR"   # pipenv locates the Pipfile from the current directory

ENV_SCRIPT="$PROJECT_DIR/Driver_Scripts/init_iri2020_env.sh"
[[ -f "$ENV_SCRIPT" ]] || die "Env script not found: $ENV_SCRIPT"

# Third-party env scripts often reference unset variables or return non-zero,
# which would abort this script under 'set -eu'.
set +eu
# shellcheck source=/dev/null
source "$ENV_SCRIPT"
env_status=$?
set -eu
(( env_status == 0 )) || echo "Warning: $ENV_SCRIPT returned status $env_status" >&2
cd "$PROJECT_DIR"   # in case the env script changed directory

# Resolve the pipenv virtualenv once instead of running 'pipenv run' per job.
# Note: unlike 'pipenv run', this does not load a project .env file.
VENV_DIR="$(pipenv --venv 2>/dev/null)" ||
    die "No pipenv virtualenv found for $PROJECT_DIR (run 'pipenv install' there?)"
[[ -d "$VENV_DIR" ]] || die "pipenv returned an unexpected virtualenv path: $VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
PY_BIN="$(command -v "$PYTHON")" || die "Python interpreter '$PYTHON' not found"
[[ "$PY_BIN" == "$VENV_DIR"/* ]] ||
    echo "Warning: $PY_BIN is outside the virtualenv $VENV_DIR" >&2

# One thread per job for numeric libraries unless the caller says otherwise,
# so N parallel jobs don't each spawn one thread per core.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# ---- Job function ----------------------------------------------------------
# Runs in a bash subshell spawned by parallel. Arguments are passed as separate
# words, so paths containing spaces are safe.
run_one() {
    local d=$1 sh=$2 eh=$3
    local log="$LOGDIR/ionosphere_tomography_ParallelRun${LABEL:+_$LABEL}_d${d}_s${sh}Z-e${eh}Z.log"
    local cmd=("$PY_BIN" -u "$SCRIPT" --date "$d" --start-hour "$sh" --end-hour "$eh" --output all)
    if [[ -n "$DRY_RUN" ]]; then
        printf '%q ' "${cmd[@]}"
        printf '> %q 2>&1\n' "$log"
        return 0
    fi
    "${cmd[@]}" > "$log" 2>&1
}
export -f run_one
export PY_BIN SCRIPT LOGDIR DRY_RUN LABEL
export PARALLEL_SHELL=bash

# ---- Run -------------------------------------------------------------------
mkdir -p "$LOGDIR"

if [[ -z "$DRY_RUN" ]]; then
    # Prevent two runs from sharing a job log (and the same per-job log files).
    # The lock is released automatically when this script exits.
    exec 9>"$JOBLOG.lock"
    flock -n 9 || die "Another run is already using $JOBLOG (pick a different -t or -l)"

    # GNU Parallel resumes by job position, not job content, so -R is only safe
    # if the job list matches the run that wrote the job log.
    JOBLIST="$JOBLOG.jobs"
    if [[ -n "$RESUME" && -f "$JOBLOG" ]]; then
        [[ -f "$JOBLIST" ]] && cmp -s "$jobfile" "$JOBLIST" ||
            die "-R: job list differs from the run that wrote $JOBLOG (use the same -s/-e/-H/-i, or a different -t/-l)"
    fi
    cp "$jobfile" "$JOBLIST"
fi

par_args=(--nice "$NICE" -j "$JOBS" --colsep ' ' --tag -a "$jobfile")
if [[ -z "$DRY_RUN" ]]; then
    # Keep dry runs from overwriting the real job log
    par_args+=(--joblog "$JOBLOG")
    (( RETRIES > 0 )) && par_args+=(--retries "$(( RETRIES + 1 ))")
    [[ -n "$RESUME" ]] && par_args+=(--resume-failed)
fi

set +e
parallel "${par_args[@]}" run_one {1} {2} {3}
status=$?
set -e

# GNU Parallel exit status: 0 = success, 1-100 = number of failed jobs,
# 101 = more than 100 failed, anything else = parallel itself errored.
if (( status == 0 )); then
    echo "All jobs succeeded." >&2
elif (( status <= 100 )); then
    echo "$status job(s) failed. See $JOBLOG for details." >&2
elif (( status == 101 )); then
    echo "More than 100 jobs failed. See $JOBLOG for details." >&2
else
    echo "GNU parallel exited with error status $status." >&2
fi
exit "$status"
