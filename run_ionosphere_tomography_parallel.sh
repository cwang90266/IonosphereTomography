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

# ---- Defaults --------------------------------------------------------------
NICE=19
LOGDIR="parallel_jobs_logs"
JOBS=4
SCRIPT="/home/knelson/src/IonosphereTomography/run_hourly_all4styles_boost_compare.py"
PYTHON="${PYTHON:-python3}"
HOURS="00-24"
JOBLOG="ionosphere_tomography_parallel_jobs.log"
DRY_RUN=""
HOURLY=""
START_DATE=""
END_DATE=""

usage() {
    cat <<EOF
Usage: $(basename "$0") -s START_DATE [options]

Required:
  -s DATE     Start date (YYYY-MM-DD)

Options:
  -e DATE     End date, inclusive (YYYY-MM-DD). Defaults to start date.
  -H RANGES   Comma-separated hour windows START-END (HH-HH), END exclusive.
              START is 00-23, END is 01-24 and must be greater than START.
              Default: "$HOURS" (the full day)
              e.g. "00-06,06-12,12-18,18-24"
  -j N        Max number of parallel processes. Default: $JOBS
  -p PATH     Python script to run. Default: $SCRIPT
  -i          Hourly mode: split each window into one-hour jobs
              (e.g. "21-24" runs 21-22, 22-23 and 23-24)
  -l FILE     GNU Parallel job log file. Default: $JOBLOG
  -L PATH     Directory to place logs from individual python script jobs
  -N N        Niceness number (max = 19, negative requires sudo access)
  -n          Dry run: print the commands without executing them
  -h          Show this help

Environment:
  PYTHON      Python interpreter to use. Default: python3
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

# ---- Parse arguments -------------------------------------------------------
while getopts ":s:e:H:j:p:l:inNLh" opt; do
    case "$opt" in
        s) START_DATE="$OPTARG" ;;
        e) END_DATE="$OPTARG" ;;
        H) HOURS="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        p) SCRIPT="$OPTARG" ;;
        l) JOBLOG="$OPTARG" ;;
        i) HOURLY=1 ;;
        n) DRY_RUN="--dry-run" ;;
        N) NICE="$OPTARG" ;;
        L) LOGDIR="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) die "Option -$OPTARG requires a value" ;;
        \?) die "Unknown option -$OPTARG (use -h for help)" ;;
    esac
done

# ---- Validation ------------------------------------------------------------
command -v parallel >/dev/null 2>&1 || die "GNU parallel is not installed"
[[ -f "$SCRIPT" ]] || die "Script not found: $SCRIPT"
[[ -n "$START_DATE" ]] || { usage; exit 1; }
END_DATE="${END_DATE:-$START_DATE}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "-j must be a positive integer"
[[ "$NICE" =~ ^([0-9]|1[0-9])$ ]] || die "-N must be 0-19"

validate_date() {
    # Checks both the format and that it's a real calendar date (e.g. rejects 2026-02-30)
    [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] &&
    [[ "$(date -d "$1" +%F 2>/dev/null)" == "$1" ]]
}

validate_date "$START_DATE" || die "Invalid start date: $START_DATE"
validate_date "$END_DATE"   || die "Invalid end date: $END_DATE"
[[ ! "$START_DATE" > "$END_DATE" ]] || die "Start date is after end date"

# ---- Build date list -------------------------------------------------------
dates=()
d="$START_DATE"
while [[ ! "$d" > "$END_DATE" ]]; do
    dates+=("$d")
    d="$(date -d "$d + 1 day" +%F)"
done

# ---- Build hour-window list ------------------------------------------------
# Each entry is "START_HOUR END_HOUR".
hour_re='^([01][0-9]|2[0-4])$'
windows=()
IFS=',' read -ra raw_ranges <<< "$HOURS"
for r in "${raw_ranges[@]}"; do
    IFS='-' read -r sh eh <<< "$r"
    [[ "$sh" =~ $hour_re && "$eh" =~ $hour_re ]] || die "Invalid hour window: '$r' (expect HH-HH 00-24)"
    s=$(( 10#$sh ))
    e=$(( 10#$eh ))
    (( s <= 23 )) || die "Start hour must be 00-23 in window: '$r'"
    (( e > s ))   || die "End hour must be greater than start hour in window: '$r'"

    if [[ -n "$HOURLY" ]]; then
        for (( h = s; h < e; h++ )); do
            windows+=("$(printf '%02d %02d' "$h" $(( h + 1 )))")
        done
    else
        windows+=("$sh $eh")
    fi
done

# ---- Build job list --------------------------------------------------------
# Each line is "DATE START_HOUR END_HOUR"; duplicates from overlapping windows are removed.
jobfile="$(mktemp)"
trap 'rm -f "$jobfile"' EXIT
for d in "${dates[@]}"; do
    for w in "${windows[@]}"; do
        printf '%s %s\n' "$d" "$w"
    done
done | sort -u > "$jobfile"

total=$(wc -l < "$jobfile")
echo "Running $total job(s) with up to $JOBS in parallel (log: $JOBLOG)" >&2

# ---- Run -------------------------------------------------------------------
mkdir -p "$LOGDIR"
source Driver_Scripts/init_iri2020_env.sh
set +e
parallel $DRY_RUN \
    --nice $NICE \
    -j "$JOBS" \
    --colsep ' ' \
    --joblog "$JOBLOG" \
    --tag \
    -a "$jobfile" \
    pipenv run "$PYTHON" -u "$SCRIPT" --date {1} --start-hour {2} --end-hour {3} --output "all" \
    '>' "$LOGDIR/ionosphere_tomography_ParallelRun_d{1}_s{2}HH-e{3}HH.log" '2>&1'
status=$?
set -e

# GNU Parallel's exit status is the number of failed jobs (capped at 101+ for special cases)
if (( status == 0 )); then
    echo "All jobs succeeded." >&2
else
    echo "$status job(s) failed. See $JOBLOG for details." >&2
fi
exit "$status"
