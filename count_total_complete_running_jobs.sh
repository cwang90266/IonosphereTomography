#!/bin/bash
# Bash script to find files modified since provided time

# Check if the correct number of arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <date_time_string> <directory>"
    echo "Example: $0 /var/log \"2026-10-20 14:30:00\""
    echo "Example: $0 . \"yesterday\""
    exit 1
fi

# Set the directory and time string defaults
TARGET_TIME="$1"
SEARCH_DIR="${2:-.}" # Defaults to current directory if not specified

# Print task to screen
echo "======================================================================="
echo "== Searching for files modified after: $TARGET_TIME in $SEARCH_DIR..."
echo "======================================================================="

# Execute find command
find "$SEARCH_DIR" -type f -iname *.log -newerBt "$TARGET_TIME" 2>/dev/null || {
    echo "WARNING: Birth time (-newerBt) not supported by your filesystem or find command."
    echo "Falling back to Modification time (-newermt)..."
    find "$SEARCH_DIR" -type f -newermt "$TARGET_TIME"
}

