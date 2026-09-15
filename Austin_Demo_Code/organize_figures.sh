#!/bin/bash
# Organize existing figures into proper subfolders with window/bin hierarchy

SAVE_DIR="${1:-.}"  # Default to current directory if not specified

echo "Organizing figures in: $SAVE_DIR"

# Create top-level subfolders
mkdir -p "$SAVE_DIR/per_window" "$SAVE_DIR/cross_window" "$SAVE_DIR/summaries"

# Helper function to move file to per_window/HHMM/bin_COUNT/
move_per_window_figure() {
    local file=$1
    local hhmm=$2
    local bin_count=$3

    if [ -f "$SAVE_DIR/$file" ]; then
        bin_folder="bin_all"
        if [ "$bin_count" != "all" ]; then
            bin_folder="bin_$bin_count"
        fi
        mkdir -p "$SAVE_DIR/per_window/$hhmm/$bin_folder"
        mv "$SAVE_DIR/$file" "$SAVE_DIR/per_window/$hhmm/$bin_folder/"
    fi
}

# Extract HHMM and bin_count from a filename
# Pattern: filename_YYYY_DOY_HHMM[_bin_COUNT][_...].png
extract_hhmm_and_bin() {
    local filename=$1
    # Remove path and extension
    local base=$(basename "$filename" .png)

    # Try to extract HHMM and bin_count using grep and sed
    if [[ $base =~ _([0-9]{4})_bin_([0-9]+) ]]; then
        # Has bin count
        hhmm="${BASH_REMATCH[1]}"
        bin_count="${BASH_REMATCH[2]}"
    elif [[ $base =~ _([0-9]{4})_bin_all ]]; then
        # Has bin_all
        hhmm="${BASH_REMATCH[1]}"
        bin_count="all"
    elif [[ $base =~ _([0-9]{4})$ ]]; then
        # Just HHMM at end (old format, no bin)
        hhmm="${BASH_REMATCH[1]}"
        bin_count="all"
    elif [[ $base =~ _([0-9]{4})_ ]]; then
        # HHMM in middle (e.g., with other suffixes)
        hhmm="${BASH_REMATCH[1]}"
        bin_count="all"
    else
        return 1  # Could not extract
    fi
}

echo "Moving per-window figures..."

# Process all PNG files in SAVE_DIR
for file in "$SAVE_DIR"/*.png; do
    [ -f "$file" ] || continue
    filename=$(basename "$file")

    # Skip if already in a subfolder
    if [[ "$file" == "$SAVE_DIR/per_window/"* ]] || \
       [[ "$file" == "$SAVE_DIR/cross_window/"* ]]; then
        continue
    fi

    # Check if it's a cross-window figure (these don't have HHMM suffixes)
    if [[ "$filename" =~ convergence_vs_measurement_count ]] || \
       [[ "$filename" =~ edp_site_rmse_across_windows ]] || \
       [[ "$filename" =~ edp_regional_rmse_across_windows ]] || \
       [[ "$filename" =~ station_.*_edp_errors ]] || \
       [[ "$filename" =~ hf_reflection_errors ]]; then
        mkdir -p "$SAVE_DIR/cross_window"
        mv "$file" "$SAVE_DIR/cross_window/"
        continue
    fi

    # Try to extract HHMM and bin from filename
    if extract_hhmm_and_bin "$filename"; then
        bin_folder="bin_all"
        [ "$bin_count" != "all" ] && bin_folder="bin_$bin_count"

        mkdir -p "$SAVE_DIR/per_window/$hhmm/$bin_folder"
        mv "$file" "$SAVE_DIR/per_window/$hhmm/$bin_folder/"
    else
        # Couldn't parse, put in per_window as-is (fallback)
        mkdir -p "$SAVE_DIR/per_window/unknown"
        mv "$file" "$SAVE_DIR/per_window/unknown/" 2>/dev/null || true
    fi
done

echo "Moving summary files..."
mkdir -p "$SAVE_DIR/summaries"
mv "$SAVE_DIR"/summary_*.csv "$SAVE_DIR/summaries/" 2>/dev/null || true
mv "$SAVE_DIR"/cross_window_summary_*.csv "$SAVE_DIR/summaries/" 2>/dev/null || true
mv "$SAVE_DIR"/cross_window_summary_*.txt "$SAVE_DIR/summaries/" 2>/dev/null || true

echo "Done! Figures organized into:"
echo "  - per_window/HHMM/bin_COUNT/"
echo "  - cross_window/"
echo "  - summaries/"
echo ""
echo "Summary:"
echo "  Per-window folders: $(find "$SAVE_DIR/per_window" -mindepth 2 -type d 2>/dev/null | wc -l)"
echo "  Cross-window files: $(ls "$SAVE_DIR/cross_window"/*.png 2>/dev/null | wc -l) PNG"
echo "  Summary files: $(ls "$SAVE_DIR/summaries"/* 2>/dev/null | wc -l) files"
