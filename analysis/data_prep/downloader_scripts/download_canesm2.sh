
#
# This script downloads the available hourly CanESM2 NetCDF files from the
# NERSC portal, grouped by year, and saves them into local year-based folders
# under the main `CanESM2` data directory.
#
# Instead of guessing filenames, it first reads each yearly directory listing,
# pulls out the matching `PNNL_WRF.PGW.CanESM2.hourly.*.{year}.nc` files, and
# then downloads them with `wget`. Each year gets its own log file in `_logs/`
# so it is easier to track what was found and what finished downloading.
#
# Right now the script is set up for the first batch of years, 1981 to 1990,
# but the start and end years can be changed at the top.
#
# This is basically a bulk downloader for the CanESM2 hourly files, with a
# final count check at the end.
#
set -Eeuo pipefail

BASE_URL="https://portal.nersc.gov/cfs/m2637/ymsong/CanESM2/"
OUT_BASE="/data0/balaji24/data/CanESM2"

# First batch currently available
START_YEAR=1981
END_YEAR=1990

mkdir -p "$OUT_BASE"
mkdir -p "$OUT_BASE/_logs"

echo "Downloading CanESM2 data into: $OUT_BASE"
echo "Years: ${START_YEAR}-${END_YEAR}"
echo

for year in $(seq "$START_YEAR" "$END_YEAR"); do
    YEAR_URL="${BASE_URL}${year}/"
    YEAR_DIR="${OUT_BASE}/${year}"
    LOG_FILE="${OUT_BASE}/_logs/${year}.log"

    mkdir -p "$YEAR_DIR"

    echo "=================================================="
    echo "Checking directory: $YEAR_URL"
    echo "Saving into:        $YEAR_DIR"

    TMP_HTML=$(mktemp)
    trap 'rm -f "$TMP_HTML"' EXIT

    # Read directory listing first so we do not guess filenames
    if ! wget -q -O "$TMP_HTML" "$YEAR_URL"; then
        echo "Failed to access $YEAR_URL" | tee -a "$LOG_FILE"
        rm -f "$TMP_HTML"
        trap - EXIT
        continue
    fi

    mapfile -t FILES < <(
        grep -oE 'PNNL_WRF\.PGW\.CanESM2\.hourly\.[A-Za-z0-9_]+\.'"${year}"'\.nc' "$TMP_HTML" | sort -u
    )

    if [ "${#FILES[@]}" -eq 0 ]; then
        echo "No .nc files found in $YEAR_URL" | tee -a "$LOG_FILE"
        rm -f "$TMP_HTML"
        trap - EXIT
        continue
    fi

    echo "Found ${#FILES[@]} files for ${year}" | tee "$LOG_FILE"

    for fname in "${FILES[@]}"; do
        FILE_URL="${YEAR_URL}${fname}"
        echo "  -> $fname" | tee -a "$LOG_FILE"

        # -c / --continue lets you resume partial downloads
        # -nv keeps output readable
        wget -c -nv -P "$YEAR_DIR" "$FILE_URL" 2>&1 | tee -a "$LOG_FILE"
    done

    echo "Finished year ${year}" | tee -a "$LOG_FILE"
    rm -f "$TMP_HTML"
    trap - EXIT
done

echo
echo "Download finished."
echo "Verifying counts..."
find "$OUT_BASE" -type f -name "*.nc" | wc -l

echo
echo "Per-year file counts:"
for year in $(seq "$START_YEAR" "$END_YEAR"); do
    printf "%s : " "$year"
    find "${OUT_BASE}/${year}" -maxdepth 1 -type f -name "*.nc" 2>/dev/null | wc -l
done
