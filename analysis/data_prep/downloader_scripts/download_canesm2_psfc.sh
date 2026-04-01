#
# This script downloads the hourly CanESM2 PSFC NetCDF files from the NERSC
# portal and saves them into the local `CanESM2` data folder.
#
# It first reads the directory listing from the remote site, pulls out the PSFC
# files for 1980 through 2020, and then downloads each one with `wget`.
# I also keep a log file in `_logs/download.log` so it is easier to see what
# finished, what failed, and how many files were picked up.
#
# This is basically just a batch downloader for the PSFC variable, not any
# extra processing beyond the final file-count check.
#
set -Eeuo pipefail

BASE_URL="https://portal.nersc.gov/cfs/m2637/ymsong/CanESM2/"
OUT_BASE="/data0/balaji24/data/CanESM2"

mkdir -p "$OUT_BASE"
mkdir -p "$OUT_BASE/_logs"

LOG_FILE="$OUT_BASE/_logs/download.log"
TMP_HTML=$(mktemp)
trap 'rm -f "$TMP_HTML"' EXIT

echo "Downloading CanESM2 PSFC data into: $OUT_BASE"
echo

if ! wget -q -O "$TMP_HTML" "$BASE_URL"; then
    echo "Failed to access $BASE_URL" | tee -a "$LOG_FILE"
    exit 1
fi

mapfile -t FILES < <(
    grep -oE 'PNNL_WRF\.PGW\.CanESM2\.hourly\.PSFC\.(19[8-9][0-9]|20[0-1][0-9]|2020)\.nc' "$TMP_HTML" | sort -u
)

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No PSFC files found." | tee -a "$LOG_FILE"
    exit 1
fi

echo "Found ${#FILES[@]} files" | tee "$LOG_FILE"

for fname in "${FILES[@]}"; do
    echo "-> $fname" | tee -a "$LOG_FILE"
    wget -c -nv -P "$OUT_BASE" "${BASE_URL}${fname}" 2>&1 | tee -a "$LOG_FILE"
done

echo
echo "Download finished." | tee -a "$LOG_FILE"
echo "Verifying count:"
find "$OUT_BASE" -maxdepth 1 -type f -name 'PNNL_WRF.PGW.CanESM2.hourly.PSFC.*.nc' | wc -l
