#!/bin/bash

# Download PRISM precipitation data from 1981-2020 and organize into decade folders
# Source: https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily/

BASE_URL="https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily"
OUTPUT_DIR="/data0/skagit_met/data_transfer/data/PRISM"

# Create decade directories
mkdir -p "$OUTPUT_DIR/1981-1990"
mkdir -p "$OUTPUT_DIR/1991-2000"
mkdir -p "$OUTPUT_DIR/2001-2010"
mkdir -p "$OUTPUT_DIR/2011-2020"

echo "PRISM Data Downloader (1981-2020)"
echo "=========================================="

# Function to get decade folder
get_decade() {
    local year=$1
    if [ $year -ge 1981 ] && [ $year -le 1990 ]; then
        echo "1981-1990"
    elif [ $year -ge 1991 ] && [ $year -le 2000 ]; then
        echo "1991-2000"
    elif [ $year -ge 2001 ] && [ $year -le 2010 ]; then
        echo "2001-2010"
    elif [ $year -ge 2011 ] && [ $year -le 2020 ]; then
        echo "2011-2020"
    fi
}

# Download files using a date range
# Using Python to generate dates to properly handle leap years
python3 << 'PYTHON_EOF'
import pandas as pd
import subprocess
import os

base_url = "https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily"
output_dir = "/data0/skagit_met/data_transfer/data/PRISM"

# Generate all dates from 1981-01-01 to 2020-12-31
all_dates = pd.date_range(start='1981-01-01', end='2020-12-31', freq='D')

print(f"Starting download of {len(all_dates)} files...")
print("Note: This may take several hours. The script will resume if interrupted.")
print("=" * 80)

decade_mapping = {
    (1981, 1990): "1981-1990",
    (1991, 2000): "1991-2000",
    (2001, 2010): "2001-2010",
    (2011, 2020): "2011-2020",
}

downloaded = 0
failed = 0
skipped = 0

for idx, date in enumerate(all_dates, 1):
    year = date.year
    date_str = date.strftime('%Y%m%d')

    # Find decade folder
    decade = None
    for (start, end), folder in decade_mapping.items():
        if start <= year <= end:
            decade = folder
            break

    if not decade:
        continue

    output_path = f"{output_dir}/{decade}/prism_ppt_us_25m_{date_str}.zip"

    # Skip if already exists
    if os.path.exists(output_path):
        skipped += 1
        if idx % 500 == 0:
            print(f"[{idx}/{len(all_dates)}] Progress: {downloaded} downloaded, {skipped} skipped, {failed} failed")
        continue

    # Download file
    file_url = f"{base_url}/{year}/prism_ppt_us_25m_{date_str}.zip"

    try:
        # Use wget with retry logic
        cmd = [
            'wget',
            '--quiet',
            '--tries=3',
            '--timeout=10',
            '-O', output_path,
            file_url
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode == 0:
            downloaded += 1
            if idx % 500 == 0:
                print(f"[{idx}/{len(all_dates)}] ✓ Downloaded: {date_str}")
        else:
            failed += 1
            if failed <= 10:  # Only print first 10 failures
                print(f"[{idx}/{len(all_dates)}] ✗ Failed: {date_str}")

    except Exception as e:
        failed += 1
        if failed <= 10:
            print(f"[{idx}/{len(all_dates)}] ✗ Error: {date_str} - {str(e)[:50]}")

    # Progress update every 500 files
    if idx % 500 == 0:
        print(f"[{idx}/{len(all_dates)}] Progress: {downloaded} downloaded, {skipped} skipped, {failed} failed")

print("\n" + "=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print(f"Downloaded: {downloaded} files")
print(f"Skipped (already exist): {skipped} files")
print(f"Failed: {failed} files")
print(f"Total processed: {len(all_dates)} files")

PYTHON_EOF

echo ""
echo "Download script completed!"
