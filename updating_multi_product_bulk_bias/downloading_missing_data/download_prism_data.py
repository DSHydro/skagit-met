#!/usr/bin/env python3
"""
Download PRISM precipitation data from 1981-2024 and organize into decade folders.
Data source: https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily/
"""

import os
import pandas as pd
import urllib.request
import urllib.error
from datetime import datetime
import time

# Configuration
BASE_URL = "https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily"
OUTPUT_DIR = "/data0/skagit_met/data_transfer/data/prism_new_hq"

# Define decade folders
# ORIGINAL (1981-2020):
# DECADE_MAPPING = {
#     (1981, 1990): "1981-1990",
#     (1991, 2000): "1991-2000",
#     (2001, 2010): "2001-2010",
#     (2011, 2020): "2011-2020",
# }

# For 2021-2024 data only:
DECADE_MAPPING = {
    (2021, 2024): "2021-2024",
}

def get_decade_folder(year):
    """Get the decade folder for a given year"""
    for (start, end), folder in DECADE_MAPPING.items():
        if start <= year <= end:
            return folder
    return None

def create_directories():
    """Create decade directories if they don't exist"""
    for decade_folder in set(DECADE_MAPPING.values()):
        folder_path = os.path.join(OUTPUT_DIR, decade_folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"✓ Directory ready: {folder_path}")

def generate_all_dates():
    """Generate all dates from 2021-01-01 to 2024-12-31, accounting for leap years"""
    # ORIGINAL (1981-01-01 to 2020-12-31):
    # start_date = pd.Timestamp('1981-01-01')
    # end_date = pd.Timestamp('2020-12-31')

    # For 2021-2024 data:
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2024-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return dates

def download_file(date, output_path, max_retries=3):
    """Download a single PRISM file"""
    date_str = date.strftime('%Y%m%d')
    file_url = f"{BASE_URL}/{date.year}/prism_ppt_us_25m_{date_str}.zip"

    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"  Downloading: {date_str}...", end=' ', flush=True)
            urllib.request.urlretrieve(file_url, output_path)
            file_size = os.path.getsize(output_path) / 1024 / 1024  # Size in MB
            print(f"✓ ({file_size:.1f} MB)")
            return True
        except urllib.error.HTTPError as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"✗ (HTTP {e.code}, retry {retry_count}/{max_retries-1})")
                time.sleep(2)
            else:
                print(f"✗ (HTTP {e.code}, giving up)")
                return False
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"✗ (Error: {str(e)[:30]}, retry {retry_count}/{max_retries-1})")
                time.sleep(2)
            else:
                print(f"✗ (Error: {str(e)[:30]}, giving up)")
                return False

    return False

def main():
    print("PRISM Data Downloader (2021-2024)")
    # ORIGINAL: print("PRISM Data Downloader (1981-2020)")
    print("="*80)

    # Create directories
    print("\nCreating decade directories...")
    create_directories()

    # Generate all dates
    print("\nGenerating dates from 2021-01-01 to 2024-12-31...")
    # ORIGINAL: print("\nGenerating dates from 1981-01-01 to 2020-12-31...")
    all_dates = generate_all_dates()
    print(f"Total dates to download: {len(all_dates)}")

    # Download files
    print("\n" + "="*80)
    print("Starting downloads...")
    print("="*80)

    downloaded = 0
    failed = 0
    skipped = 0
    failed_dates = []

    for idx, date in enumerate(all_dates, 1):
        decade_folder = get_decade_folder(date.year)
        if not decade_folder:
            print(f"✗ No decade folder for year {date.year}")
            skipped += 1
            continue

        output_dir = os.path.join(OUTPUT_DIR, decade_folder)
        date_str = date.strftime('%Y%m%d')
        output_file = os.path.join(output_dir, f"prism_ppt_us_25m_{date_str}.zip")

        # Show progress
        if idx % 100 == 0 or idx == 1:
            print(f"\n[{idx}/{len(all_dates)}] {date.strftime('%Y-%m-%d')}:")

        # Skip if file already exists
        if os.path.exists(output_file):
            skipped += 1
            continue

        # Download file
        if download_file(date, output_file):
            downloaded += 1
        else:
            failed += 1
            failed_dates.append(date.strftime('%Y-%m-%d'))

        # Be nice to the server - add small delay
        time.sleep(0.5)

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Downloaded: {downloaded} files")
    print(f"Skipped (already exist): {skipped} files")
    print(f"Failed: {failed} files")
    print(f"Total processed: {len(all_dates)} files")

    if failed_dates:
        print(f"\nFailed dates:")
        for date_str in failed_dates[:20]:  # Show first 20
            print(f"  - {date_str}")
        if len(failed_dates) > 20:
            print(f"  ... and {len(failed_dates) - 20} more")

        print(f"\nTo retry failed downloads, run this command:")
        print(f"python3 {__file__} --retry-failed")

if __name__ == "__main__":
    import sys

    if "--help" in sys.argv:
        print(__doc__)
        sys.exit(0)

    main()