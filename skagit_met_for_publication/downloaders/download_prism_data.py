#!/usr/bin/env python3
"""
Download daily PRISM precipitation (AN81d, 4 km, CONUS) and organize it into decade folders.

Data source: https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily/<year>/
Each day is the original PRISM zip (GeoTIFF + metadata), kept unmodified.

Usage:
    python download_prism_data.py

    Set START_DATE / END_DATE / OUTPUT_DIR below. Files already on disk are skipped,
    so re-running the script resumes an interrupted download or retries failures.

Output layout:
    OUTPUT_DIR/<decade>/prism_ppt_us_25m_YYYYMMDD.zip
    e.g. prism_new_hq/1991-2000/prism_ppt_us_25m_19951130.zip
    ("25m" is PRISM's name for the 2.5 arc-minute, i.e. 4 km, grid)

Used by (PRISM_DIR / PRISM_ROOT pointing at prism_new_hq):
    analysis/3_bulk_bias_new_version.py
    plots/plot_specific_ar_and_non_ar_events/*.py
    plots/precipitation_long_term_means/water_year_cumulative.py
    plots/spatial_bias_distribution/*.py
    plots/top_50_ar_vs_non-AR/ar_vs_non_ar_panel_figure.py
"""

import os
import pandas as pd
import urllib.request
import urllib.error
import time

# Configuration
BASE_URL = "https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily"
OUTPUT_DIR = "/data0/skagit_met/data_transfer/data/prism_new_hq"

START_DATE = "1981-01-01"
END_DATE = "2024-12-31"

# Decade folders (year range -> folder name), matching the layout the analysis scripts read
DECADE_MAPPING = {
    (1981, 1990): "1981-1990",
    (1991, 2000): "1991-2000",
    (2001, 2010): "2001-2010",
    (2011, 2020): "2011-2020",
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
    """Generate all daily dates from START_DATE to END_DATE"""
    return pd.date_range(start=START_DATE, end=END_DATE, freq='D')

def download_file(date, output_path, max_retries=3):
    """Download a single PRISM file.

    Writes to '<output_path>.part' and renames it to output_path only once the
    transfer completes, so an interrupted download never leaves a truncated .zip
    that a later run would skip as already downloaded.
    """
    date_str = date.strftime('%Y%m%d')
    file_url = f"{BASE_URL}/{date.year}/prism_ppt_us_25m_{date_str}.zip"
    part_path = output_path + ".part"

    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"  Downloading: {date_str}...", end=' ', flush=True)
            urllib.request.urlretrieve(file_url, part_path)
            os.replace(part_path, output_path)
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
                break
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"✗ (Error: {str(e)[:30]}, retry {retry_count}/{max_retries-1})")
                time.sleep(2)
            else:
                print(f"✗ (Error: {str(e)[:30]}, giving up)")
                break

    # Remove any leftover partial file from the failed attempts
    if os.path.exists(part_path):
        os.remove(part_path)
    return False

def main():
    print(f"PRISM Data Downloader ({START_DATE} to {END_DATE})")
    print("="*80)

    # Create directories
    print("\nCreating decade directories...")
    create_directories()

    # Generate all dates
    print(f"\nGenerating dates from {START_DATE} to {END_DATE}...")
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

        print("\nTo retry failed downloads, re-run this script (existing files are skipped).")

if __name__ == "__main__":
    main()