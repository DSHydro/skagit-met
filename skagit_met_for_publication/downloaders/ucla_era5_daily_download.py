#!/usr/bin/env python3
"""Download UCLA ERA5-WRF daily (post-processed) data from the WRF-CMIP6 S3 bucket.

UCLA's ERA5-driven WRF downscaling (9 km, domain d02) is published as one file per
variable per water year, already aggregated to daily values and covering the full
d02 domain (no clipping is done here; the analysis applies the basin mask itself).
Files are copied as-is, so they match the bucket byte for byte.

Data source (public, no login):
    https://wrf-cmip6-noversioning.s3.amazonaws.com/index.html#downscaled_products/reanalysis/era5/postprocess/d02/
    https://wrf-cmip6-noversioning.s3.amazonaws.com/index.html#downscaled_products/wrf_coordinates/
Reference: https://dept.atmos.ucla.edu/alexhall/downscaling-cmip6

File naming: <var>.daily.era5.d02.<YYYY>.nc holds water year YYYY, i.e. Sep 1 YYYY
to Aug 31 YYYY+1 (e.g. prec.daily.era5.d02.2021.nc = 2021-09-01 .. 2022-08-31).

Coverage: water years 1950-2024. 

Usage:
    python ucla_era5_daily_download.py

    Set VARIABLES / START_YEAR / END_YEAR / OUTPUT_DIR below. Files that already exist
    are skipped, so re-running resumes an interrupted run or retries failures.
    Each file (~130 MB) is written to a .part file and renamed only once its size
    matches what the server reported.

Output layout:
    OUTPUT_DIR/<var>/<var>.daily.era5.d02.<YYYY>.nc   e.g. prec/prec.daily.era5.d02.2021.nc
    OUTPUT_DIR/static/wrfinput_d02_coord.nc           2D lat/lon of the d02 grid

Used by (UCLA_PREC_DIR / UCLA_STATIC / VAULT_DIR pointing at ucla_era5_d02_daily):
    analysis/3_bulk_bias_new_version.py
    plots/plot_specific_ar_and_non_ar_events/*.py
    plots/precipitation_long_term_means/water_year_cumulative.py
    plots/spatial_bias_distribution/*.py
"""
import os
import time

import requests

# Configuration
VARIABLES = ["prec"]  # daily precipitation (mm/d); add "t2" for 2 m temperature
START_YEAR = 1950     # first water year
END_YEAR = 2024       # last water year on the bucket (2024 is partial: Sep 2024-Mar 2025)

OUTPUT_DIR = "/data0/skagit_met/data_transfer/data/ucla_era5_d02_daily"

BUCKET_URL = "https://wrf-cmip6-noversioning.s3.amazonaws.com/downscaled_products"
DAILY_URL = f"{BUCKET_URL}/reanalysis/era5/postprocess/d02"
COORD_URL = f"{BUCKET_URL}/wrf_coordinates/wrfinput_d02_coord.nc"


def download_file(session, url, filepath):
    """Download url to filepath via a .part file. Raises on any failure."""
    part_path = filepath + ".part"  # renamed to filepath only once fully downloaded
    try:
        with session.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

        # Fail if the connection dropped before the whole file arrived
        size = os.path.getsize(part_path)
        if total_size and size != total_size:
            raise IOError(f"incomplete download ({size} of {total_size} bytes)")

        os.replace(part_path, filepath)
        return size
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)


def main():
    # (url, local path) for every file to fetch: the grid coordinates, then each var/year
    jobs = [(COORD_URL, os.path.join(OUTPUT_DIR, "static", "wrfinput_d02_coord.nc"))]
    for var in VARIABLES:
        for year in range(START_YEAR, END_YEAR + 1):
            filename = f"{var}.daily.era5.d02.{year}.nc"
            jobs.append((f"{DAILY_URL}/{filename}", os.path.join(OUTPUT_DIR, var, filename)))

    print("=" * 60)
    print(f"UCLA ERA5-WRF daily: {VARIABLES}, water years {START_YEAR}-{END_YEAR}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    success_count = 0
    skipped_count = 0
    failed_files = []
    session = requests.Session()

    for i, (url, filepath) in enumerate(jobs, start=1):
        name = os.path.relpath(filepath, OUTPUT_DIR)

        # Skip if already downloaded
        if os.path.exists(filepath):
            skipped_count += 1
            continue

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"[{i}/{len(jobs)}] Downloading: {name}...", end=" ", flush=True)
        try:
            size = download_file(session, url, filepath)
            print(f"✓ ({size / 1024**2:.1f} MB)")
            success_count += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed_files.append(name)

        # Small delay to be respectful to server
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total files:     {len(jobs)}")
    print(f"Downloaded:      {success_count}")
    print(f"Already existed: {skipped_count}")
    print(f"Failed:          {len(failed_files)}")

    if failed_files:
        print("\nFailed downloads (re-run to retry):")
        for f in failed_files:
            print(f"  - {f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
