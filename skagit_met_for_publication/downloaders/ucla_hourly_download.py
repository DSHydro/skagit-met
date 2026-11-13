"""Download hourly ERA5 reanalysis data from S3 for a specified date range.

Downloads NetCDF files from the WRF-CMIP6 S3 bucket, skipping files that already
exist locally. Includes error handling, progress reporting, and a download summary.
"""
import os
import requests
from datetime import datetime, timedelta
import time
from urllib.parse import quote

# Configuration
# start_date = datetime(1960, 9, 30, 23, 0, 0)
# end_date = datetime(1960, 12, 31, 23, 0, 0)
# download_dir = "access-cm2_r5i1p1f1_historical"
download_dir = "/data0/skagit_met/data_transfer/data/ucla_era5_d02_hourly/hourly_ar_non_ar_events"
# base_url = "https://wrf-cmip6-noversioning.s3.amazonaws.com/downscaled_products/gcm/access-cm2_r5i1p1f1_historical/hourly"
base_url = "https://wrf-cmip6-noversioning.s3.amazonaws.com/downscaled_products/reanalysis/era5/hourly"

# Date ranges for AR and non-AR events from plots
date_ranges = [
    # AR Events
    (datetime(1990, 11, 20, 23, 0, 0), datetime(1990, 11, 29, 23, 0, 0)),  # 1990-11-24 AR=5.0
    (datetime(1995, 11, 25, 23, 0, 0), datetime(1995, 12, 5, 23, 0, 0)),   # 1995-11-30 AR=4.0  1995-11-29 AR=4.0
    (datetime(1990, 11, 6, 23, 0, 0), datetime(1990, 11, 18, 23, 0, 0)),   # 1990-11-13      1990-11-10 AR=4.0
    (datetime(2006, 11, 3, 23, 0, 0), datetime(2006, 11, 12, 23, 0, 0)),   # 2006-11-07 AR=5.0
    (datetime(2003, 10, 13, 23, 0, 0), datetime(2003, 10, 28, 23, 0, 0)),  # 2003-10-21 AR=5.0
    (datetime(2021, 11, 12, 23, 0, 0), datetime(2021, 11, 20, 23, 0, 0)),  # 2021-11-15 AR=4.0
    # non-AR Events
    # # (datetime(2021, 11, 12, 23, 0, 0), datetime(2021, 11, 21, 23, 0, 0)),  # 2021-11-16 AR=0.0
    # (datetime(1995, 11, 28, 23, 0, 0), datetime(1995, 12, 7, 23, 0, 0)),   # 1995-12-02 AR=0.0
    # # (datetime(1990, 11, 10, 23, 0, 0), datetime(1990, 11, 19, 23, 0, 0)),  # 1990-11-14 AR=0.0
    # (datetime(2011, 1, 14, 23, 0, 0), datetime(2011, 1, 23, 23, 0, 0)),    # 2011-01-18 AR=0.0
    # (datetime(2015, 11, 10, 23, 0, 0), datetime(2015, 11, 19, 23, 0, 0)),  # 2015-11-14 AR=0.0
    # (datetime(2007, 3, 9, 23, 0, 0), datetime(2007, 3, 18, 23, 0, 0)),    # 2007-03-13 AR=0.0
    # # (datetime(2021, 11, 29, 23, 0, 0), datetime(2021, 12, 9, 23, 0, 0)),    # 2021-12-03 AR=0.0
    # (datetime(2010, 12, 10, 23, 0, 0), datetime(2010, 12, 20, 23, 0, 0)),    # 2010-12-14 AR=0.0
    # (datetime(2009, 11, 23, 23, 0, 0), datetime(2009, 12, 3, 23, 0, 0)),    # 2010-12-14 AR=0.0
    # non-AR Events (5 day window taking max)
    # (datetime(1985, 6, 4, 23, 0, 0), datetime(1985, 6, 13, 23, 0, 0)),    # 1985-06-08 AR=0.0
    # (datetime(2020, 5, 28, 23, 0, 0), datetime(2020, 6, 6, 23, 0, 0)),    # 2020-06-01 AR=0.0
    # (datetime(2022, 7, 1, 23, 0, 0), datetime(2022, 7, 10, 23, 0, 0)),    # 2022-07-05 AR=0.0
    # (datetime(1997, 6, 20, 23, 0, 0), datetime(1997, 6, 29, 23, 0, 0)),   # 1997-06-24 AR=0.0
    # (datetime(1982, 6, 30, 23, 0, 0), datetime(1982, 7, 9, 23, 0, 0)),    # 1982-07-04 AR=0.0
    # (datetime(2002, 6, 15, 23, 0, 0), datetime(2002, 6, 24, 23, 0, 0)),   # 2002-06-19 AR=0.0
    # non-AR Events (5 day window taking min)
    (datetime(1996, 4, 21, 23, 0, 0), datetime(1996, 4, 30, 23, 0, 0)),   # 1996-04-25
    (datetime(2004, 8, 22, 23, 0, 0), datetime(2004, 8, 31, 23, 0, 0)),   # 2004-08-26
    (datetime(2017, 12, 26, 23, 0, 0), datetime(2018, 1, 4, 23, 0, 0)),   # 2017-12-30
    (datetime(1997, 9, 13, 23, 0, 0), datetime(1997, 9, 22, 23, 0, 0)),   # 1997-09-17
    (datetime(2014, 2, 13, 23, 0, 0), datetime(2014, 2, 22, 23, 0, 0)),   # 2014-02-17
    (datetime(1982, 1, 28, 23, 0, 0), datetime(1982, 2, 6, 23, 0, 0)),    # 1982-02-01
]

# Create download directory
os.makedirs(download_dir, exist_ok=True)

# Generate list of dates
file_count = 0
success_count = 0
failed_count = 0
failed_files = []

print("=" * 60)
print(f"Starting download for {len(date_ranges)} event date ranges")
print("=" * 60)

def get_water_year_dir(date):
    """Get calendar year directory for UCLA S3 bucket.
    Data is organized by calendar year (not water year)."""
    return date.year

for start_date, end_date in date_ranges:
    print(f"\nProcessing: {start_date.date()} to {end_date.date()}")
    current_date = start_date

    while current_date <= end_date:
        # Format: YYYY-MM-DD_HH:00:00
        date_str = current_date.strftime("%Y-%m-%d_%H:00:00")
        year = str(get_water_year_dir(current_date))

        # Construct filename and URL
        filename = f"auxhist_d01_{date_str}.nc"
        filepath = os.path.join(download_dir, filename)
        # URL-encode the filename to handle colons
        encoded_filename = quote(f"auxhist_d01_{date_str}", safe="")
        url = f"{base_url}/{year}/d02/{encoded_filename}"

        file_count += 1

        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"[{file_count}] Already exists: {date_str}")
            success_count += 1
        else:
            try:
                print(f"[{file_count}] Downloading: {date_str}...", end=" ", flush=True)

                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Write to file
                with open(filepath, 'wb') as f:
                    f.write(response.content)

                file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
                print(f"✓ ({file_size:.2f} MB)")
                success_count += 1

            except requests.exceptions.RequestException as e:
                print(f"✗ ERROR: {str(e)}")
                failed_count += 1
                failed_files.append(date_str)

                # Remove partial file if it exists
                if os.path.exists(filepath):
                    os.remove(filepath)

            # Small delay to be respectful to server
            time.sleep(0.5)

        # Move to next hour
        current_date += timedelta(hours=1)

# Summary
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
print(f"Total files:     {file_count}")
print(f"Successful:      {success_count}")
print(f"Failed:          {failed_count}")

if failed_files:
    print(f"\nFailed downloads:")
    for f in failed_files:
        print(f"  - {f}")

print("=" * 60)