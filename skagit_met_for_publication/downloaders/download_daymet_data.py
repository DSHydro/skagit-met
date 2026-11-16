#!/usr/bin/env python3
"""
Download Daymet V4R1 daily precipitation from ORNL DAAC and subset it to the Skagit basin.

For each year, downloads the full North America file (~4.5 GB), keeps only the grid
cells inside the lat/lon bounding box of the Skagit subbasins (Upper Skagit, Sauk,
Lower Skagit), saves the subset, and deletes the full download. Cells on the edge
of the kept window that fall outside the box are NaN (the Daymet grid is projected,
so lat/lon are 2D).

Data source: https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data/
Reference:   https://daymet.ornl.gov/

Requirements:
    A NASA Earthdata account (https://urs.earthdata.nasa.gov) with credentials in ~/.netrc:
        machine urs.earthdata.nasa.gov login <username> password <password>
    then `chmod 600 ~/.netrc`. Credentials are never stored in this script.

Usage:
    python download_daymet_data.py

    Set START_YEAR / END_YEAR / OUTPUT_DIR below. Years whose output already exists
    are skipped, so re-running the script resumes an interrupted run or retries failures.

Output layout:
    OUTPUT_DIR/prcp_<year>_subset.nc      e.g. daymet_new_hq/prcp_1995_subset.nc

Used by (DAYMET_DIR / DAYMET_ROOT pointing at daymet_new_hq):
    analysis/3_bulk_bias_new_version.py
    plots/plot_specific_ar_and_non_ar_events/*.py
    plots/precipitation_long_term_means/water_year_cumulative.py
    plots/spatial_bias_distribution/*.py
"""

import os
import sys

import geopandas as gpd
import requests
import xarray as xr
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
START_YEAR = 1981
END_YEAR = 2024

REGION = "na"  # North America
VAR = "prcp"   # Precipitation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # skagit_met_for_publication/
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
SUBBASINS = ["Upper Skagit", "Sauk", "Lower Skagit"]
OUTPUT_DIR = "/data0/skagit_met/data_transfer/data/daymet_new_hq"

BASE_URL = "https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data"


def get_bounding_box():
    """Return (north, south, east, west) of the Skagit subbasins, in degrees."""
    print("\nReading Skagit subbasin geometry...")
    gdf = gpd.read_file(HUC8_GEO)
    skagit_gdf = gdf[gdf["Name"].str.contains("|".join(SUBBASINS), case=False, regex=True)]

    if len(skagit_gdf) == 0:
        sys.exit(f"ERROR: Could not find subbasins: {SUBBASINS}")

    print(f"Found {len(skagit_gdf)} subbasins:")
    for _, row in skagit_gdf.iterrows():
        print(f"  - {row['Name']}")

    west, south, east, north = skagit_gdf.total_bounds  # [minx, miny, maxx, maxy]

    print("\nBounding box:")
    print(f"  North: {north:.6f}°")
    print(f"  South: {south:.6f}°")
    print(f"  East:  {east:.6f}°")
    print(f"  West:  {west:.6f}°")
    return north, south, east, west


def process_year(year, session, bbox):
    """Download one year, subset it to bbox, and save it. Raises on any failure."""
    north, south, east, west = bbox
    filename = f"daymet_v4_daily_{REGION}_{VAR}_{year}.nc"
    url = f"{BASE_URL}/{filename}"
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{filename}")
    output_file = os.path.join(OUTPUT_DIR, f"{VAR}_{year}_subset.nc")
    part_file = output_file + ".part"  # renamed to output_file only once fully written

    try:
        # Download the full-domain file with a progress bar
        response = session.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(temp_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, leave=False, desc=f"  Downloading {filename}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Fail if the connection dropped before the whole file arrived
        if total_size and os.path.getsize(temp_file) != total_size:
            raise IOError(f"incomplete download ({os.path.getsize(temp_file)} of {total_size} bytes)")

        # Subset spatially to the bounding box (lat/lon are 2D on the Daymet grid)
        with tqdm(total=1, unit=" ", leave=False, desc=f"  Subsetting {filename}") as pbar:
            ds = xr.open_dataset(temp_file)
            ds_subset = ds.where(
                (ds['lat'] >= south) & (ds['lat'] <= north) &
                (ds['lon'] >= west) & (ds['lon'] <= east),
                drop=True
            )

            # Save subsetted file to .part, then rename once complete
            ds_subset.to_netcdf(part_file, engine="netcdf4")
            ds.close()
            ds_subset.close()
            os.replace(part_file, output_file)
            pbar.update(1)
    finally:
        # Never leave the full-domain download or a half-written subset behind
        for leftover in (temp_file, part_file):
            if os.path.exists(leftover):
                os.remove(leftover)


def main():
    print("="*80)
    print("DAYMET Direct Download Script with Subsetting")
    print("="*80)

    bbox = get_bounding_box()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\n" + "="*80)
    print("Downloading and subsetting data...")
    print("="*80 + "\n")

    downloaded = 0
    skipped = 0
    failed_years = []

    # Session keeps the Earthdata login cookies across the redirect; requests reads ~/.netrc
    session = requests.Session()

    for year in tqdm(range(START_YEAR, END_YEAR + 1), desc="Processing years", unit="year"):
        # Skip years already processed (each full-domain download is ~4.5 GB)
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{VAR}_{year}_subset.nc")):
            skipped += 1
            continue

        try:
            process_year(year, session, bbox)
            downloaded += 1
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                tqdm.write(f"  ✗ Year {year}: File not found (404)")
            else:
                tqdm.write(f"  ✗ Year {year}: HTTP Error {e.response.status_code}")
            failed_years.append(year)
        except Exception as e:
            tqdm.write(f"  ✗ Year {year}: {str(e)[:60]}")
            failed_years.append(year)

    print("\n" + "="*80)
    print("Download and subset complete!")
    print("="*80)
    print(f"Successfully processed: {downloaded} files")
    print(f"Skipped (already exist): {skipped} files")
    print(f"Failed: {len(failed_years)} files")
    print(f"Total years: {END_YEAR - START_YEAR + 1}")

    if failed_years:
        print(f"\nFailed years: {failed_years}")
        print("To retry, re-run this script (finished years are skipped).")

    print(f"\nData saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
