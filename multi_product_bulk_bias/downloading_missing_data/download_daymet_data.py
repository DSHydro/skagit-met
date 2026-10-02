#!/usr/bin/env python3
"""
Download Daymet gridded daily precipitation data from ORNL DAAC.

Spatially subsets data to the Skagit subbasins (Upper Skagit, Sauk, Lower Skagit).
Downloads data from 1981 to 2020 and saves to /data0/skagit_met/data_transfer/data/daymet_new_hq

Data source: https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/
Reference: https://daymet.ornl.gov/
"""

import os
import geopandas as gpd
import xarray as xr
import rioxarray
import requests
from tqdm import tqdm

# ============================================================================
# VARIABLES - Temporal subset
# ============================================================================
startyr = 2021 #1981
endyr = 2024

# ============================================================================
# VARIABLES - Region and variable
# ============================================================================
region = "na"  # North America
var = "prcp"   # Precipitation

# ============================================================================
# VARIABLES - Spatial subset - Get bounding box from Skagit subbasins
# ============================================================================
BASE_DIR = "/data0/hernanqd/plots_code/skagit-met"
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
OUTPUT_DIR = "/data0/skagit_met/data_transfer/data/daymet_new_hq"

# ============================================================================
# Download Configuration
# ============================================================================
BASE_URL = "https://data.ornldaac.earthdata.nasa.gov/protected/daymet/Daymet_Daily_V4R1/data"

# Earthdata credentials
EDL_USERNAME = "hernanqd"
EDL_PASSWORD = "Uwashington13191017."

print("="*80)
print("DAYMET Direct Download Script with Subsetting")
print("="*80)

# ============================================================================
# Read Skagit subbasin geometry
# ============================================================================
print("\nReading Skagit subbasin geometry...")
gdf = gpd.read_file(HUC8_GEO)

# Filter for the three subbasins
subbasins = ["Upper Skagit", "Sauk", "Lower Skagit"]
skagit_gdf = gdf[gdf["Name"].str.contains("|".join(subbasins), case=False, regex=True)]

if len(skagit_gdf) == 0:
    print(f"ERROR: Could not find subbasins: {subbasins}")
    exit(1)

print(f"Found {len(skagit_gdf)} subbasins:")
for idx, row in skagit_gdf.iterrows():
    print(f"  - {row['Name']}")

# Get combined bounding box
bounds = skagit_gdf.total_bounds  # [minx, miny, maxx, maxy]

north = bounds[3]  # maxy
south = bounds[1]  # miny
east = bounds[2]   # maxx
west = bounds[0]   # minx

print(f"\nBounding box:")
print(f"  North: {north:.6f}°")
print(f"  South: {south:.6f}°")
print(f"  East:  {east:.6f}°")
print(f"  West:  {west:.6f}°")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================================
# Download and subset data
# ============================================================================
print("\n" + "="*80)
print("Downloading and subsetting data...")
print("="*80 + "\n")

downloaded = 0
failed = 0
failed_years = []

# Set up authentication
auth = (EDL_USERNAME, EDL_PASSWORD)

# Main loop with progress bar
for i in tqdm(range(startyr, endyr + 1), desc="Processing years", unit="year"):
    # Construct download URL
    filename = f"daymet_v4_daily_{region}_{var}_{i}.nc"
    url = f"{BASE_URL}/{filename}"
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{filename}")
    output_file = os.path.join(OUTPUT_DIR, f"{var}_{i}_subset.nc")

    try:
        # Download file with progress bar
        response = requests.get(url, auth=auth, stream=True)
        response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get('content-length', 0))

        with open(temp_file, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, leave=False, desc=f"  Downloading {filename}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Subset spatially to bounding box
        with tqdm(total=1, unit=" ", leave=False, desc=f"  Subsetting {filename}") as pbar:
            ds = xr.open_dataset(temp_file)

            # Subset using lat/lon coordinates (handles 2D coordinate arrays)
            ds_subset = ds.where(
                (ds['lat'] >= south) & (ds['lat'] <= north) &
                (ds['lon'] >= west) & (ds['lon'] <= east),
                drop=True
            )

            # Save subsetted file
            ds_subset.to_netcdf(output_file)
            ds.close()
            ds_subset.close()
            pbar.update(1)

        # Remove temporary file
        os.remove(temp_file)
        downloaded += 1

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            tqdm.write(f"  ✗ Year {i}: File not found (404)")
        else:
            tqdm.write(f"  ✗ Year {i}: HTTP Error {e.response.status_code}")
        failed += 1
        failed_years.append(i)
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception as e:
        tqdm.write(f"  ✗ Year {i}: {str(e)[:60]}")
        failed += 1
        failed_years.append(i)
        if os.path.exists(temp_file):
            os.remove(temp_file)

print("\n" + "="*80)
print("Download and subset complete!")
print("="*80)
print(f"Successfully processed: {downloaded} files")
print(f"Failed: {failed} files")
print(f"Total years: {endyr - startyr + 1}")

if failed_years:
    print(f"\nFailed years: {failed_years}")

print(f"\nData saved to: {OUTPUT_DIR}")
