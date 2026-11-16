#!/usr/bin/env python3
"""
Download gridMET daily precipitation and clip it to the Skagit basin.

For each year, downloads the full-CONUS gridMET precipitation file (pr_<year>.nc,
~60 MB, 1/24° ≈ 4 km), clips it to SkagitBoundary.json, and saves the result as a
Zarr. The full-CONUS .nc files are kept in the output folder as a cache.

gridMET covers the contiguous US only, so the Canadian part of the Upper Skagit
(north of 49°N) is NaN. Basin means from gridMET therefore exclude that area,
unlike the other products.

Data source: https://www.northwestknowledge.net/metdata/data/
Reference:   https://www.climatologylab.org/gridmet.html
             Abatzoglou (2013), Int. J. Climatol., doi:10.1002/joc.3413

Usage:
    python gridmet_downloader.py                      # all years, START_YEAR-END_YEAR
    python gridmet_downloader.py --years 2022 2023    # specific years
    python gridmet_downloader.py --years 2021 --overwrite

    Years whose Zarr already exists are skipped unless --overwrite is given, so
    re-running resumes an interrupted run or retries failures. Downloads and Zarr
    writes go to a .part path and are renamed only once complete.

Output layout (variable 'prcp', mm/day, dims time/lat/lon):
    OUTPUT_DIR/<year>_daily_4km_gridMET_data.zarr     e.g. gridmet/2021_daily_4km_gridMET_data.zarr
    OUTPUT_DIR/pr_<year>.nc                           full-CONUS download

Used by (pointing at /data0/skagit_met/data_transfer/data/gridmet):
    analysis/3_bulk_bias_new_version.py
    plots/plot_specific_ar_and_non_ar_events/*.py
    plots/precipitation_long_term_means/water_year_cumulative.py
    plots/spatial_bias_distribution/*.py
    plots/top_50_ar_vs_non-AR/exploring/cumulative_precipitation_plot/ar_vs_non_ar_comparison.py
"""

import argparse
import os
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import requests
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

BASE_URL = "https://www.northwestknowledge.net/metdata/data"
START_YEAR = 1981
END_YEAR = 2025

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]  # skagit_met_for_publication/
DEFAULT_GEOJSON = BASE_DIR / "data/GIS/SkagitBoundary.json"
DEFAULT_OUTPUT_DIR = "/data0/skagit_met/data_transfer/data/gridmet/"

def setup_args():
    parser = argparse.ArgumentParser(description="Download gridMET daily precipitation and clip to Skagit basin.")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(START_YEAR, END_YEAR + 1)),
                        help=f"Years to download (default: {START_YEAR}-{END_YEAR})")
    parser.add_argument("--outputDir", default=DEFAULT_OUTPUT_DIR, help="Directory to save Zarr output.")
    parser.add_argument("--geojson", default=DEFAULT_GEOJSON, help="GeoJSON file for clipping.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess years whose Zarr already exists (default: skip them).")
    return parser.parse_args()

def zarr_path_for(year, output_dir):
    return os.path.join(output_dir, f"{year}_daily_4km_gridMET_data.zarr")

def download_gridmet_year(year, output_dir):
    filename = f"pr_{year}.nc"
    url = f"{BASE_URL}/{filename}"
    local_path = os.path.join(output_dir, filename)
    part_path = local_path + ".part"  # renamed to local_path only once fully downloaded

    if os.path.exists(local_path):
        print(f"[SKIP] {filename} already exists.")
        return local_path

    print(f"[DOWNLOAD] {url} -> {local_path}")
    try:
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(part_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Fail if the connection dropped before the whole file arrived
        size = os.path.getsize(part_path)
        if total_size and size != total_size:
            raise IOError(f"incomplete download ({size} of {total_size} bytes)")

        os.replace(part_path, local_path)
        return local_path
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return None
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)

def process_to_zarr(nc_path, year, boundaries_gdf, output_dir):
    if not nc_path:
        return

    zarr_path = zarr_path_for(year, output_dir)
    part_path = zarr_path + ".part"  # renamed to zarr_path only once fully written

    print(f"[PROCESS] {nc_path} -> {zarr_path}")
    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")

        # gridMET names its time dimension 'day'; the analysis expects 'time'
        if 'day' in ds.coords:
            ds = ds.rename({'day': 'time'})

        # Precipitation variable in gridMET 'pr' files (fallback: any 'precip*' variable)
        var_name = 'precipitation_amount'
        if var_name not in ds.data_vars:
            var_name = [v for v in ds.data_vars if 'precip' in v.lower()][0]

        da = ds[var_name]

        # gridMET is on a regular lat/lon grid (WGS84); clip to the basin outline
        da = da.rio.write_crs("EPSG:4326")
        clipped = da.rio.clip(boundaries_gdf.geometry, boundaries_gdf.crs)

        # Saved as 'prcp' (the name the analysis scripts read)
        clipped = clipped.to_dataset(name='prcp')

        # Save to a .part Zarr, then swap it in once complete
        clipped.to_zarr(part_path, mode="w")
        ds.close()
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        os.replace(part_path, zarr_path)
        print(f"[SUCCESS] Saved {zarr_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {nc_path}: {e}")
    finally:
        if os.path.exists(part_path):
            shutil.rmtree(part_path)

if __name__ == "__main__":
    args = setup_args()
    os.makedirs(args.outputDir, exist_ok=True)

    # Skip years already processed unless --overwrite
    years = [y for y in args.years
             if args.overwrite or not os.path.exists(zarr_path_for(y, args.outputDir))]
    skipped = sorted(set(args.years) - set(years))
    if skipped:
        print(f"[SKIP] {len(skipped)} year(s) already have a Zarr (use --overwrite to redo): {skipped}")
    if not years:
        print("Nothing to do.")
        raise SystemExit(0)

    boundaries = gpd.read_file(args.geojson).to_crs("EPSG:4326")

    with ThreadPoolExecutor(max_workers=2) as executor:
        nc_paths = list(executor.map(lambda y: download_gridmet_year(y, args.outputDir), years))

    for year, nc_path in zip(years, nc_paths):
        process_to_zarr(nc_path, year, boundaries, args.outputDir)
