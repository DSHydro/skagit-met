import os
import argparse
import sys

from herbie import FastHerbie, wgrib2
import shapely
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import dask as dask
import cfgrib
    
# Parse command arguments from script run in the command line
def setupArgs() -> None:
    parser = argparse.ArgumentParser(description='Download HRRR data using Herbie, and segment to a specific geographic region')
    parser.add_argument('--model', 
                        default='hrrr',
                        type=str,
                        required=False,
                        help='Model of data to download, defaults to hrrr')
    parser.add_argument('--product', 
                        default='sfc',
                        type=str,
                        required=False,
                        help='Model product to download, defuaults to surface level features')
    parser.add_argument('--parameters', 
                        type=str,
                        required=True,
                        help='Comma seperated tring containing the variables and level of the vars that will be downloaded e.g. TMP:surface,RH:2 m above ground')
    parser.add_argument('--startDate', 
                        default='2020-01-01',
                        type=str,
                        required=True,
                        help='Start date of data to download e.g. 2020-10-01 for October 1, 2020')
    parser.add_argument('--endDate', 
                        default='2020-01-02',
                        type=str,
                        required=True,
                        help='End date of data to download e.g. 2020-10-01 for October 1, 2020')
    parser.add_argument('--geoJson', 
                        default='data/GIS/SkagitBoundary.json',
                        type=str,
                        required=False,
                        help='Path to/name of geo_json file that geogrpahically limits the downloaded data')
    parser.add_argument('--outputDir', 
                        default='../../../data0/balaji24/data/weather_data/',
                        type=str,
                        help='Directory/path to download data/output zarr to.')
    return parser.parse_args()

def getFastHerbie(start_date: str, end_date: str, model: str, product: str, save_dir: str ) -> FastHerbie:
    date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq="1h"
    )
    return FastHerbie(
        date_range, model=model, product=product, fxx=range(0,2),
        save_dir=save_dir, priority=["aws", "nomads", "pando"])

# Parse GeoJson File into tuple containing boundaries
def parseGeoJson(geojson_path: str) -> tuple[float, float, float, float]:
    mask = gpd.read_file(geojson_path)
    minLon, minLat, maxLon, maxLat = mask.total_bounds
    return (minLon, maxLon, minLat, maxLat)

def limitGeographicRange(bounds, subsetFiles):
    keep = []
    for f in subsetFiles:
        try:
            keep.append(wgrib2.region(f, bounds, name='skagit-basin'))
        except Exception as e:
            print(f"wgrib2 failed on {f}: {e} — skipping")
    return keep

# Use Fast herbie to subset and download parameters
def downloadParameters(parameters, fh):
    fields = [f":{p}" for p in parameters]
    param_regex = fr"^(?:{'|'.join(fields)})"
    files = fh.download(param_regex)

    good = []
    for f in files:
        if not f:
            continue
        try:
            # be stricter; genuinely usable files are >> 500 KB
            if os.path.getsize(f) > 500_000:
                good.append(f)
            else:
                print(f"Tiny/corrupt file, skipping: {f}")
                os.unlink(f)
        except FileNotFoundError:
            pass
    return good


def parseParameters(paramString: str) -> list[str]:
    return paramString.split(',')

def cleanUpFiles(subsetFiles:list) -> None:
    [os.unlink(f) for f in subsetFiles]

def maskDataset(ds: xr.Dataset, mask_file: str) -> xr.Dataset:
    mask_shape = gpd.read_file(mask_file)
    mask = shapely.contains_xy(mask_shape.geometry[0], ds.longitude.values, ds.latitude.values)
    masked_data_set = ds.where(mask)

    return masked_data_set

def mergeDatasets(regionSubsetGribFiles: list) -> xr.Dataset:
    datasets = []
    dropVars = ["surface", "heightAboveGround", "valid_time", "step"]
    dropVarsStep = dropVars + ["t", "r2", "si10", "sdswrf", "sdlwrf"]

    for f in regionSubsetGribFiles:
        try:
            groups = cfgrib.open_datasets(f, indexpath='', decode_timedelta=False)
        except Exception as e:
            print(f"cfgrib could not open {f}: {e} — skipping")
            continue

        # robust step detection
        merged = xr.merge([
            ds.drop_vars(dropVarsStep, errors="ignore")
            if getattr(ds, "step", None) is not None and np.array_equal(ds.step.values, np.timedelta64(1, "h"))
            else ds.drop_vars(dropVars, errors="ignore")
            for ds in groups
        ])
        try:
            merged.load()
        except Exception as e:
            print(f"load() failed for {f}: {e} — skipping")
            continue

        datasets.append(merged)

    if not datasets:
        raise RuntimeError("No valid datasets to merge")

    other_vars = [ds for ds in datasets if 'tp' not in ds.variables]
    tp_f001   = [ds for ds in datasets if 'tp' in ds.variables]

    parts = []
    if tp_f001:
        parts.append(xr.concat(tp_f001, dim='time'))
    if other_vars:
        parts.append(xr.concat(other_vars, dim='time'))
    if not parts:
        raise RuntimeError("No datasets after separating tp/other")

    with xr.set_options(keep_attrs=True):
        combined_ds = xr.combine_by_coords(parts, compat='override')

    # convert lon to [-180, 180] if source was [0, 360]
    if (combined_ds.longitude > 180).any():
        combined_ds['longitude'] = combined_ds.longitude - 360

    return combined_ds

def write_to_zarr(dataset:xr.Dataset, output_dir: str, path:str) -> None:
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    dataset.to_zarr(output_dir + '/' + path, mode='w')

def iter_months(start: str, end: str):
    """Yield (month_start, month_end) as strings YYYY-MM-DD for each month in [start, end]."""
    s = pd.to_datetime(start).normalize()
    e = pd.to_datetime(end).normalize()
    month_starts = pd.date_range(start=s, end=e, freq="MS")
    for ms in month_starts:
        me = (ms + pd.offsets.MonthEnd(0))
        if me > e:
            me = e
        yield ms.strftime("%Y-%m-%d"), me.strftime("%Y-%m-%d")

if __name__ == "__main__":
    args = setupArgs()
    parameters = parseParameters(args.parameters)

    # Read geo bounds once
    bounds = parseGeoJson(args.geoJson)

    for m_start, m_end in iter_months(args.startDate, args.endDate):
        out_name = f"{m_start[:7]}_HRRR_data.zarr"
        out_path = os.path.join(args.outputDir.rstrip("/"), out_name)
        if os.path.exists(out_path):
            print(f"Exists, skipping: {out_path}")
            continue

        print(f"\nDownloading HRRR {args.model}/{args.product} for {m_start} → {m_end}")
        fh = getFastHerbie(m_start, m_end, args.model, args.product, args.outputDir)

        fh_files = []
        geo_limited_files = []
        try:
            fh_files = downloadParameters(parameters, fh)
            if not fh_files:
                print(f"No usable downloads for {m_start} → {m_end}; skipping.")
                continue

            geo_limited_files = limitGeographicRange(bounds, fh_files)
            if not geo_limited_files:
                print(f"No geo-clipped files for {m_start} → {m_end}; skipping.")
                continue

            try:
                mergedDs = mergeDatasets(geo_limited_files)
            except RuntimeError as e:
                print(f" {e} for {m_start} → {m_end}; skipping.")
                continue

            maskedDs = maskDataset(mergedDs, args.geoJson)
            write_to_zarr(maskedDs, args.outputDir, out_name)
            print(f"Wrote: {out_path}")
        finally:
            if fh_files:
                cleanUpFiles(fh_files)
            if geo_limited_files:
                cleanUpFiles([str(f) + '.idx' for f in geo_limited_files])
                cleanUpFiles(geo_limited_files)



    
