import os, sys, pathlib

def first_existing(paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None

def open_and_clip_nc(nc_path, mask_gdf):
    # Try engines that don't need GDAL plugins
    for eng in ("h5netcdf", "netcdf4"):
        try:
            ds = xr.open_dataset(nc_path, engine=eng, chunks={})
            break
        except Exception:
            ds = None
    if ds is None:
        raise RuntimeError(f"Could not open {nc_path} with h5netcdf or netcdf4")

    # Pick a variable to clip (each file usually has one primary var)
    var = [v for v in ds.data_vars][0]
    da = ds[var]

    # Ensure we have lon/lat names, then set spatial dims for rioxarray
    # Adjust these if your files use 'x','y' or 'longitude','latitude'
    rename_map = {}
    if "x" in da.dims: rename_map["x"] = "lon"
    if "y" in da.dims: rename_map["y"] = "lat"
    if rename_map:
        da = da.rename(rename_map)

    if {"lon","lat"}.issubset(da.dims):
        da = da.rio.write_crs("EPSG:4326", inplace=False)
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        da = da.rio.clip(mask_gdf.geometry, mask_gdf.crs)
    else:
        # Fallback: if coords named differently, tweak names above
        raise RuntimeError(f"{nc_path} missing expected lon/lat dims")

    return da.to_dataset(name=var)

# Candidate locations for PROJ & GDAL data
conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
candidates_proj = [
    os.environ.get("PROJ_LIB"),
    os.path.join(conda_prefix, "share", "proj"),
    os.path.join(sys.prefix, "share", "proj"),
    "/usr/share/proj",
]
candidates_gdal = [
    os.environ.get("GDAL_DATA"),
    os.path.join(conda_prefix, "share", "gdal"),
    os.path.join(sys.prefix, "share", "gdal"),
    "/usr/share/gdal",
]

PROJ_DIR = first_existing(candidates_proj)
GDAL_DIR = first_existing(candidates_gdal)

if not PROJ_DIR:
    raise RuntimeError(f"Could not locate PROJ data dir. Tried: {candidates_proj}")
if not GDAL_DIR:
    # Not fatal for many ops, but warn loudly
    print(f"Warning: Could not locate GDAL data dir. Tried: {candidates_gdal}")

os.environ["PROJ_LIB"] = PROJ_DIR
if GDAL_DIR:
    os.environ["GDAL_DATA"] = GDAL_DIR
os.environ.setdefault("PROJ_NETWORK", "ON")

# Tell pyproj explicitly (must be after envs, before anyone uses CRS)
try:
    from pyproj import datadir as _pyproj_datadir
    _pyproj_datadir.set_data_dir(PROJ_DIR)
except Exception as e:
    print(f"Warning: failed to set pyproj data dir: {e}")

print(f"[DEBUG] Using PROJ_LIB={PROJ_DIR}")
if GDAL_DIR:
    print(f"[DEBUG] Using GDAL_DATA={GDAL_DIR}")

import xarray as xr
import rioxarray as rxr
import fsspec
from concurrent.futures import ThreadPoolExecutor, wait
from urllib.error import URLError
import time
import pathlib
from datetime import datetime as dt
import dask as dask
import geopandas as gpd
import os
import glob
import argparse
import helper.ornl_mapper as mapper

CACHE_DIR = os.path.expanduser("~/.cache/fsspec_ornl")
os.makedirs(CACHE_DIR, exist_ok=True)


def open_nc_as_ds(path):
    last = None
    for eng in ("netcdf4", "h5netcdf"):
        try:
            # chunk if you like: chunks={"time": 30}
            return xr.open_dataset(path, engine=eng)
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to open {path} with netcdf4/h5netcdf: {last}")

def prep_and_clip(ds, mask_gdf):
    # Normalize lon/lat names
    if "longitude" in ds.dims: ds = ds.rename({"longitude":"lon"})
    if "latitude" in ds.dims:  ds = ds.rename({"latitude":"lat"})
    if "x" in ds.dims and "y" in ds.dims:
        ds = ds.rename({"x":"lon","y":"lat"})

    # Write CRS & spatial dims for rioxarray
    ds = ds.rio.write_crs("EPSG:4326", inplace=False)
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

    # Clip to your basin polygon
    return ds.rio.clip(mask_gdf.geometry, mask_gdf.crs, drop=True)
    
# Parse command arguments from script run in the command line
def setupArgs() -> None:
    parser = argparse.ArgumentParser(description='Download Daily ORNL 4KM downsampled data and clip to region and save as zarr. See https://hydrosource.ornl.gov/data/datasets/9505v3_1/')
    parser.add_argument('--parameters', 
                        type=str,
                        default='',
                        help='Comma seperated string containing the variables that will be downloaded - defaults to prcp,tmax,tmin,wind,rhum,srad,lrad')
    parser.add_argument('--startYear', 
                        type=str,
                        required=True,
                        help='Start year of daily data to download e.g. 1999')
    parser.add_argument('--endYear', 
                        type=str,
                        required=True,
                        help='End year of data to download e.g 2001')
    parser.add_argument('--outputDir', 
                        default=mapper.DEFAULT_OUTPUT_PATH,
                        type=str,
                        help='Directory/path to download data/output zarr to.')
    parser.add_argument('--geojson', 
                        default=mapper.DEFAULT_SKAGIT_GEOJSON,
                        type=str,
                        help='Path to/name of geo_json file that geogrpahically limits the downloaded data')
    parser.add_argument('--reference', 
                        default=mapper.DEFAULT_REF_SIM,
                        type=str,
                        choices=mapper.ALLOWED_REF_MET_OBS,
                        help="""Reference meteorological observations to use e.g. DaymetV4 or Livneh
                        Default is DaymetV4, if only ths is priovided, will use simulation driven only from the reference data
                        If using a GCM, include the GCM name, climate scenario, and downscaling method""")
    parser.add_argument('--hydroModel', 
                        default=mapper.DEFAULT_HYDRO,
                        choices=mapper.ALLOWED_HYDRO_MODELS,
                        type=str,
                        help='Hydro model to use e.g. VIC4 (currently only one supported)')
    parser.add_argument('--gcm', 
                        choices=mapper.ALLOWED_GCMS,
                        type=str,
                        help='Global climate model to use e.g. ACCESS-CM2, CNRM-ESM-1, etc')
    parser.add_argument('--climateScenario', 
                        choices=mapper.ALLOWED_CLIMATE_SCENARIOS,
                        type=str,
                        help='Climate scenario to use e.g. ssp585, ssp245, etc')
    parser.add_argument('--downscalingMethod', 
                        choices=mapper.ALLOWED_DOWNSCALING_METHODS,
                        type=str,
                        help='Downscaling method used to downscale GCM data to 4KM resolution, e.g. DBCCA')
    return parser.parse_args()
    
def pull_from_globus(url: str, retries: int = 5, blocksize: int = 1024 * 1024) -> str:
    """
    Download to simplecache, force a full read to populate the cache,
    then verify local size matches remote. Retry on mismatch.
    """
    httpfs = fsspec.filesystem("http")
    target_name = os.path.basename(url)
    local_path = os.path.join(CACHE_DIR, target_name)

    for attempt in range(1, retries + 1):
        # open via simplecache and stream the whole file to ensure it fully caches
        try:
            with fsspec.open(
                f"simplecache::{url}",
                "rb",
                simplecache={"cache_storage": CACHE_DIR, "same_names": True, "check_files": True},
            ) as f:
                while True:
                    chunk = f.read(blocksize)
                    if not chunk:
                        break
            # verify size
            remote_size = httpfs.info(url).get("size", None)
            local_size = os.path.getsize(local_path) if os.path.exists(local_path) else -1
            if remote_size is None or local_size == remote_size:
                return local_path  # success
            else:
                # size mismatch -> retry after removing partial file
                try:
                    os.remove(local_path)
                except OSError:
                    pass
                time.sleep(min(2 ** attempt, 10))
        except Exception as e:
            # network hiccup: clean and retry
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
            except OSError:
                pass
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(f"Failed to download (verified) after {retries} attempts: {url}")

def create_ornl_dataset(
    start_year: str,
    end_year: str,
    dest_path: str,
    geojson: str,
    reference: str,
    gcm: str,
    climate_scenario: str,
    downscaling_method: str,
) -> xr.Dataset:

    # reference-only vs GCM path
    ref = not (gcm and climate_scenario and downscaling_method)

    # Output Zarr path
    out_name = (
        f"{start_year}_{end_year}"
        f"{('_ref_' + reference) if ref else f'_{gcm}_{climate_scenario}_{downscaling_method}'}"
        "_ORNL_data.zarr"
    )
    output_file = os.path.join(dest_path.rstrip("/"), out_name)

    # Mask geometry (force to WGS84 so it matches lon/lat grids)
    mask = gpd.read_file(geojson)
    if mask.crs is None:
        raise RuntimeError(f"GeoJSON has no CRS: {geojson}")
    mask = mask.to_crs("EPSG:4326")

    # Deterministic file list (and fail early if nothing was downloaded)
    nc_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.nc")))
    if not nc_files:
        raise RuntimeError(f"No NetCDF files found in cache: {CACHE_DIR}")

    rasters: list[xr.Dataset] = []

    for f in nc_files:
        try:
            clipped = open_and_clip_nc(f, mask)   # your helper that opens with netcdf4/h5netcdf, normalizes lon/lat, sets CRS & clips
            # Optional: chunk for nicer Zarr writes (tune as needed)
            if "time" in clipped.dims:
                clipped = clipped.chunk({"time": 30})
            rasters.append(clipped)
        except Exception as e:
            print(f"Error opening {f}: {e}\n Trying to continue...")
            continue

    if not rasters:
        raise RuntimeError("All NetCDFs failed to open/clip. Check engines or file formats.")

    # Merge safely and silence future warnings:
    weather_dataset = xr.merge(
        rasters,
        join="outer",               # future default will be 'exact' -> be explicit
        compat="override",          # future default will change -> be explicit
        combine_attrs="drop_conflicts",
    )

    # Cleanups
    weather_dataset = weather_dataset.drop_vars("spatial_ref", errors="ignore")
    if "time" in weather_dataset.coords:
        weather_dataset["time"] = weather_dataset.time.dt.floor("D")

    # Write Zarr (prefer v2 for broad compatibility; avoid “consolidated v3” warning)
    weather_dataset.to_zarr(
        output_file,
        mode="w",
        consolidated=False,         # avoid v3 consolidated warning in your environment
        zarr_version=2,             # most tools expect v2 today
    )

    print(f"Wrote: {output_file}")
    return weather_dataset

def parseParameters(paramString: str) -> list[str]:
    param_list = paramString.split(',')
    if not param_list[0]:
        return mapper.DEFAULT_VARIABLES
    
    for var in param_list:
        if var not in mapper.ALLOWED_VARIABLES:
            print(f'Variable {var} not found in allowed variables list. Please check the variable name.')
            print(f'Removing {var} from the list of variables and continuing with download.')
            param_list.remove(var)
    
    return param_list

def clean_up_files(files:list) -> None:
    [pathlib.Path(f).unlink(missing_ok=True) for f in files]

if __name__ == "__main__":
    # Get Arguments - model, variables, product, date range, and geo_json
    args = setupArgs()
    parameters = parseParameters(args.parameters)
    output_dir = args.outputDir
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]

    files = mapper.generate_file_names(args.reference, args.hydroModel, parameters, args.startYear, args.endYear, args.gcm, args.climateScenario, args.downscalingMethod)
    start_time = dt.now()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pull_from_globus, file) for file in files]
        wait(futures)

    def downloaded_files(futures: list) -> list:
        downloaded_files = []
        for f in futures:
            try:
                downloaded_files.append(f.result())
            except Exception as e:
                print(f"Error downloading file: {f}, exception: {e}")
                pass

        return downloaded_files

    downloaded_files = downloaded_files(futures)
    end_time = dt.now()

    print('Time to download {} files: {} seconds'.format(len(downloaded_files), (end_time - start_time).seconds))

    nc_files = list(glob.iglob(os.path.join(CACHE_DIR, "*.nc")))
    if not nc_files:
        raise RuntimeError("No NetCDF files were downloaded successfully; try re-running or check network.")

    # Create Dataset, write out to zarr
    create_ornl_dataset(args.startYear, args.endYear, output_dir, args.geojson,\
                        args.reference, args.gcm, args.climateScenario, args.downscalingMethod)

    # cleanup
    to_clean = glob.iglob(os.path.join(CACHE_DIR, '*.nc'))
    clean_up_files(to_clean)