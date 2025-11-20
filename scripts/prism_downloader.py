import os, sys
from pyproj import datadir

os.environ.setdefault("PROJ_LIB",  os.path.join(sys.prefix, "share", "proj"))
os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "share", "gdal"))
datadir.set_data_dir(os.environ["PROJ_LIB"])  # ensure pyproj sees proj.db


def configure_spatial_env(proj_dir: str | None, gdal_dir: str | None) -> None:
    """Update PROJ/GDAL environment variables when overridden via CLI."""
    if proj_dir:
        os.environ["PROJ_LIB"] = proj_dir
        try:
            datadir.set_data_dir(proj_dir)
        except Exception:
            pass
    if gdal_dir:
        os.environ["GDAL_DATA"] = gdal_dir
    if proj_dir or gdal_dir:
        os.environ.setdefault("PROJ_NETWORK", "ON")

import geopandas as gpd
gpd.options.io_engine = "fiona"
os.environ["GEOPANDAS_USE_PYOGRIO"] = "0"

import rioxarray as rxr
import xarray as xr
import os
import requests
import geopandas as gpd
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile
import pathlib  # Python >= 3.4
import dask as dask
import argparse



BASE_URL = 'https://services.nacse.org/prism/data/get'
# Format options, we need 
DEFAULT_REGION = 'us'
REGION_OPTIONS = ['us', 'ak', 'hi', 'pr']
DEFAULT_RESOLUTION = '4km'
RESOLUTION_OPTIONS = ['4km', '800m', '400m']
DEFAULT_FORMAT = 'nc'
FORMAT_OPTIONS = ['nc', 'bil', 'asc', 'geotiff']
DEFAULT_FREQUENCY = 'daily'
FREQUENCY_OPTIONS = ['daily', 'monthly', 'annual']
DEFAULT_PARAMS = ['tmean', 'tmax', 'tmin', 'ppt', 'vpdmax', 'vpdmin', 'tdmean']

def setupArgs() -> None:
    parser = argparse.ArgumentParser(description='''Download PRISM downsampled data and clip to region and save as zarr. See https://www.prism.oregonstate.edu/ and https://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf
                                     Uses the prism Webservice -- https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf''')
    parser.add_argument('--parameters', 
                        type=str,
                        default=DEFAULT_PARAMS,
                        choices=DEFAULT_PARAMS,
                        nargs='+',
                        help='the variables that will be downloaded - these are PRISM output vars and defaults to all')
    parser.add_argument('--startDate', 
                        type=str,
                        required=True,
                        help='Start date of data to download e.g. 2020-10-01 for October 1, 2020')
    parser.add_argument('--endDate', 
                        type=str,
                        required=True,
                        help='End date of data to download e.g. 2020-10-01 for October 1, 2020')
    parser.add_argument('--outputDir', 
                        default='data/weather_data/',
                        type=str,
                        help='Directory/path to download data/output zarr to.')
    parser.add_argument('--geojson', 
                        default='data/GIS/SkagitBoundary.json',
                        type=str,
                        help='Path to/name of geo_json file that geogrpahically limits the downloaded data')
    parser.add_argument('--region',
                        default=DEFAULT_REGION,
                        type=str,
                        choices=REGION_OPTIONS,
                        help='Region to download data for. Default is US')
    parser.add_argument('--resolution',
                        default=DEFAULT_RESOLUTION,
                        type=str,
                        choices=RESOLUTION_OPTIONS,
                        help='Resolution of data to download. Default is 4km. Only 4km and 800m are available.')
    parser.add_argument('--format',
                        default=DEFAULT_FORMAT,
                        type=str,
                        choices=FORMAT_OPTIONS,
                        help='Format of data to download. Default is nc')
    parser.add_argument('--frequency',
                        default=DEFAULT_FREQUENCY,
                        type=str,
                        choices=FREQUENCY_OPTIONS,
                        help='Frequency of data to download. Default is daily')
    parser.add_argument('--keepZip',
                        type=bool,
                        default=False,
                        help='Keep the zipped files after download. Default is False')
    default_proj = os.environ.get("PROJ_LIB") or os.path.join(sys.prefix, "share", "proj")
    default_gdal = os.environ.get("GDAL_DATA") or os.path.join(sys.prefix, "share", "gdal")
    parser.add_argument('--projLib',
                        type=str,
                        required=False,
                        default=default_proj,
                        help='Path to the PROJ data directory (sets PROJ_LIB).')
    parser.add_argument('--gdalData',
                        type=str,
                        required=False,
                        default=default_gdal,
                        help='Path to the GDAL data directory (sets GDAL_DATA).')

    return parser.parse_args()

def create_prism_dataset(min_date: str, max_date: str, dest_path: str,
                         boundaries_gdf: gpd.GeoDataFrame,
                         data_paths: list[str], frequency: str, resolution: str) -> xr.Dataset:
    output_file = f"{dest_path}/{min_date}_{max_date}_{frequency}_{resolution}_PRISM_data.zarr"

    rasters = []
    nc_files = []

    # Build a normalized list of NetCDF file entries
    for p in filter(None, data_paths):
        if p.endswith(".zip"):
            with ZipFile(p, "r") as zf:
                nc_in_zip = [f for f in zf.namelist() if f.endswith(".nc")]
                if not nc_in_zip:
                    continue
                member = nc_in_zip[0]
                zf.extract(member, path=dest_path)
                nc_path = os.path.join(dest_path, member)
        else:
            nc_path = p

        fname = os.path.basename(nc_path)
        parts = fname.split("_")
        try:
            variable = parts[1]
            date_part = parts[4].split(".")[0] if len(parts) > 4 else parts[-1].split(".")[0]
        except Exception:
            variable = "var"
            date_part = "".join([c for c in fname if c.isdigit()])[:8]

        nc_files.append({"full_path": nc_path, "variable": variable, "date": date_part})

    for f in nc_files:
        # Prefer xarray for NetCDF, then prepare for clip with rioxarray
        ds = xr.open_dataset(f["full_path"], engine="netcdf4")

        # Drop metadata-only variables that cause CRS issues
        ds = ds.drop_vars(["crs", "spatial_ref"], errors="ignore")

        # Select the actual data variable
        var_name = f["variable"] if f["variable"] in ds.data_vars else list(ds.data_vars)[0]
        da = ds[var_name]

        # Add a time coordinate
        if frequency == "daily":
            date = dt.strptime(f["date"], "%Y%m%d")
        elif frequency == "monthly":
            date = dt.strptime(f["date"], "%Y%m")
        else:
            date = dt.strptime(f["date"], "%Y")

        da = da.expand_dims(dim="time").assign_coords(time=("time", [date]))

        # Normalize coordinate names
        rename = {}
        if "x" in da.dims: rename["x"] = "lon"
        if "y" in da.dims: rename["y"] = "lat"
        if "longitude" in da.dims: rename["longitude"] = "lon"
        if "latitude" in da.dims: rename["latitude"] = "lat"
        if rename:
            da = da.rename(rename)

        # Write CRS and set spatial dims only once on proper data var
        da = da.rio.write_crs("EPSG:4326", inplace=False)
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)

        # Clip with shapefile boundary
        da = da.rio.clip(boundaries_gdf.to_crs("EPSG:4326").geometry)

        rasters.append(da.rename(f["variable"]))

    weather_dataset = xr.merge(rasters)
    weather_dataset.to_zarr(output_file, mode="w")
    return weather_dataset

def parseDateRange(startDateString: str, endDateString: str,  frequency: str) -> pd.DatetimeIndex:
    if frequency == 'daily':
        start_date = dt.strptime(startDateString, "%Y-%m-%d")
        end_date = dt.strptime(endDateString, "%Y-%m-%d")
        freq = '1D'
        date_range = pd.date_range(start_date, end_date, freq=freq, normalize=True).strftime('%Y%m%d')
    elif frequency == 'monthly':
        start_date = dt.strptime(startDateString, "%Y-%m")
        end_date = dt.strptime(endDateString, "%Y-%m")
        freq = '1ME'
        date_range = pd.date_range(start_date, end_date, freq=freq, normalize=True).strftime('%Y%m')
    elif frequency == 'annual':
        start_date = dt.strptime(startDateString, "%Y")
        end_date = dt.strptime(endDateString, "%Y")
        freq = '1Y'
        date_range = pd.date_range(start_date, end_date, freq=freq ,normalize=True).strftime('%Y')
    
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}")
    
    return date_range

def clean_up_files(files: list) -> None:
    [pathlib.Path(f).unlink(missing_ok=True) for f in files]

if __name__ == "__main__":
    # Get Arguments - model, variables, product, date range, and geo_json
    args = setupArgs()
    configure_spatial_env(args.projLib, args.gdalData)
    parameters = args.parameters
    dates = parseDateRange(args.startDate, args.endDate, args.frequency)
    output_dir = args.outputDir
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    
    # No Longer using FTP Client
    # '/us/4km/tmin/202204?format=nc'
    url_params = f'/{args.region}/{args.resolution}/'
    query_params = {'format': args.format}

    def download(var, date, output_dir) -> str:
        url = f"{BASE_URL}/{args.region}/{args.resolution}/{var}/{date}"
        try:
            r = requests.get(url, params={"format": args.format}, stream=True)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").lower()
    
            base = os.path.join(output_dir, f"{var}_{date}_{args.resolution}")
            # If it's a zip, keep .zip; otherwise save whatever it is as .nc
            if "zip" in ctype:
                out_path = base + ".zip"
            else:
                out_path = base + ".nc"
    
            with open(out_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
    
            print(f"[DOWNLOAD] {url} -> {out_path} ({ctype or 'unknown content-type'})")
            return out_path
    
        except Exception as e:
            print(f"[DOWNLOAD-ERROR] {url} -> {e}")
            return None


    start_time = dt.now()
    futures = []
    zip_paths = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(download, var, date, output_dir)
            for var in parameters for date in dates
        ]
        zip_paths = [future.result() for future in futures]

    end_time = dt.now()

    print('Time to download {} {}(s): {} seconds'.format(len(dates), args.frequency, (end_time - start_time).seconds))

    # Create Dataset, write out to zarr
    mask = gpd.read_file(args.geojson)
    if args.format != 'nc':
        print('Will only merge and write out to zarr if format is nc for now')
    else:
        print('Creating zarr dataset...')
        create_prism_dataset(args.startDate, args.endDate, output_dir, mask, zip_paths, args.frequency, args.resolution)
        print('Zarr dataset created...')

    # cleanup
    if args.keepZip:
        print('Keeping zipped files...')
    else:
        print('Cleaning up zipped files...')
        clean_up_files(zip_paths)
    
    print('Cleaning up extracted files...')
    clean_up_files([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.nc')])
