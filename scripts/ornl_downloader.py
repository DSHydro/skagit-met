import xarray as xr
import rioxarray as rxr
import fsspec
from concurrent.futures import ThreadPoolExecutor, wait
import pandas as pd
import pathlib
from datetime import datetime as dt
import dask
import geopandas as gpd
import os
import glob
import argparse

GLOBUS_ROOT = 'https://g-e320e6.63720f.75bc.data.globus.org/gen101/world-shared/doi-data/OLCF/202402/10.13139_OLCF_2311812'
DEFAULT_PARAMS = ['prcp', 'tmax', 'tmin', 'wind', 'rhum', 'srad', 'lrad']
DEFAULT_SIM = 'DaymetV4'
DEFAULT_HYDRO = 'VIC4'
DEFAULT_SKAGIT_GEOJSON = 'data/GIS/SkagitBoundary.json'
DEFAULT_OUTPUT_PATH = 'data/weather_data/'


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
                        default=DEFAULT_OUTPUT_PATH,
                        type=str,
                        help='Directory/path to download data/output zarr to.')
    parser.add_argument('--geojson', 
                        default=DEFAULT_SKAGIT_GEOJSON,
                        type=str,
                        help='Path to/name of geo_json file that geogrpahically limits the downloaded data')
    parser.add_argument('--reference', 
                        default=DEFAULT_SIM,
                        type=str,
                        help='Reference meteorological observations to use e.g. DaymetV4 or Livneh')
    parser.add_argument('--hydroModel', 
                        default=DEFAULT_HYDRO,
                        type=str,
                        help='Hydro model to use e.g. VIC4 (currently only one supported)')
    return parser.parse_args()

def generate_file_names(met_data: str, hydro_model:str, variables:list, start_year: str, end_year: str) -> list:
    files = []
    for variable in variables:
        for y in pd.date_range(start_year, end_year, freq='YS'):
            file_path = f'{met_data}/{variable}/{met_data}_{hydro_model}_{variable}_{y.year}.nc'
            url = f'{GLOBUS_ROOT}/{file_path}'
            files.append(url)
    return files
    
def pull_from_globus(url: str) -> str:
    local_path = fsspec.open_local(f"filecache::{url}", filecache={'cache_storage': '/tmp/fsspec_cache'}, same_names=True)
    return local_path

def create_ornl_dataset(start_year: str, end_year: str, dest_path: str, geojson: str) -> xr.Dataset:
    #Output Zarr
    output_file = "%s/%s_%s_ORNL_data.zarr" % (dest_path, start_year, end_year)
    
    # Collect Individual Variable Data arrays
    rasters = []
    mask = gpd.read_file(geojson)
    l = glob.iglob(os.path.join('/tmp/fsspec_cache/', '*.nc'))

    for f in l:
        # open weather file and clip to watershed boundaries
        raster = rxr.open_rasterio(f, masked=True)
        raster = raster.rio.write_crs(mask.crs)
        raster = raster.rio.clip(mask.geometry)

        #add timestamp to list
        rasters.append(raster)
        
        weather_dataset = xr.merge(rasters)
        weather_dataset = weather_dataset.rename({'x': 'lon', 'y': 'lat'})
        weather_dataset = weather_dataset.drop_vars('spatial_ref')
        weather_dataset['time'] = weather_dataset.time.dt.floor('D')
        weather_dataset.to_zarr(output_file, mode='w')
    
    return weather_dataset

def parseParameters(paramString: str) -> list[str]:
    param_list = paramString.split(',')
    if not param_list[0]:
        return DEFAULT_PARAMS
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

    files = generate_file_names(args.reference, args.hydroModel, parameters, args.startYear, args.endYear)
    start_time = dt.now()
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [executor.submit(pull_from_globus, file) for file in files]
        wait(futures, 60)

    def downloaded_files(futures: list) -> list:
        downloaded_files = []
        for f in futures:
            try:
                downloaded_files.append(f.result())
            except:
                pass

        return downloaded_files

    downloaded_files = downloaded_files(futures)
    end_time = dt.now()

    print('Time to download {} files: {} seconds'.format(len(downloaded_files), (end_time - start_time).seconds))

    # Create Dataset, write out to zarr
    create_ornl_dataset(args.startYear, args.endYear, output_dir, args.geojson)

    # cleanup
    to_clean = glob.iglob(os.path.join('/tmp/fsspec_cache/', '*.nc'))
    clean_up_files(to_clean)