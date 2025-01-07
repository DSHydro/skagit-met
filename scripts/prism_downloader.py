import pyPRISMClimate
import rioxarray as rxr
import xarray as xr
import os
import glob
import geopandas as gpd
import pandas as pd
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import pathlib  # Python >= 3.4
import dask as dask
import argparse
import time

DEFAULT_PARAMS = ['tmean', 'tmax', 'tmin', 'ppt', 'vpdmax', 'vpdmin']
# Parse command arguments from script run in the command line
def setupArgs() -> None:
    parser = argparse.ArgumentParser(description='Download PRISM 4KM downsampled data and clip to region and save as zarr. See https://www.prism.oregonstate.edu/ and https://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf')
    parser.add_argument('--parameters', 
                        type=str,
                        default='',
                        help='Comma seperated string containing the variables that will be downloaded - these are PRISM output vars and defaults to all')
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
                        default='data/GIS/SkagitRiver_BasinBoundary.json',
                        type=str,
                        help='Path to/name of geo_json file that geogrpahically limits the downloaded data')
    return parser.parse_args()

def create_prism_dataset(min_date: str, max_date: str, dest_path: str, boundaries_gdf: gpd.GeoDataFrame) -> xr.Dataset:
    #Output Zarr
    output_file = "%s/%s_%s_PRISM_data.zarr" % (dest_path, min_date, max_date)
    
    # Collect Individual Variable Data arrays
    rasters = []
    nc_files = pyPRISMClimate.utils.prism_iterator('../data/weather_data/')

    for f in nc_files:
       # open weather file and clip to watershed boundaries
        raster = rxr.open_rasterio(f['full_path'], masked=True)
        raster = raster.rio.clip(boundaries_gdf.to_crs(raster.rio.crs).geometry)
    
        # get date from filename and add as a time coordinate
        date = dt.strptime(f['date'], "%Y-%m-%d")
        raster = raster.expand_dims(dim='time')
        raster.coords['time'] = ('time',[date])
        raster = raster.drop_vars(['spatial_ref']).sel(band=1).drop_vars(['band']).rename(f['variable']).rename({'x':'lon', 'y':'lat'})
        raster.drop_attrs()
    
        #add timestamp to list
        rasters.append(raster)
        
    weather_dataset = xr.merge(rasters)
    weather_dataset.to_zarr(output_file, mode='a')
    
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
    dates = pd.date_range(args.startDate, args.endDate, freq='1d', inclusive='right', normalize=True)
    output_dir = args.outputDir
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    
    # Download Up to 2 vars at a time
    # FTP Client blocks if try to download too many days at once
    # Adding a delay between downloads to avoid getting blocked by the FTP server
    def download_with_backoff(var, min_date, max_date, dest_path):
        max_retries = 5
        delay = 1
        for attempt in range(max_retries):
            try:
                pyPRISMClimate.get_prism_dailys(
                    var,
                    min_date=min_date,
                    max_date=max_date,
                    dest_path=dest_path,
                    keep_zip=False
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
        else:
            print(f"Failed to download {var} after {max_retries} attempts")

    start_time = dt.now()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(download_with_backoff, var, dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"), output_dir)
            for var in parameters
        ]
        for future in futures:
            future.result()  # Wait for all downloads to complete
    end_time = dt.now()

    print('Time to download {} days: {} seconds'.format(len(dates), (end_time - start_time).seconds))

    # Create Dataset, write out to zarr
    mask = gpd.read_file(args.geojson)
    create_prism_dataset(args.startDate, args.endDate, output_dir, mask)

    # cleanup
    weather_files = [f['full_path'] for f in pyPRISMClimate.utils.prism_iterator('../data/weather_data/')]
    meta_data_files =  glob.glob(os.path.join(output_dir, 'PRISM_*_bil.*'))
    clean_up_files(weather_files)
    clean_up_files(meta_data_files)