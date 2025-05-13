import xarray as xr
import rioxarray as rxr
import fsspec
from concurrent.futures import ThreadPoolExecutor, wait
import pathlib
from datetime import datetime as dt
import dask as dask
import geopandas as gpd
import os
import glob
import argparse
import helper.ornl_mapper as mapper

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
    
def pull_from_globus(url: str) -> str:
    local_path = fsspec.open_local(f"simplecache::{url}", simplecache={'cache_storage': '/tmp/fsspec_cache'}, same_names=True)
    return local_path

def create_ornl_dataset(start_year: str, end_year: str, dest_path: str, geojson: str, reference: str, gcm: str, climate_scenario: str, downscaling_method: str) -> xr.Dataset:
    
    ref = False
    if gcm is None or climate_scenario is None or downscaling_method is None:
        ref = True
    
    #Output Zarr
    output_file = f'{dest_path}/{start_year}_{end_year}{"_ref_" + reference  if ref else ""}{"_"+ gcm + "_" + climate_scenario + "_" + downscaling_method if not ref else ""}_ORNL_data.zarr'
    
    # Collect Individual Variable Data arrays
    rasters = []
    mask = gpd.read_file(geojson)
    nc_files = glob.iglob(os.path.join('/tmp/fsspec_cache/', '*.nc'))
    weather_dataset = None

    for f in nc_files:
        # open weather file and clip to watershed boundaries
        try:
            raster = rxr.open_rasterio(f, masked=True)
            raster = raster.rio.write_crs(mask.crs)
            raster = raster.rio.clip(mask.geometry)
        except Exception as e:
            print(f'Error opening {f}: {e}\n Trying to continue...')
            continue

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
    with ThreadPoolExecutor(max_workers=4) as executor:
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

    # Create Dataset, write out to zarr
    create_ornl_dataset(args.startYear, args.endYear, output_dir, args.geojson,\
                        args.reference, args.gcm, args.climateScenario, args.downscalingMethod)

    # cleanup
    to_clean = glob.iglob(os.path.join('/tmp/fsspec_cache/', '*.nc'))
    clean_up_files(to_clean)