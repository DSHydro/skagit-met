import boto3.exceptions
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask as dask
import argparse
import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from shapely import vectorized

BUCKET_NAME = 'wrf-cmip6-noversioning'
config = Config(
    signature_version = UNSIGNED,
    max_pool_connections = 24,
    retries = {'mode': 'standard', 'max_attempts': 10}
)
s3 = boto3.client('s3', config=config)

# Parse command arguments from script run in the command line
def setupArgs() -> None:
    parser = argparse.ArgumentParser(description='Download WRF downsampled data and clip to region and save as zarr. See https://dept.atmos.ucla.edu/sites/default/files/alexhall/files/aws_tiers_dirstructure_nov22.pdf')
    parser.add_argument('--model', 
                        required=True,
                        type=str,
                        help='GCM Model and variant of data to download, e.g. cesm2_r11i1p1f1_ssp245  see https://dept.atmos.ucla.edu/alexhall/downscaling-cmip6 for more')
    parser.add_argument('--dataTier', 
                        default=2,
                        type=int,
                        help='Data Tier - which set of WRF output variables/granularity to download. Will be one of 1 (six hourly), 2 (hourly), 3 (daily)')
    parser.add_argument('--domain', 
                        default=2,
                        type=int,
                        help='Downsample scale to select. Will be one of 1 (45KM), 2 (9KM), 3 (3KM CA), or 4 (3KM WY).')
    parser.add_argument('--biasCorrected', 
                        default=False,
                        type=bool,
                        help='Whether to select bias corrected version of model')
    parser.add_argument('--historical', 
                        default=False,
                        type=bool,
                        help='Whether to select historical version of model')
    parser.add_argument('--parameters', 
                        type=str,
                        default='',
                        help='Comma seperated string containing the variables that will be downloaded - these are WRF output vars and defaults to all')
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
    return parser.parse_args()

def generateFileNames(start_date: str, end_date: str, model: str, data_tier: int, domain: int, historical: bool, bias_correction: bool) -> list[str]:
    r = pd.date_range(start_date, end_date, freq='1h', inclusive='both', normalize=True)
    file_prefix = {1: "wrfout", 2: "auxhist"}
    path_prefix = "downscaled_products/gcm"
    if model.startswith("era5"):
        path_prefix = "downscaled_products/reanalysis"
    path = f'{path_prefix}/{model}{"_historical" if historical else ""}{"_bc" if bias_correction else ""}/hourly'
    # Gross check since files start sept 1 in each yearly directory
    return ["%s/%s/d0%s/%s_d01_%s" % (path, d.year if d.month > 9 else d.year - 1, domain, file_prefix[data_tier], pd.to_datetime(d).strftime('%Y-%m-%d_%H:%M:%S')) for d in r]

def downloadS3File(bucket: str, file: str, output_dir: str) -> str:
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    output_file = "%s/%s_%s.nc" % (output_dir, file.split('/')[2], file.split('/')[-1].replace(':', '-'))
    try:
        s3.download_file(bucket, file, output_file)
    except ClientError as e:
        print(f"Failed to download {file} from S3:  {e.response}")
        return None

    return output_file

def downloadMetadataFile(domain: int, output_dir: str, coord: bool = False) -> str:
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    file_name = f'wrfinput_d0{domain}{"_coord.nc" if coord else ""}'
    s3_path = "downscaled_products/wrf_coordinates/%s" % (file_name)
    output_file = "%s/%s" % (output_dir, file_name)

    try:
        s3.download_file(BUCKET_NAME, s3_path, output_file)
    except ClientError as e:
        print(f"Failed to download metadata at {file_name} from S3")
    
    return output_file

def getLatLonHgtFromMetadata(metadata_file: str) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    data = xr.open_dataset(metadata_file)
    lat = data.variables["XLAT"]
    lon = data.variables["XLONG"]
    hgt = data.variables["HGT"]
    
    lon_wrf = lon[0,:,:]
    lat_wrf = lat[0,:,:]
    hgt_wrf = hgt[0,:,:]
    lat_wrf = xr.DataArray(lat_wrf, dims=["y", "x"])
    lon_wrf = xr.DataArray(lon_wrf, dims=["y", "x"])
    hgt_wrf = xr.DataArray(hgt_wrf, dims=["y", "x"])

    return (lat_wrf, lon_wrf, hgt_wrf)

def formatWrfArray(wrf_data: xr.Dataset, lat: xr.DataArray, lon: xr.DataArray, hgt: xr.DataArray, parameters_to_keep: list[str]) -> xr.Dataset:
    wrf_data = wrf_data.assign_coords(lat=lat, lon=lon, hgt=hgt).rename({'south_north': 'y', 'west_east': 'x'})
    time_strs = wrf_data['Times'].astype(str)
    time_strs = [t.replace("_", " ") for t in time_strs.values]
    dts = pd.to_datetime(time_strs).floor('h')
    wrf_data = wrf_data.rename({'Time': 'time'}).assign(time=dts).drop_vars('Times')

    if parameters_to_keep:
        wrf_data = wrf_data[parameters_to_keep]
    
    return wrf_data

def geoMaskWrfArray(wrf_array: xr.Dataset, gejson_path: str) -> xr.Dataset:
    boundary = gpd.read_file(gejson_path)
    mask = vectorized.contains(boundary.geometry[0], wrf_array.lon.values, wrf_array.lat.values)
    
    return wrf_array.where(mask)

def parseParameters(paramString: str) -> list[str]:
    param_list = paramString.split(',')
    if not param_list[0]:
        return []
    return param_list

def cleanUpFiles(files:list) -> None:
    [os.unlink(f) for f in files]

def write_to_zarr(dataset:xr.Dataset, output_dir: str, path:str) -> None:
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    
    dataset.to_zarr(output_dir + '/' + path, mode='w')

if __name__ == "__main__":
    # Get Arguments - model, variables, product, date range, and geo_json
    args = setupArgs()
    parameters = parseParameters(args.parameters)
    files_to_download = generateFileNames(args.startDate, args.endDate, args.model, args.dataTier, args.domain, args.historical, args.biasCorrected)

    # Download 24 hrs at a time
    start_time = dt.now()
    with ThreadPoolExecutor(24) as executor:
        downloaded_files = list(executor.map(lambda file: downloadS3File(BUCKET_NAME, file, args.outputDir), files_to_download))
    end_time = dt.now()

    failed_files = [f for f in downloaded_files if f is None]
    downloaded_files = [f for f in downloaded_files if f is not None]
    print('Time to download {} files: {} seconds'.format(len(downloaded_files), (end_time - start_time).seconds))
    print(f'{len(failed_files)} failed to download')

    if len(downloaded_files) == 0:
        print('No files downloaded. Exiting...')
        exit(0)

    # Get Metadata File for Lat, Lon
    md_file = downloadMetadataFile(args.domain, args.outputDir)
    lat, lon, hgt = getLatLonHgtFromMetadata(md_file)

    # Format, then geo limit by masking
    wrf_array = xr.open_mfdataset(downloaded_files, combine='nested', concat_dim='Time')
    wrf_array_formatted = formatWrfArray(wrf_array, lat, lon, hgt, parameters)
    wrf_array_masked = geoMaskWrfArray(wrf_array_formatted, args.geojson)

    # Write to zarr and cleanup
    write_to_zarr(wrf_array_masked, args.outputDir, args.startDate + '_' + args.endDate + '_wrf_' + args.model + '_data.zarr')
    cleanUpFiles(downloaded_files)
    cleanUpFiles([md_file])