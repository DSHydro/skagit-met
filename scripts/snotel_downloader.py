from datetime import datetime
import pandas as pd
import geopandas as gpd
from metloom.pointdata import SnotelPointData
from metloom.variables import SnotelVariables
import argparse
import xarray as xr
from pathlib import Path
import requests

DEFAULT_SNOTEL_VARS = [SnotelPointData.ALLOWED_VARIABLES.SNOWDEPTH,
            SnotelPointData.ALLOWED_VARIABLES.SWE,
            SnotelPointData.ALLOWED_VARIABLES.PRECIPITATION,
            SnotelPointData.ALLOWED_VARIABLES.PRECIPITATIONACCUM,
            SnotelPointData.ALLOWED_VARIABLES.TEMP,
            SnotelPointData.ALLOWED_VARIABLES.TEMPAVG,
            SnotelPointData.ALLOWED_VARIABLES.TEMPMAX,
            SnotelPointData.ALLOWED_VARIABLES.TEMPMIN]
ALLOWED_SNOTEL_VARS = ['SNOWDEPTH', 'SWE', 'PRECIPITATION', 'ACCUMULATED PRECIPITATION', 'AIR TEMP', 'AVG AIR TEMP', 'MAX AIR TEMP']
SNOTEL_VAR_LOOKUP = dict(zip(ALLOWED_SNOTEL_VARS, DEFAULT_SNOTEL_VARS))
FREQUENCY_CHOICES = ['hourly', 'daily']
DEFAULT_FREQUENCY = 'hourly'
DEFAULT_GEOJSON = 'data/GIS/SkagitBoundary.json'
DEFAULT_OUTPUT_DIR = 'data/weather_data/'

# Parse command arguments from script run in the command line
def setupArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='''Download Snotel point data and save as zarr.
                                      See https://www.nrcs.usda.gov/wps/portal/wcc/home/aboutUs/monitoringPrograms/automatedSnowMonitoring/ and https://metloom.readthedocs.io/en/latest/''')
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--variables', 
                        type=str,
                        default='',
                        help='''Comma seperated string containing the variables that will be downloaded (e.g. SWE,TEMP,PRECIPITATION) - these are snotel variables and defaults to all available.
                            Current options are:
                            SNOWDEPTH, SWE, PRECIPITATION, ACCUMULATED PRECIPITATION, AIR TEMP, AVG AIR TEMP, MAX AIR TEMP, RELATIVE HUMIDITY
                            Snotel rarely has RH.
                        ''')
    group.add_argument('--stationIDs',
                        type=str,
                        help='''Comma seperated string containing the station IDs to download e.g. 460:WA:SNTL,460:WA:SNTL,460:WA:SNTL
                        Must provide either this, or a GeoJSON file with the boundaries of the region to download data from.
                        Not Both''')
    group.add_argument('--geojson', 
                        default=DEFAULT_GEOJSON,
                        type=str,
                        help='''Path to/name of geo_json file that geographically limits the downloaded data
                        Must provide either this, or a list of station ids.
                        Not Both''')
    parser.add_argument('--startDate', 
                        type=str,
                        required=True,
                        help='Start date of data to download e.g. 2020-10-01 for October 1, 2020')
    parser.add_argument('--endDate', 
                        type=str,
                        required=True,
                        help='End date of data to download e.g. 2020-10-01 for October 1, 2020')
    parser.add_argument('--outputDir', 
                        default=DEFAULT_OUTPUT_DIR,
                        type=str,
                        help='Directory/path to download data/output zarr to.')
    parser.add_argument('--frequency',
                        default=DEFAULT_FREQUENCY,
                        type=str,
                        choices=FREQUENCY_CHOICES,
                        help='Frequency of data to download. Options are daily or hourly. Defaults to hourly')
    return parser.parse_args()

def parseVariables(paramString: str) -> tuple[list[str], list[SnotelVariables]]:
    var_list = paramString.split(',')
    if not var_list[0]:
        return ALLOWED_SNOTEL_VARS, DEFAULT_SNOTEL_VARS
    else:
        # Let user nkow variable not allowed, but continue with rest
        var_strs = []
        for var in var_list:
            if var.strip() not in SNOTEL_VAR_LOOKUP:
                print(f'{var} is not a valid Snotel variable. Skipping...')
            else:
                var_strs.append(var.strip())
        # Try to get some variables 
        return var_strs, [SNOTEL_VAR_LOOKUP[var.strip()] for var in var_strs]

def parseStationIDs(paramString: str) -> list[str]:
    return paramString.split(',')

def getStationData(stations: list[str], frequency: str, start: datetime, end: datetime, variables: list[SnotelVariables], var_strs: list[str]) -> xr.Dataset:
    points = [SnotelPointData(station, '') for station in stations]
    dfs = [getDataByFrequency(point, frequency, start, end, variables) for point in points]
    return createDataset(dfs, frequency, var_strs)

def getGeometryData(geojson: str, frequency: str, start: datetime, end: datetime, variables: list[SnotelVariables], var_strs: list[str]) -> xr.Dataset:
    geometry = gpd.read_file(geojson)
    points = SnotelPointData.points_from_geometry(geometry, variables)
    dfs = [getDataByFrequency(point, frequency, start, end, variables) for point in points]
    return createDataset(dfs, frequency, var_strs)
    
def createDataset(dataframes: list[gpd.GeoDataFrame], frequency: str, var_strs: list[str]) -> xr.Dataset:
    try:
        snotel_df = pd.concat(dataframes)
    except ValueError:
        print('No variable data found for the given stations on given dates. Exiting...')
        exit(0)
    # Sort by date:
    snotel_df.sort_index(level=0, inplace=True)
    # Convert to xarray dataset and make sure in correct datetime format
    if frequency == 'hourly':
        snotel_df.index = snotel_df.index.set_levels(snotel_df.index.levels[0].floor('h'), level=0)
    elif frequency == 'daily':
        snotel_df.index = snotel_df.index.set_levels(snotel_df.index.levels[0].floor('D'), level=0)
    snotel_xr = snotel_df.to_xarray()
    snotel_xr = snotel_xr.assign_coords(site=snotel_xr.site.astype(str))
    snotel_xr = snotel_xr.assign_coords(datetime=pd.to_datetime(snotel_xr.datetime))
    snotel_xr = snotel_xr.rename({'datetime': 'time'})

    # Ensure 'site' column exists
    if 'site' not in snotel_df.columns:
        snotel_df['site'] = snotel_df.index.get_level_values('site')

    # Do the same for site name
    if 'site_name' not in snotel_df.columns:
        snotel_df['site_name'] = snotel_df.index.get_level_values('site_name')
        
    # Extract coordinates from the geometry column
    snotel_df['lat'] = snotel_df.geometry.apply(lambda geom: geom.y)
    snotel_df['lon'] = snotel_df.geometry.apply(lambda geom: geom.x)
    snotel_df['elevation_ft'] = snotel_df.geometry.apply(lambda geom: geom.z)
    
    # Filter snotel_df to only include sites that exist in snotel_xr
    snotel_df = snotel_df[snotel_df['site'].isin(snotel_xr.site.values)]
    
    # Assign new coordinates to the xarray dataset
    # Ensure unique 'site' values
    snotel_df = snotel_df.drop_duplicates(subset='site')
    # Same for site_name
    snotel_df = snotel_df.drop_duplicates(subset='site_name')
    
    snotel_xr = snotel_xr.assign_coords(
        lat=('site', snotel_df['lat'].values),
        lon=('site', snotel_df['lon'].values),
        elevation_ft=('site', snotel_df['elevation_ft'].values),
        site_name=('site', snotel_df['site_name'].values)
    )
    # Clean up issue here where some variables are not in the data
    var_strs = [var for var in var_strs if var in snotel_xr]
    snotel_xr = snotel_xr[var_strs]
    return snotel_xr

def getDataByFrequency(point: SnotelPointData, frequency: str, start: datetime, end: datetime, variables: list[SnotelVariables]) -> pd.DataFrame:
    try:
        if frequency == 'hourly':
            df = point.get_hourly_data(start, end, variables=variables)
        elif frequency == 'daily':
            df = point.get_daily_data(start, end, variables=variables)
        else:
            raise ValueError('Frequency must be either daily or hourly')
        df['site_name'] = point.name
        return df
    except requests.exceptions.HTTPError as e:
        print(f'Error downloading data for {point.station_id}. Skipping...')
        return pd.DataFrame()
    
def writeToZarr(ds: xr.Dataset, output_dir: str, startDate: str, endDate: str, frequency: str) -> None:
    #Output Zarr
    output_file = "%s/%s_%s_SNOTEL_%s_data.zarr" % (output_dir, startDate, endDate, frequency)

    ds.to_zarr(output_file, mode='w')

if __name__ == '__main__':
    args = setupArgs()
    var_strs, variables = parseVariables(args.variables)
    # Return if no valid vars
    if len(variables) == 0:
        print('No valid variables provided. Exiting....')
        exit(0)
    
    output_dir = args.outputDir
    if output_dir[-1] == '/':
        output_dir = output_dir[:-1]
    # Verify Output Dir
    if not Path(output_dir).exists():
        print(f'Output directory {output_dir} does not exist. Try creating before running. Exiting...')
        exit(0)
    
    # Dates
    try:
        startDate = datetime.fromisoformat(args.startDate)
        endDate = datetime.fromisoformat(args.endDate)
    except ValueError:
        raise ValueError('Dates must be in the format YYYY-MM-DD')
    
    startTime = datetime.now()
    if args.stationIDs:
        stationIDs = parseStationIDs(args.stationIDs)
        if len(stationIDs) == 0:
            print('No station IDs provided. Exiting...')
            exit(0)
        xr = getStationData(stationIDs, args.frequency, startDate, endDate, variables, var_strs)
    else:
        xr = getGeometryData(args.geojson, args.frequency, startDate, endDate, variables, var_strs)
    
    writeToZarr(xr, output_dir, args.startDate, args.endDate, args.frequency)
    endTime = datetime.now()
    print('Time to download: {} seconds'.format((endTime - startTime).seconds))
    