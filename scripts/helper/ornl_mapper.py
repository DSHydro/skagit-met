import pandas as pd

GLOBUS_ROOT = 'https://hydrosource2.ornl.gov/files/SWA9505V3'
DEFAULT_VARIABLES = ['prcp', 'tmax', 'tmin', 'wind', 'rhum', 'srad', 'lrad']
DEFAULT_REF_SIM = 'DaymetV4'
DEFAULT_HYDRO = 'VIC4'
DEFAULT_SKAGIT_GEOJSON = 'data/GIS/SkagitBoundary.json'
DEFAULT_OUTPUT_PATH = 'data/weather_data/'

ALLOWED_HYDRO_MODELS = ['VIC4']
ALLOWED_VARIABLES = ['prcp', 'tmax', 'tmin', 'wind', 'rhum', 'srad', 'lrad', 'qair',\
                      'vp', 'vpd', 'pres', 'runoff', 'runoffs', 'runoffb', 'swe',\
                      'evap', 'pet', 'soilm', 'PRMS_runoff', 'PRMS_runoffs', 'PRMS_runoffb',\
                        'PRMS_swe', 'PRMS_evap', 'PRMS_pet', 'PRMS_soilm']
ALLOWED_REF_MET_OBS = ['DaymetV4', 'Livneh']
ALLOWED_GCMS = ['ACCESS-CM2','BCC-CSM2-MR', 'CNRM-ESM2-1', 'EC-Earth3', 'MPI-ESM1-2-HR', 'MRI-ESM2-0','NorESM2-MM']
ALLOWED_DOWNSCALING_METHODS = ['DBCCA', 'RegCM']
ALLOWED_CLIMATE_SCENARIOS = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
ENSEMBLE_ID_MAP = {
    'ACCESS-CM2': 'r1i1p1f1',
    'BCC-CSM2-MR': 'r1i1p1f1',
    'CNRM-ESM2-1': 'r1i1p1f2',
    'EC-Earth3': 'r1i1p1f1',
    'MPI-ESM1-2-HR': 'r1i1p1f1',
    'MRI-ESM2-0': 'r1i1p1f1',
    'NorESM2-MM': 'r1i1p1f1'
}

HISTORICAL_START_YEAR = 1950
HISTORICAL_END_YEAR = 2018
REF_START_YEAR = 1980
REF_END_YEAR = 2099

def generate_file_names(ref_met: str, hydro_model:str, variables:list, start_year: str, end_year: str, gcm: str, climate_scenario:str, downscaling_method:str) -> list:
    files = []

    historical = False
    if gcm is None or climate_scenario is None or downscaling_method is None:
        print('No GCM, climate scenario or downscaling method provided. Using reference data only.')
        # DaymetV4/pet
        historical = True

    if not historical:
        ensemble_id = ENSEMBLE_ID_MAP.get(gcm, None)
        if ensemble_id is None:
            print(f'GCM {gcm} not found in ensemble ID map. Please check the GCM name.')
            return files
    
    for variable in variables:
        for y in pd.date_range(start_year, end_year, freq='YS'):
            if historical and y.year > HISTORICAL_END_YEAR:
                print(f'Year {y.year} is greater than {HISTORICAL_END_YEAR}. Skipping...')
                continue

            if historical and y.year < HISTORICAL_START_YEAR:
                print(f'Year {y.year} is less than {HISTORICAL_START_YEAR}. Skipping...')
                continue

            if not historical and y.year < REF_START_YEAR:
                print(f'Year {y.year} is less than {REF_START_YEAR}. Skipping...')
                continue

            if not historical and y.year > REF_END_YEAR:
                print(f'Year {y.year} is greater than {REF_END_YEAR}. Skipping...')
                continue

            if historical:            
                file_path = f'{ref_met}/{variable}/{ref_met}_{hydro_model}_{variable}_{y.year}.nc'
            else:
                if ref_met == 'DaymetV4':
                    ref_met = 'Daymet'
            
                file_path = f'{gcm}_{climate_scenario}_{ensemble_id}_{downscaling_method}_{ref_met}/{variable}/{gcm}_{climate_scenario}_{ensemble_id}_{downscaling_method}_{ref_met}_{hydro_model}_{variable}_{y.year}.nc'
            url = f'{GLOBUS_ROOT}/{file_path}'
            files.append(url)
    return files