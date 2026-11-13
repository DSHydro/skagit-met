"""Extract basin-mean precipitation from multiple products for each AR / non-AR day.

For every date T in the output of script 2 (AR days and isolated non-AR days),
basin-mean daily precipitation is computed for each product and summarised
over windows ending on T:

* 3-day window (T-2 to T): total and daily maximum - PRISM only
  (``prism_3d_tot``, ``prism_3d_max``).
* 7-day window (T-6 to T): total and daily maximum - all products
  (``<product>_7d_tot``, ``<product>_7d_max``).

A window metric is NaN unless all days of the window are available.

Products (column prefix):
    prism        PRISM daily 4 km (zipped GeoTIFF, one file per day)
    pnnl         PNNL WRF historical run, hourly summed to daily (up to PNNL_LAST_YEAR)
    daymet       Daymet daily
    conus        CONUS404 daily
    ucla         UCLA WRF-ERA5 d02 daily
    gridmet      gridMET daily
    hrrr         HRRR forecast hour 1, hourly summed to daily (from HRRR_FIRST_YEAR)
    ornl_mean    ORNL DBCCA-Daymet downscaled CMIP6 (ssp585), mean of the 6 GCMs
    ornl_median  ORNL DBCCA-Daymet downscaled CMIP6 (ssp585), median of the 6 GCMs
                 (both from ORNL_CSV, pre-computed basin means; its DaymetV4
                 column is excluded)

Basin mean: unweighted mean of grid cells inside the union of the HUC8
sub-basins in REGION_NAMES (Upper Skagit, Sauk, Lower Skagit), using
regionmask on each product's native grid.

Deduplication: after extraction, runs of consecutive AR days with the same
``ar_scale`` - and runs of consecutive non-AR days - are each reduced to the
single day with the highest ``prism_7d_tot``.

Inputs:
    <BASE_DIR>/analysis/outputs/2_ar_scale_and_sampling.csv   (from script 2)
    <BASE_DIR>/data/GIS/SkagitSubBasin_HUC8.geojson
    External precipitation datasets - see "PATHS - external" below.

Output:
    <BASE_DIR>/analysis/outputs/3_multi_product_bulk_bias_data.csv
        Columns of the input plus the 3-day and 7-day metrics listed above.

Usage:
    python 3_bulk_bias_new_version.py
"""

import os
import tempfile
import warnings
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import rioxarray
import xarray as xr
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ============================================================
# PATHS - project files (relative to this repository)
# ============================================================
# Project root = parent of the folder containing this script (i.e. skagit_met_for_publication/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
BULK_CSV = os.path.join(BASE_DIR, "analysis/outputs/2_ar_scale_and_sampling.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs/3_multi_product_bulk_bias_data.csv")

# ============================================================
# PATHS - external precipitation datasets (edit to your local copies)
# ============================================================
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PROJECT_DIR = "/data0/hernanqd/plots_code/skagit-met"

PRISM_DIR = os.path.join(VAULT_DIR, "prism_new_hq")           # <decade>/prism_ppt_us_25m_YYYYMMDD.zip
DAYMET_DIR = os.path.join(VAULT_DIR, "daymet_new_hq")         # prcp_YYYY_subset.nc
PNNL_DIR = os.path.join(VAULT_DIR, "PNNL/historical")         # YYYY/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.YYYY.nc
PNNL_GEO_FILE = os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc")
UCLA_DIR = os.path.join(VAULT_DIR, "ucla_era5_d02_daily")     # prec/prec.daily.era5.d02.YYYY.nc
UCLA_GEO_FILE = os.path.join(UCLA_DIR, "static/wrfinput_d02_coord.nc")
GRIDMET_DIR = os.path.join(VAULT_DIR, "gridmet")              # YYYY_daily_4km_gridMET_data.zarr
GRIDMET_GRID_FILE = os.path.join(GRIDMET_DIR, "2021_daily_4km_gridMET_data.zarr")
HRRR_DIR = os.path.join(VAULT_DIR, "weather_data")            # YYYY-MM_HRRR_data[_fixed_time].zarr
HRRR_GRID_FILE = os.path.join(HRRR_DIR, "2014-12_HRRR_data.zarr")
CONUS404_ZARR = os.path.join(PROJECT_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
ORNL_CSV = "/data0/hernanqd/plots_code/exploring/daymet_skagit_precip_daily_basin_backfill.csv"

# ============================================================
# SETTINGS
# ============================================================
REGION_NAMES = ["Upper Skagit", "Sauk", "Lower Skagit"]  # HUC8 sub-basins forming the basin mask
PNNL_LAST_YEAR = 2020    # PNNL historical run ends in 2020
HRRR_FIRST_YEAR = 2014   # HRRR data start in 2014
N_JOBS = 12              # parallel workers (one year per worker)

def load_regions():
    """Load the Skagit HUC8 sub-basin polygons in geographic coordinates (EPSG:4326)."""
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    return gdf

def get_mask(gdf, lon, lat):
    """Return a 3D (region, y, x) boolean mask of the REGION_NAMES sub-basins on the given grid.

    Callers combine the regions with ``.any(dim='region')`` to get a 2D basin mask.
    """
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    for r in REGION_NAMES:
        idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
        region_indices.append(idx)
    return mask.sel(region=region_indices)

def get_decade_folder(year: int) -> str:
    """Return the PRISM sub-folder name that holds a given year (e.g. 1981-1990, 2021-2024)."""
    if 2021 <= year <= 2024:
        return "2021-2024"
    start_decade = (year // 10) * 10
    if start_decade == 1980 and year > 1980:
        return "1981-1990"
    if year % 10 == 0 and year > 1980:
        return f"{start_decade-9}-{start_decade}"
    if start_decade % 10 == 0:
        return f"{start_decade+1}-{start_decade+10}"
    return f"{start_decade}-{(year//10)*10+9}"

def load_prism_from_new_format(date, prism_root):
    """Load one day of PRISM precipitation from its zipped GeoTIFF.

    Returns a 2D DataArray (y, x), or None if the file is missing or unreadable.
    """
    date_str = date.strftime('%Y%m%d')
    year = date.year
    decade_folder = get_decade_folder(year)
    zip_path = os.path.join(prism_root, decade_folder, f"prism_ppt_us_25m_{date_str}.zip")

    if not os.path.exists(zip_path):
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith('.tif'):
                            filepath = os.path.join(root, f)
                            da = rioxarray.open_rasterio(filepath)
                            return da.squeeze().copy(deep=True)
    except Exception as e:
        print(f"  Error loading PRISM (new format) for {date_str}: {e}")

    return None

def load_daymet_from_netcdf(year):
    """Open the Daymet netCDF file for one year, or return None if missing or unreadable."""
    daymet_nc_path = os.path.join(DAYMET_DIR, f"prcp_{year}_subset.nc")

    if not os.path.exists(daymet_nc_path):
        return None

    try:
        ds = xr.open_dataset(daymet_nc_path)
        return ds
    except Exception as e:
        print(f"  Error loading Daymet netCDF for {year}: {e}")
        return None

def calculate_basin_mean(da, mask_2d):
    """Unweighted mean of the grid cells inside the basin mask.

    Parameters
    ----------
    da : xarray.DataArray
        3D (time, y, x) or 2D (y, x) precipitation field.
    mask_2d : xarray.DataArray or numpy.ndarray
        2D basin mask on the same grid (cells > 0 are inside the basin).

    Returns
    -------
    pandas.Series (3D input), indexed by time, or float (2D input).
    For 2D input (PRISM), negative values (no-data) are set to NaN first.
    NaNs are ignored in the mean.
    """
    data = da.values
    mask = mask_2d.values if hasattr(mask_2d, 'values') else mask_2d
    if data.ndim == 3:
        time_steps = data.shape[0]
        data_flat = data.reshape(time_steps, -1)
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            mean_vals = np.nanmean(data_flat[:, mask_idx], axis=1)
        else:
            mean_vals = np.full(time_steps, np.nan)
        time_coord = da.time.values if 'time' in da.coords else (da.day.values if 'day' in da.coords else range(time_steps))
        return pd.Series(mean_vals, index=pd.to_datetime(time_coord))
    elif data.ndim == 2:
        data = data.astype(float)
        data[data < 0] = np.nan
        data_flat = data.flatten()
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            mean_val = np.nanmean(data_flat[mask_idx])
        else:
            mean_val = np.nan
        return mean_val
    else:
        raise ValueError(f"Unexpected data dimensions: {data.ndim}")

def process_year(year, event_dates, masks_2d):
    """Compute the 3-day and 7-day precipitation metrics for all event dates in one year.

    Parameters
    ----------
    year : int
        Calendar year being processed (one parallel job per year).
    event_dates : array of datetime64
        Dates T (AR or non-AR days) in this year.
    masks_2d : dict
        2D basin masks per product grid ('PNNL', 'CONUS404', 'UCLA', 'GridMET',
        'HRRR'); the Daymet mask is added here from the Daymet file itself.

    Returns
    -------
    pandas.DataFrame
        One row per event date with columns 'date', 'prism_3d_tot', 'prism_3d_max'
        and '<product>_7d_tot', '<product>_7d_max' for every product.
    """
    print(f"Processing Year {year} with {len(event_dates)} events...")

    # Construct the set of all window dates (T-6 through T for 7-day window)
    window_dates = set()
    for dt in event_dates:
        for i in range(6, -1, -1):
            window_dates.add(dt - pd.Timedelta(days=i))
    window_dates = sorted(list(window_dates))
    
    prism_daily = None
    pnnl_daily = None
    daymet_daily = None
    conus_daily = None
    ucla_daily = None
    gridmet_daily = None
    hrrr_daily = None
    
    # 1. PRISM (new format - TIFF zip files)
    prism_data = {}
    gdf = load_regions()

    for date in window_dates:
        da = load_prism_from_new_format(date, PRISM_DIR)
        if da is None:
            continue
        lon = da.x.values
        lat = da.y.values
        m_prism = get_mask(gdf, lon, lat).any(dim='region')
        mean_val = calculate_basin_mean(da, m_prism)
        prism_data[date.normalize()] = mean_val

    if prism_data:
        prism_daily = pd.Series(prism_data)
        prism_daily.index = pd.to_datetime(prism_daily.index).normalize()
        
    # 2. PNNL
    if year <= PNNL_LAST_YEAR:
        pnnl_path = os.path.join(PNNL_DIR, str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        if os.path.exists(pnnl_path):
            try:
                ds = xr.open_dataset(pnnl_path)
                hourly_dates = []
                for d in window_dates:
                    hourly_dates.extend(pd.date_range(d, d + pd.Timedelta(hours=23), freq='h'))
                available_hourly = [d for d in hourly_dates if d in ds.time.values]
                if available_hourly:
                    da_subset = ds['PREC_ACC_NC'].sel(time=available_hourly)
                    s_hourly = calculate_basin_mean(da_subset, masks_2d['PNNL'])
                    pnnl_daily = s_hourly.resample('1D').sum()
                    pnnl_daily.index = pnnl_daily.index.normalize()
                ds.close()
            except: pass

    # 3. Daymet (new format - netCDF files)
    try:
        ds = load_daymet_from_netcdf(year)
        if ds is not None:
            ds['time'] = pd.to_datetime(ds.time.values).normalize()
            da = ds['prcp']

            # Create Daymet mask if not cached
            if 'Daymet' not in masks_2d:
                lon_2d = ds['lon'].values
                lat_2d = ds['lat'].values
                masks_2d['Daymet'] = get_mask(gdf, lon_2d, lat_2d).any(dim='region')

            da_window = da.sel(time=window_dates, method='nearest')
            daymet_daily = calculate_basin_mean(da_window, masks_2d['Daymet'])
            daymet_daily.index = daymet_daily.index.normalize()
            daymet_daily = daymet_daily[~daymet_daily.index.duplicated(keep='first')]
            ds.close()
    except Exception as e:
        print(f"Error processing Daymet for year {year}: {e}")

    # 3.5 ORNL (Mean and Median Ensemble from backfill CSV)
    ornl_mean_daily = None
    ornl_median_daily = None
    if os.path.exists(ORNL_CSV):
        try:
            ornl_df = pd.read_csv(ORNL_CSV, index_col=0, parse_dates=True)
            ornl_df.index = pd.to_datetime(ornl_df.index).normalize()
            ornl_mean_daily = ornl_df.iloc[:, 1:].mean(axis=1)  # mean across models
            ornl_median_daily = ornl_df.iloc[:, 1:].median(axis=1)  # median across models
            # Filter to window dates
            available_dates = [d for d in window_dates if d in ornl_mean_daily.index]
            if available_dates:
                ornl_mean_daily = ornl_mean_daily.loc[available_dates]
                ornl_median_daily = ornl_median_daily.loc[available_dates]
        except: pass

    # 4. CONUS404
    if os.path.exists(CONUS404_ZARR):
        try:
            ds = xr.open_zarr(CONUS404_ZARR)
            ds['time'] = pd.to_datetime(ds.time.values).normalize()
            da = ds['precip_daily'].sel(time=window_dates, method='nearest')
            conus_daily = calculate_basin_mean(da, masks_2d['CONUS404'])
            conus_daily.index = conus_daily.index.normalize()
            ds.close()
        except: pass
        
    # 5. UCLA
    u_paths = [
        os.path.join(UCLA_DIR, "prec", f"prec.daily.era5.d02.{year-1}.nc"),
        os.path.join(UCLA_DIR, "prec", f"prec.daily.era5.d02.{year}.nc")
    ]
    series_list = []
    for p in u_paths:
        if os.path.exists(p):
            try:
                ds = xr.open_dataset(p)
                u_var = "prec" if "prec" in ds.data_vars else "pr"
                if 'day' in ds.dims: ds = ds.rename({'day': 'time'})
                ds['time'] = pd.to_datetime(ds.time.values).normalize()
                available_dates = [d for d in window_dates if d in ds.time.values]
                if available_dates:
                    da_subset = ds[u_var].sel(time=available_dates)
                    s = calculate_basin_mean(da_subset, masks_2d['UCLA'])
                    s.index = s.index.normalize()
                    series_list.append(s)
                ds.close()
            except: pass
    if series_list:
        ucla_daily = pd.concat(series_list).sort_index()
        ucla_daily = ucla_daily[~ucla_daily.index.duplicated(keep='first')]

    # 6. GridMET
    gridmet_path = os.path.join(GRIDMET_DIR, f"{year}_daily_4km_gridMET_data.zarr")
    if os.path.exists(gridmet_path):
        try:
            ds = xr.open_zarr(gridmet_path)
            if 'day' in ds.dims: ds = ds.rename({'day': 'time'})
            ds['time'] = pd.to_datetime(ds.time.values).normalize()
            var = 'prcp' if 'prcp' in ds.data_vars else 'precipitation_amount'
            da = ds[var].sel(time=window_dates, method='nearest')
            gridmet_daily = calculate_basin_mean(da, masks_2d['GridMET'])
            gridmet_daily.index = gridmet_daily.index.normalize()
            ds.close()
        except: pass
        
    # 7. HRRR
    if year >= HRRR_FIRST_YEAR:
        hrrr_paths = []
        for m in range(1, 13):
            p = os.path.join(HRRR_DIR, f"{year}-{m:02d}_HRRR_data.zarr")
            if os.path.exists(p): hrrr_paths.append(p)
            p_fixed = os.path.join(HRRR_DIR, f"{year}-{m:02d}_HRRR_data_fixed_time.zarr")
            if os.path.exists(p_fixed): hrrr_paths.append(p_fixed)
        next_jan = os.path.join(HRRR_DIR, f"{year+1}-01_HRRR_data.zarr")
        if os.path.exists(next_jan): hrrr_paths.append(next_jan)
        
        series_list = []
        for p in hrrr_paths:
            try:
                ds = xr.open_zarr(p, consolidated=False)
                var_name = 'APCP_sfc' if 'APCP_sfc' in ds.data_vars else 'tp'
                if var_name in ds.data_vars:
                    da_full = ds[var_name]
                    if "forecast_hour" in ds.coords:
                        da_full = da_full.where(ds.forecast_hour == 1, drop=True)
                    elif "step" in ds.coords:
                        if np.issubdtype(ds.step.dtype, np.timedelta64):
                            da_full = da_full.where(ds.step == np.timedelta64(1, "h"), drop=True)
                        else:
                            da_full = da_full.where(ds.step == 1, drop=True)
                    da_full = da_full.sortby('time').drop_duplicates('time')
                    
                    hourly_dates = []
                    for d in window_dates:
                        hourly_dates.extend(pd.date_range(d, d + pd.Timedelta(hours=23), freq='h'))
                    available_hourly = [d for d in hourly_dates if d in da_full.time.values]
                    if available_hourly:
                        da_subset = da_full.sel(time=available_hourly)
                        s_hourly = calculate_basin_mean(da_subset, masks_2d['HRRR'])
                        s_daily = s_hourly.resample('1D').sum()
                        s_daily.index = s_daily.index.normalize()
                        series_list.append(s_daily)
                ds.close()
            except: pass
        if series_list:
            hrrr_daily = pd.concat(series_list).sort_index()
            hrrr_daily = hrrr_daily[~hrrr_daily.index.duplicated(keep='first')]

    # Calculate 3-day and 7-day window totals and maximums
    results = []
    for dt in event_dates:
        w_dates_3d = [dt - pd.Timedelta(days=2), dt - pd.Timedelta(days=1), dt]
        w_dates_7d = [dt - pd.Timedelta(days=6), dt - pd.Timedelta(days=5), dt - pd.Timedelta(days=4),
                      dt - pd.Timedelta(days=3), dt - pd.Timedelta(days=2), dt - pd.Timedelta(days=1), dt]
        row = {'date': dt}

        # Calculate PRISM 3-day and 7-day metrics
        if prism_daily is not None:
            if all(d in prism_daily.index for d in w_dates_3d):
                row['prism_3d_tot'] = prism_daily.loc[w_dates_3d].sum()
                row['prism_3d_max'] = prism_daily.loc[w_dates_3d].max()
            else:
                row['prism_3d_tot'] = np.nan
                row['prism_3d_max'] = np.nan

            if all(d in prism_daily.index for d in w_dates_7d):
                row['prism_7d_tot'] = prism_daily.loc[w_dates_7d].sum()
                row['prism_7d_max'] = prism_daily.loc[w_dates_7d].max()
            else:
                row['prism_7d_tot'] = np.nan
                row['prism_7d_max'] = np.nan
        else:
            row['prism_3d_tot'] = np.nan
            row['prism_3d_max'] = np.nan
            row['prism_7d_tot'] = np.nan
            row['prism_7d_max'] = np.nan

        # Calculate 7-day window totals and maximums for other products (T-6 through T)
        for name, daily_series in [('pnnl', pnnl_daily), ('daymet', daymet_daily),
                                    ('conus', conus_daily), ('ucla', ucla_daily), ('gridmet', gridmet_daily),
                                    ('hrrr', hrrr_daily), ('ornl_mean', ornl_mean_daily), ('ornl_median', ornl_median_daily)]:
            if daily_series is not None and all(d in daily_series.index for d in w_dates_7d):
                row[f'{name}_7d_tot'] = daily_series.loc[w_dates_7d].sum()
                row[f'{name}_7d_max'] = daily_series.loc[w_dates_7d].max()
            else:
                row[f'{name}_7d_tot'] = np.nan
                row[f'{name}_7d_max'] = np.nan
        results.append(row)
        
    return pd.DataFrame(results)

def deduplicate_events_by_prism(df):
    """Reduce runs of consecutive days to one representative day each.

    - AR days (ar_scale > 0): consecutive dates with the same ar_scale form one group.
    - Non-AR days (ar_scale == 0): consecutive dates form one group.

    From each group, the day with the highest prism_7d_tot is kept. Groups in
    which prism_7d_tot is NaN for every day are dropped.
    """
    df = df.sort_values('date').reset_index(drop=True)

    # Process AR events (ar_scale > 0)
    ar_mask = df['ar_scale'] > 0
    ar_events = df[ar_mask].copy().reset_index(drop=True)

    if len(ar_events) > 0:
        # Detect scale changes and date gaps
        ar_events['scale_changed'] = ar_events['ar_scale'].ne(ar_events['ar_scale'].shift()).fillna(True)
        ar_events['date_gap'] = ar_events['date'].diff() > pd.Timedelta(days=1)
        ar_events.loc[0, 'date_gap'] = False
        ar_events['group_id'] = (ar_events['scale_changed'] | ar_events['date_gap']).cumsum()

        # Keep row with max prism_7d_tot for each group (skip groups with all NaN)
        idx_list = []
        for _, group_df in ar_events.groupby('group_id'):
            valid_idx = group_df['prism_7d_tot'].idxmax()
            if pd.notna(valid_idx):
                idx_list.append(valid_idx)
        ar_deduplicated = ar_events.loc[idx_list].drop(columns=['scale_changed', 'date_gap', 'group_id']).reset_index(drop=True)
    else:
        ar_deduplicated = pd.DataFrame()

    # Process non-AR events (ar_scale == 0)
    non_ar_events = df[~ar_mask].copy().reset_index(drop=True)

    if len(non_ar_events) > 0:
        # Detect date gaps
        non_ar_events['date_gap'] = non_ar_events['date'].diff() > pd.Timedelta(days=1)
        non_ar_events.loc[0, 'date_gap'] = False
        non_ar_events['group_id'] = non_ar_events['date_gap'].cumsum()

        # Keep row with max prism_7d_tot for each group (skip groups with all NaN)
        idx_list = []
        for _, group_df in non_ar_events.groupby('group_id'):
            valid_idx = group_df['prism_7d_tot'].idxmax()
            if pd.notna(valid_idx):
                idx_list.append(valid_idx)
        non_ar_deduplicated = non_ar_events.loc[idx_list].drop(columns=['date_gap', 'group_id']).reset_index(drop=True)
    else:
        non_ar_deduplicated = pd.DataFrame()

    # Combine and sort
    result = pd.concat([ar_deduplicated, non_ar_deduplicated]).sort_values('date').reset_index(drop=True)
    return result

def main():
    """Build basin masks, extract metrics for all years in parallel, deduplicate and save."""
    print("Loading geometry and template CSV...")
    gdf = load_regions()
    df_bulk = pd.read_csv(BULK_CSV)
    df_bulk['date'] = pd.to_datetime(df_bulk['date'])
    
    print("Pre-calculating spatial masks...")
    masks_2d = {}

    # PNNL
    sample_geo = xr.open_dataset(PNNL_GEO_FILE)
    masks_2d['PNNL'] = get_mask(gdf, sample_geo.XLONG_M.values[0], sample_geo.XLAT_M.values[0]).any(dim='region')

    # (Daymet mask is created inside process_year from the first Daymet file read)

    # CONUS404
    sample_conus = xr.open_zarr(CONUS404_ZARR)
    masks_2d['CONUS404'] = get_mask(gdf, sample_conus.lon, sample_conus.lat).any(dim='region')
    
    # UCLA
    ucla_static = xr.open_dataset(UCLA_GEO_FILE)
    masks_2d['UCLA'] = get_mask(gdf, ucla_static.lon2d.values, ucla_static.lat2d.values).any(dim='region')
    
    # GridMET
    sample_gridmet = xr.open_zarr(GRIDMET_GRID_FILE)
    masks_2d['GridMET'] = get_mask(gdf, sample_gridmet.lon, sample_gridmet.lat).any(dim='region')
    
    # HRRR
    sample_hrrr = xr.open_zarr(HRRR_GRID_FILE)
    masks_2d['HRRR'] = get_mask(gdf, sample_hrrr.longitude, sample_hrrr.latitude).any(dim='region')

    years = sorted(df_bulk['date'].dt.year.unique())
    print(f"Starting parallel extraction across {len(years)} years with n_jobs={N_JOBS}...")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_year)(
            year, 
            df_bulk[df_bulk['date'].dt.year == year]['date'].dt.normalize().unique(), 
            masks_2d
        ) for year in years
    )
    
    print("Combining parallel results...")
    corrected_df = pd.concat(results, ignore_index=True)
    corrected_df['date'] = pd.to_datetime(corrected_df['date'])
    
    # Merge back into original dataframe by date
    # Overwrite the spatial columns with corrected ones
    cols_to_overwrite = [col for col in corrected_df.columns if col != 'date']
    
    print("Merging corrected columns into original dataframe...")
    df_bulk_updated = df_bulk.drop(columns=cols_to_overwrite, errors='ignore')
    df_bulk_updated = pd.merge(df_bulk_updated, corrected_df, on='date', how='inner')

    # Ensure all expected data columns exist (create with NaN if missing)
    expected_cols = ['prism_3d_tot', 'prism_3d_max', 'prism_7d_tot', 'prism_7d_max',
                     'pnnl_7d_tot', 'pnnl_7d_max', 'daymet_7d_tot', 'daymet_7d_max',
                     'conus_7d_tot', 'conus_7d_max', 'ucla_7d_tot', 'ucla_7d_max',
                     'gridmet_7d_tot', 'gridmet_7d_max', 'hrrr_7d_tot', 'hrrr_7d_max',
                     'ornl_mean_7d_tot', 'ornl_mean_7d_max', 'ornl_median_7d_tot', 'ornl_median_7d_max']
    for col in expected_cols:
        if col not in df_bulk_updated.columns:
            df_bulk_updated[col] = np.nan

    # Deduplicate contiguous events by keeping highest prism_7d_tot
    print("Deduplicating contiguous events by prism_7d_tot...")
    df_bulk_updated = deduplicate_events_by_prism(df_bulk_updated)

    # Save the updated CSV
    print(f"Saving updated CSV back to {OUTPUT_CSV}...")
    df_bulk_updated.to_csv(OUTPUT_CSV, index=False)
    print("CSV updated successfully!")

if __name__ == "__main__":
    main()
