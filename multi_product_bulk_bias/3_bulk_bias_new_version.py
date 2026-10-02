"""Multi-product bulk precipitation bias analysis for Skagit Basin.

Loads precipitation data from multiple weather datasets (PRISM, PNNL, Daymet,
CONUS404, UCLA, GridMET, HRRR, and ORNL) over the Skagit Basin region.
For each precipitation event date, computes 3-day window totals and maximums
(spanning days T-2, T-1, T) across all products. Merges results back into
the bulk bias CSV for downstream bias analysis.

Datasets are loaded from:
- /data0/skagit_met/data_transfer/data (VAULT_DIR)
- /data0/hernanqd/plots_code/skagit_basin_de (PROJECT_DIR)

Output is saved to outputs/3_multi_product_bulk_bias_data.csv with columns for
each dataset's 3-day totals and maximums.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
import regionmask
import warnings
from joblib import Parallel, delayed
import tempfile
import zipfile
import rioxarray

warnings.filterwarnings('ignore')

BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/multi_product_bulk_bias"
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PROJECT_DIR = "/data0/hernanqd/plots_code/skagit-met"
HUC8_GEO = os.path.join(PROJECT_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
BULK_CSV = os.path.join(BASE_DIR, "outputs/2_ar_scale_and_sampling.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs/3_multi_product_bulk_bias_data.csv")

def load_regions():
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    return gdf

def get_mask(gdf, lon, lat):
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    for r in ["Upper Skagit", "Sauk", "Lower Skagit"]:
        idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
        region_indices.append(idx)
    return mask.sel(region=region_indices)

def get_decade_folder(year: int) -> str:
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
    """Load PRISM data from new format (TIFF) zip file for a specific date"""
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
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith('.tif'):
                            filepath = os.path.join(root, f)
                            da = rioxarray.open_rasterio(filepath)
                            return da.squeeze().copy(deep=True)
    except Exception as e:
        print(f"  Error loading PRISM (new format) for {date_str}: {e}")

    return None

def load_daymet_from_netcdf(year):
    """Load Daymet data from netCDF file for a specific year"""
    daymet_nc_path = os.path.join(VAULT_DIR, "daymet_new_hq", f"prcp_{year}_subset.nc")

    if not os.path.exists(daymet_nc_path):
        return None

    try:
        ds = xr.open_dataset(daymet_nc_path)
        return ds
    except Exception as e:
        print(f"  Error loading Daymet netCDF for {year}: {e}")
        return None

def calculate_basin_mean(da, mask_2d):
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
    print(f"Processing Year {year} with {len(event_dates)} events...")
    
    # Construct the set of all window dates (T-2, T-1, T)
    window_dates = set()
    for dt in event_dates:
        window_dates.add(dt - pd.Timedelta(days=2))
        window_dates.add(dt - pd.Timedelta(days=1))
        window_dates.add(dt)
    window_dates = sorted(list(window_dates))
    
    prism_daily = None
    pnnl_daily = None
    daymet_daily = None
    conus_daily = None
    ucla_daily = None
    gridmet_daily = None
    hrrr_daily = None
    
    # 1. PRISM (new format - TIFF zip files)
    prism_root = os.path.join(VAULT_DIR, "prism_new_hq")
    prism_data = {}
    gdf = load_regions()

    for date in window_dates:
        da = load_prism_from_new_format(date, prism_root)
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
    if year <= 2020:
        pnnl_path = os.path.join(VAULT_DIR, "PNNL/historical", str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
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
    ornl_path = "/data0/hernanqd/plots_code/exploring/daymet_skagit_precip_daily_basin_backfill.csv"
    ornl_mean_daily = None
    ornl_median_daily = None
    if os.path.exists(ornl_path):
        try:
            ornl_df = pd.read_csv(ornl_path, index_col=0, parse_dates=True)
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
    conus_path = os.path.join(PROJECT_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
    if os.path.exists(conus_path):
        try:
            ds = xr.open_zarr(conus_path)
            ds['time'] = pd.to_datetime(ds.time.values).normalize()
            da = ds['precip_daily'].sel(time=window_dates, method='nearest')
            conus_daily = calculate_basin_mean(da, masks_2d['CONUS404'])
            conus_daily.index = conus_daily.index.normalize()
            ds.close()
        except: pass
        
    # 5. UCLA
    u_paths = [
        os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year-1}.nc"),
        os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year}.nc")
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
    gridmet_path = os.path.join(VAULT_DIR, f"gridmet/{year}_daily_4km_gridMET_data.zarr")
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
    if year >= 2014:
        hrrr_paths = []
        for m in range(1, 13):
            p = os.path.join(VAULT_DIR, "weather_data", f"{year}-{m:02d}_HRRR_data.zarr")
            if os.path.exists(p): hrrr_paths.append(p)
            p_fixed = os.path.join(VAULT_DIR, "weather_data", f"{year}-{m:02d}_HRRR_data_fixed_time.zarr")
            if os.path.exists(p_fixed): hrrr_paths.append(p_fixed)
        next_jan = os.path.join(VAULT_DIR, "weather_data", f"{year+1}-01_HRRR_data.zarr")
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

    # Calculate 3-day window totals and maximums (T-2, T-1, T)
    results = []
    for dt in event_dates:
        w_dates = [dt - pd.Timedelta(days=2), dt - pd.Timedelta(days=1), dt]
        row = {'date': dt}
        
        for name, daily_series in [('prism', prism_daily), ('pnnl', pnnl_daily), ('daymet', daymet_daily),
                                    ('conus', conus_daily), ('ucla', ucla_daily), ('gridmet', gridmet_daily),
                                    ('hrrr', hrrr_daily), ('ornl_mean', ornl_mean_daily), ('ornl_median', ornl_median_daily)]:
            if daily_series is not None and all(d in daily_series.index for d in w_dates):
                row[f'{name}_3d_tot'] = daily_series.loc[w_dates].sum()
                row[f'{name}_3d_max'] = daily_series.loc[w_dates].max()
            else:
                row[f'{name}_3d_tot'] = np.nan
                row[f'{name}_3d_max'] = np.nan
        results.append(row)
        
    return pd.DataFrame(results)

def deduplicate_events_by_prism(df):
    """
    Group contiguous AR/non-AR events and keep the date with highest prism_3d_tot.
    AR: groups contiguous dates with same ar_scale
    Non-AR: groups contiguous non-AR dates
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

        # Keep row with max prism_3d_tot for each group (skip groups with all NaN)
        idx_list = []
        for group_id, group_df in ar_events.groupby('group_id'):
            valid_idx = group_df['prism_3d_tot'].idxmax()
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

        # Keep row with max prism_3d_tot for each group (skip groups with all NaN)
        idx_list = []
        for group_id, group_df in non_ar_events.groupby('group_id'):
            valid_idx = group_df['prism_3d_tot'].idxmax()
            if pd.notna(valid_idx):
                idx_list.append(valid_idx)
        non_ar_deduplicated = non_ar_events.loc[idx_list].drop(columns=['date_gap', 'group_id']).reset_index(drop=True)
    else:
        non_ar_deduplicated = pd.DataFrame()

    # Combine and sort
    result = pd.concat([ar_deduplicated, non_ar_deduplicated]).sort_values('date').reset_index(drop=True)
    return result

def main():
    print("Loading geometry and template CSV...")
    gdf = load_regions()
    df_bulk = pd.read_csv(BULK_CSV)
    df_bulk['date'] = pd.to_datetime(df_bulk['date'])
    
    print("Pre-calculating spatial masks...")
    masks_2d = {}

    # PNNL
    sample_geo = xr.open_dataset(os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc"))
    masks_2d['PNNL'] = get_mask(gdf, sample_geo.XLONG_M.values[0], sample_geo.XLAT_M.values[0]).any(dim='region')

    # Daymet - will be created dynamically in process_year for netCDF format
    # (2D coordinates require loading a sample file first)
    
    # CONUS404
    sample_conus = xr.open_zarr(os.path.join(PROJECT_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr"))
    masks_2d['CONUS404'] = get_mask(gdf, sample_conus.lon, sample_conus.lat).any(dim='region')
    
    # UCLA
    ucla_static = xr.open_dataset(os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc"))
    masks_2d['UCLA'] = get_mask(gdf, ucla_static.lon2d.values, ucla_static.lat2d.values).any(dim='region')
    
    # GridMET
    sample_gridmet = xr.open_zarr(os.path.join(VAULT_DIR, "gridmet/2021_daily_4km_gridMET_data.zarr"))
    masks_2d['GridMET'] = get_mask(gdf, sample_gridmet.lon, sample_gridmet.lat).any(dim='region')
    
    # HRRR
    sample_hrrr = xr.open_zarr(os.path.join(VAULT_DIR, "weather_data/2014-12_HRRR_data.zarr"))
    masks_2d['HRRR'] = get_mask(gdf, sample_hrrr.longitude, sample_hrrr.latitude).any(dim='region')

    years = sorted(df_bulk['date'].dt.year.unique())
    print(f"Starting parallel extraction across {len(years)} years with n_jobs=12...")
    
    results = Parallel(n_jobs=12)(
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
    expected_cols = ['prism_3d_tot', 'prism_3d_max', 'pnnl_3d_tot', 'pnnl_3d_max',
                     'daymet_3d_tot', 'daymet_3d_max', 'conus_3d_tot', 'conus_3d_max',
                     'ucla_3d_tot', 'ucla_3d_max', 'gridmet_3d_tot', 'gridmet_3d_max',
                     'hrrr_3d_tot', 'hrrr_3d_max', 'ornl_mean_3d_tot', 'ornl_mean_3d_max',
                     'ornl_median_3d_tot', 'ornl_median_3d_max']
    for col in expected_cols:
        if col not in df_bulk_updated.columns:
            df_bulk_updated[col] = np.nan

    # Deduplicate contiguous events by keeping highest prism_3d_tot
    print("Deduplicating contiguous events by prism_3d_tot...")
    df_bulk_updated = deduplicate_events_by_prism(df_bulk_updated)

    # Save the updated CSV
    print(f"Saving updated CSV back to {OUTPUT_CSV}...")
    df_bulk_updated.to_csv(OUTPUT_CSV, index=False)
    print("CSV updated successfully!")

if __name__ == "__main__":
    main()
