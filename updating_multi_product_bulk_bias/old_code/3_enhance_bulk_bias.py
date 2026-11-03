import os
import pandas as pd
import xarray as xr
import numpy as np
import fsspec

# --- Paths ---
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de/multi_product_bulk_bias"
DATA_DIR = os.path.join(BASE_DIR, "outputs")
INPUT_CSV = os.path.join(DATA_DIR, "prism_pnnl_bulk_bias_data.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "multi_product_bulk_bias_data.csv")

VAULT_DIR = "/data0/skagit_met/data_transfer/data"
CONUS_PATH = os.path.join(BASE_DIR, "data", "weather_data", "conus404_skagit_precip_daily_basin_only.csv")
DAYMET_BACKFILL_PATH = os.path.join(BASE_DIR, "data", "weather_data", "daymet_skagit_precip_daily_basin_backfill.csv")
GRIDMET_BACKFILL_PATH = os.path.join(BASE_DIR, "data", "weather_data", "gridmet_skagit_precip_daily_basin_backfill.csv")
ORNL_PATH = "/data0/hernanqd/plots_code/exploring/daymet_skagit_precip_daily_basin_backfill.csv"

def get_decade_folder(year: int) -> str:
    start_decade = (year // 10) * 10
    if start_decade == 1980 and year > 1980: return "1981-1990"
    if year % 10 == 0 and year > 1980: return f"{start_decade-9}-{start_decade}"
    if start_decade % 10 == 0: return f"{start_decade+1}-{start_decade+10}"
    return f"{start_decade}-{(year//10)*10+9}"


def get_ornl_zarr(year):
    if 1981 <= year <= 2011:
        p = os.path.join(VAULT_DIR, "climate_sets/1981_2011_ORNL_data.zarr")
        if os.path.exists(p): return p
    p1 = os.path.join(VAULT_DIR, f"ornl/{year}_{year}_ref_DaymetV4_ORNL_data.zarr")
    if os.path.exists(p1): return p1
    p2 = os.path.join(VAULT_DIR, f"DaymetV4/{year}_{year}_ORNL_data.zarr")
    if os.path.exists(p2): return p2
    return None

def enhance_bulk():
    print("Loading base PRISM/PNNL dataset...")
    df = pd.read_csv(INPUT_CSV)
    df['date'] = pd.to_datetime(df['date'])
    
    for col in ['daymet_3d_tot', 'daymet_3d_max', 'conus_3d_tot', 'conus_3d_max', 'hrrr_3d_tot', 'hrrr_3d_max', 'ucla_3d_tot', 'ucla_3d_max', 'gridmet_3d_tot', 'gridmet_3d_max', 'ornl_mean_3d_tot', 'ornl_mean_3d_max', 'ornl_median_3d_tot', 'ornl_median_3d_max']:
        if col not in df.columns:
            df[col] = np.nan

    # 1) Pre-load CONUS Daily Series (1981-Present)
    conus_daily = None
    if os.path.exists(CONUS_PATH):
        try:
            print("Loading CONUS404 basin average CSV...")
            conus_daily = pd.read_csv(CONUS_PATH, index_col=0, parse_dates=True).iloc[:, 0]
            conus_daily.index = pd.to_datetime(conus_daily.index).normalize()
            
            print("Loading Daymet backfill CSV...")
            daymet_backfill = pd.read_csv(DAYMET_BACKFILL_PATH, index_col=0, parse_dates=True).iloc[:, 0]
            daymet_backfill = daymet_backfill.sort_index()
            daymet_backfill.index = pd.to_datetime(daymet_backfill.index).normalize()

            print("Loading GridMET backfill CSV...")
            gridmet_backfill = pd.read_csv(GRIDMET_BACKFILL_PATH, index_col=0, parse_dates=True).iloc[:, 0]
            gridmet_backfill = gridmet_backfill.sort_index()
            gridmet_backfill.index = pd.to_datetime(gridmet_backfill.index).normalize()
            print(f"GridMET backfill loaded: {len(gridmet_backfill)} rows")

            print("Loading ORNL backfill CSV...")
            ornl_daily_mean = None
            ornl_daily_median = None
            if os.path.exists(ORNL_PATH):
                try:
                    ornl = pd.read_csv(ORNL_PATH, index_col=0, parse_dates=True)
                    ornl.index = pd.to_datetime(ornl.index).normalize()
                    ornl_daily_mean = ornl.iloc[:, 1:].mean(axis=1)  # mean across all model columns
                    ornl_daily_median = ornl.iloc[:, 1:].median(axis=1)  # median across all model columns
                    print(f"ORNL backfill loaded: {len(ornl_daily_mean)} rows")
                except Exception as e:
                    print(f"Could not load ORNL data: {e}")
            
            # Combine Daymet from vault and backfill
            print("Combining Daymet sources...")
            daymet_full = daymet_backfill
            
            print("  Pre-computed data load complete.")
        except Exception as e:
            print(f"Could not load pre-computed data: {e}")

    years = sorted(df['date'].dt.year.unique())
    total_events = len(df)
    processed = 0

    for year in years:
        year_mask = df['date'].dt.year == year
        year_dates = df.loc[year_mask, 'date']
        print(f"Processing Year {year} ({len(year_dates)} events)...")
        
        # --- Pre-calculate PRISM (Backfill missing 2014-2020 data and historical missing data) ---
        prism_daily = None
        
        prism_path_800 = os.path.join(VAULT_DIR, "prism_ppt_800m", f"{year}-01-01_{year}-12-31_daily_800m_PRISM_data.zarr")
        dec = get_decade_folder(year)
        prism_path_4k = os.path.join(VAULT_DIR, "PRISM", dec, f"{year}-01-01_{year}-12-31_daily_4km_PRISM_data.zarr")
        
        prism_path = prism_path_800 if os.path.exists(prism_path_800) else prism_path_4k

        if os.path.exists(prism_path):
            try:
                ds_p = xr.open_zarr(prism_path, consolidated=False)
                spatial_dims = [d for d in ds_p['ppt'].dims if d not in ['time']]
                prism_daily = ds_p['ppt'].mean(dim=spatial_dims, skipna=True).compute().to_series()
                prism_daily.index = prism_daily.index.normalize()
                ds_p.close()
            except Exception as e:
                pass

        # --- Pre-calculate Daymet ---
        daymet_daily = None
        # Try to use backfill/full series first
        if daymet_backfill is not None:
            daymet_daily = daymet_backfill.loc[pd.Timestamp(year,1,1)-pd.Timedelta(days=2) : pd.Timestamp(year,12,31)+pd.Timedelta(days=2)]
        
        # If year not in backfill (e.g. historical), fallback to vault
        if daymet_daily is None or daymet_daily.empty:
            ornl_path = get_ornl_zarr(year)
            if ornl_path:
                try:
                    is_cons = (1981 <= year <= 2011)
                    ds_d = xr.open_zarr(ornl_path, consolidated=is_cons)
                    spatial_dims = [d for d in ds_d['prcp'].dims if d not in ['time']]
                    time_mask = (ds_d.time.dt.year == year)
                    ds_d_year = ds_d.isel(time=time_mask)
                    daymet_daily = ds_d_year['prcp'].mean(dim=spatial_dims, skipna=True).compute().to_series()
                    daymet_daily.index = daymet_daily.index.normalize()
                    if not is_cons: ds_d.close()
                except Exception: pass
                
        # --- Pre-calculate GridMET ---
        gridmet_daily = None
        if gridmet_backfill is not None:
            # GridMET backfill covers 1981-2025, but boundary events need buffer
            gridmet_daily = gridmet_backfill.loc[pd.Timestamp(year,1,1)-pd.Timedelta(days=2) : pd.Timestamp(year,12,31)+pd.Timedelta(days=2)]
            if year == 2024:
                print(f"  2024 GridMET daily loaded: {len(gridmet_daily)} rows")
            
        if gridmet_daily is None or gridmet_daily.empty:
            g_file = os.path.join(VAULT_DIR, "gridmet", f"gridmet_skagit_{year}.nc")
            if os.path.exists(g_file):
                try:
                    ds_g = xr.open_dataset(g_file)
                    gridmet_daily = ds_g['precipitation_amount'].mean(dim=['lat', 'lon'], skipna=True).compute().to_series()
                    gridmet_daily.index = gridmet_daily.index.normalize()
                    ds_g.close()
                except Exception: pass

        # --- Pre-calculate HRRR for the Year ---
        hrrr_daily = None
        if year >= 2014:
            hrrr_paths = []
            for m in range(1, 13):
                p = os.path.join(VAULT_DIR, "weather_data", f"{year}-{m:02d}_HRRR_data.zarr")
                if os.path.exists(p): hrrr_paths.append(p)
                p_fixed = os.path.join(VAULT_DIR, "weather_data", f"{year}-{m:02d}_HRRR_data_fixed_time.zarr")
                if os.path.exists(p_fixed): hrrr_paths.append(p_fixed)
            
            # To handle boundary events at the end of the year, append next year's January if available
            next_jan = os.path.join(VAULT_DIR, "weather_data", f"{year+1}-01_HRRR_data.zarr")
            if os.path.exists(next_jan): hrrr_paths.append(next_jan)
            
            if hrrr_paths:
                series_list = []
                for p in hrrr_paths:
                    try:
                        ds_h = xr.open_zarr(p, consolidated=False)
                        var_name = 'APCP_sfc' if 'APCP_sfc' in ds_h.data_vars else 'tp'
                        if var_name in ds_h.data_vars:
                            hrrr_spatial_dims = [d for d in ds_h[var_name].dims if d not in ['time']]  # ADD THIS LINE
                            s = ds_h[var_name].mean(dim=hrrr_spatial_dims, skipna=True).compute().to_series()
                            s = s.sort_index().resample('1D').sum() / 6.0
                            series_list.append(s)
                        ds_h.close()
                    except Exception as e:
                        print(f"  Failed HRRR extraction for chunk {p}: {e}")
                
                if series_list:
                    hrrr_daily = pd.concat(series_list).sort_index()
                    hrrr_daily = hrrr_daily[~hrrr_daily.index.duplicated(keep='first')]
                    hrrr_daily.index = hrrr_daily.index.normalize()


        # --- Pre-calculate UCLA (Water-Year based files: Sept to Aug) ---
        ucla_daily = None
        try:
            # Calendar year X requires:
            # 1. (X-1).nc for Jan-Aug
            # 2. (X).nc   for Sept-Dec
            u_paths = [
                os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year-1}.nc"),
                os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year}.nc")
            ]
            series_list = []
            for p in u_paths:
                if os.path.exists(p):
                    ds_u = xr.open_dataset(p)
                    u_var = "prec" if "prec" in ds_u.data_vars else "pr"
                    spatial_dims = [d for d in ds_u[u_var].dims if d not in ['time', 'day']]
                    s = ds_u[u_var].mean(dim=spatial_dims, skipna=True).compute().to_series()
                    if 'day' in ds_u.coords and not pd.api.types.is_datetime64_any_dtype(s.index):
                        s.index = pd.to_datetime(ds_u['day'].values)
                    s.index = s.index.normalize()
                    series_list.append(s)
                    ds_u.close()
            
            if series_list:
                ucla_daily = pd.concat(series_list).sort_index()
                ucla_daily = ucla_daily[~ucla_daily.index.duplicated(keep='first')]
        except Exception as e:
            print(f"  UCLA Error for {year}: {e}")

        # Process each event
        for idx, row_dt in year_dates.items():
            dt_norm = row_dt.normalize()
            start_date = dt_norm - pd.Timedelta(days=1)
            end_date = dt_norm + pd.Timedelta(days=1)
            
            # Extract PRISM if missing
            if pd.isna(df.at[idx, 'prism_3d_tot']) and prism_daily is not None:
                window = prism_daily.loc[start_date:end_date]
                if len(window) == 3:
                    df.at[idx, 'prism_3d_tot'] = window.sum()
                    df.at[idx, 'prism_3d_max'] = window.max()

            # Extract Daymet
            if daymet_daily is not None:
                window = daymet_daily.loc[start_date:end_date]
                if len(window) == 3:
                    df.at[idx, 'daymet_3d_tot'] = window.sum()
                    df.at[idx, 'daymet_3d_max'] = window.max()

            # Extract CONUS
            if conus_daily is not None:
                try:
                    window = conus_daily.loc[start_date:end_date]
                    if len(window) == 3:
                        df.at[idx, 'conus_3d_tot'] = window.sum()
                        df.at[idx, 'conus_3d_max'] = window.max()
                except Exception as e:
                    pass
                
            # Extract HRRR
            if hrrr_daily is not None:
                try:
                    window = hrrr_daily.loc[start_date:end_date]
                    if len(window) == 3:
                        df.at[idx, 'hrrr_3d_tot'] = window.sum()
                        df.at[idx, 'hrrr_3d_max'] = window.max()
                except Exception as e:
                    pass

            # Extract GridMET
            if gridmet_daily is not None:
                window = gridmet_daily.loc[start_date:end_date]
                if len(window) == 3:
                    val = window.sum()
                    df.at[idx, 'gridmet_3d_tot'] = val
                    df.at[idx, 'gridmet_3d_max'] = window.max()
                else:
                    if processed < 10 or processed % 100 == 0:
                        print(f"  GridMET window incomplete for {dt_norm}: length {len(window)}")

            # Extract ORNL (mean)
            if ornl_daily_mean is not None:
                window = ornl_daily_mean.loc[start_date:end_date]
                if len(window) == 3:
                    df.at[idx, 'ornl_mean_3d_tot'] = window.sum()
                    df.at[idx, 'ornl_mean_3d_max'] = window.max()

            # Extract ORNL (median)
            if ornl_daily_median is not None:
                window = ornl_daily_median.loc[start_date:end_date]
                if len(window) == 3:
                    df.at[idx, 'ornl_median_3d_tot'] = window.sum()
                    df.at[idx, 'ornl_median_3d_max'] = window.max()

            # Extract UCLA
            if ucla_daily is not None:
                try:
                    window = ucla_daily.loc[start_date:end_date]
                    if len(window) == 3:
                        df.at[idx, 'ucla_3d_tot'] = window.sum()
                        df.at[idx, 'ucla_3d_max'] = window.max()
                    else:
                        if processed < 10 or processed % 100 == 0:
                            print(f"  UCLA window incomplete for {dt_norm}: length {len(window)}")
                except Exception as e:
                    pass            
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed}/{total_events} events...")

    print("\nCalculating Multi-Product Biases...")
    df['bias_tot_PNNL_PRISM'] = df['pnnl_3d_tot'] - df['prism_3d_tot']
    df['bias_max_PNNL_PRISM'] = df['pnnl_3d_max'] - df['prism_3d_max']
    df['bias_tot_PRISM_ORNL'] = df['prism_3d_tot'] - df['daymet_3d_tot']
    df['bias_max_PRISM_ORNL'] = df['prism_3d_max'] - df['daymet_3d_max']
    df['bias_tot_PRISM_CONUS'] = df['prism_3d_tot'] - df['conus_3d_tot']
    df['bias_max_PRISM_CONUS'] = df['prism_3d_max'] - df['conus_3d_max']
    df['bias_tot_PRISM_HRRR'] = df['prism_3d_tot'] - df['hrrr_3d_tot']
    df['bias_max_PRISM_HRRR'] = df['prism_3d_max'] - df['hrrr_3d_max']
    df['bias_tot_PRISM_UCLA'] = df['prism_3d_tot'] - df['ucla_3d_tot']
    df['bias_max_PRISM_UCLA'] = df['prism_3d_max'] - df['ucla_3d_max']
    df['bias_tot_PRISM_GRIDMET'] = df['prism_3d_tot'] - df['gridmet_3d_tot']
    df['bias_max_PRISM_GRIDMET'] = df['prism_3d_max'] - df['gridmet_3d_max']
    df['bias_tot_PRISM_ORNL_mean'] = df['prism_3d_tot'] - df['ornl_mean_3d_tot']
    df['bias_max_PRISM_ORNL_mean'] = df['prism_3d_max'] - df['ornl_mean_3d_max']
    df['bias_tot_PRISM_ORNL_median'] = df['prism_3d_tot'] - df['ornl_median_3d_tot']
    df['bias_max_PRISM_ORNL_median'] = df['prism_3d_max'] - df['ornl_median_3d_max']

    print(f"Saving enriched dataset to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("Done!")

if __name__ == "__main__":
    enhance_bulk()
