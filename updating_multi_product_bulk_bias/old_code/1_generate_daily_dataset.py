import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import glob

# ============================================================
# PATHS
# ============================================================
BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
EXP_DATA_DIR = os.path.join(BASE_DIR, "experiments_2/data")
HYDRO_Q_PATH = os.path.join(EXP_DATA_DIR, "usgs_12200500_discharge.rdb")
HYDRO_H_PATH = os.path.join(EXP_DATA_DIR, "usgs_12200500_gage_height.rdb")
GRIDMET_DIR = "/data0/skagit_met/data_transfer/data/gridmet/"
SNOTEL_DATA_DIR = "/data0/skagit_met/data_transfer/data/snotel/"
PRISM_DATA_DIR = "/data0/skagit_met/data_transfer/data/prism_ppt/"
OUTPUT_DIR = os.path.join("/data0/hernanqd/plots_code/multi_product_bulk_bias_data/outputs")

# ============================================================
# LOADERS
# ============================================================

def load_usgs_rdb(filepath, param_name):
    print(f"Loading {param_name} from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep='\t', comment='#')
        df = df.iloc[1:].reset_index(drop=True)
        val_cols = [c for c in df.columns if '_00' in c and not c.endswith('_cd')]
        if not val_cols: return pd.DataFrame(columns=['date', param_name])
        val_col = val_cols[0]
        df = df[['datetime', val_col]]
        df.columns = ['date', param_name]
        df['date'] = pd.to_datetime(df['date'])
        df[param_name] = pd.to_numeric(df[param_name], errors='coerce')
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame(columns=['date', param_name])

def load_ar_events(filepath):
    cols = [
        'start_date_str', 'max_date_str', 'end_date_str', 'duration', 
        'max_ivt', 'ar_scale', 'avg_ivt', 'max_ivt_dir', 'avg_ivt_dir', 
        'max_iwv', 'avg_iwv', 'avg_uivt', 'avg_vivt'
    ]
    try:
        df = pd.read_csv(filepath, names=cols)
        for c in ['start_date_str', 'max_date_str', 'end_date_str']:
            df[c.replace('_str', '')] = pd.to_datetime(df[c], format='%Y%m%d%H', errors='coerce')
        return df
    except Exception as e:
        print(f"  Error loading AR data {filepath}: {e}")
        return pd.DataFrame()

def load_gridmet_precip(start_year, end_year):
    print(f"Loading GridMET precipitation...")
    datasets = []
    for yr in range(start_year, end_year + 1):
        fpath = os.path.join(GRIDMET_DIR, f"pr_{yr}.nc")
        if os.path.exists(fpath):
            try:
                ds = xr.open_dataset(fpath)
                if 'day' in ds.coords:
                    ds = ds.rename({'day': 'time'})
                datasets.append(ds['precipitation_amount'])
            except Exception as e:
                print(f"  Error loading {fpath}: {e}")
    
    if not datasets:
        return pd.DataFrame(columns=['date', 'gridmet_precip_mm'])
    
    combined = xr.concat(datasets, dim='time')
    basin_mean = combined.mean(dim=['lat', 'lon']).to_dataframe().reset_index()
    basin_mean = basin_mean.rename(columns={'time': 'date', 'precipitation_amount': 'gridmet_precip_mm'})
    basin_mean['date'] = pd.to_datetime(basin_mean['date']).dt.tz_localize(None)
    return basin_mean

def load_prism_precip():
    print(f"Loading PRISM precipitation from {PRISM_DATA_DIR}...")
    all_dfs = []
    # Match sequential blocks and yearly blocks
    zarr_paths = glob.glob(os.path.join(PRISM_DATA_DIR, "*PRISM_data.zarr"))
    for zp in sorted(zarr_paths):
        try:
            ds = xr.open_zarr(zp)
            if 'ppt' in ds.data_vars:
                # Average over basin (lat/lon)
                df = ds['ppt'].mean(dim=['lat', 'lon']).to_dataframe().reset_index()
                df = df.rename(columns={'time': 'date', 'ppt': 'prism_precip_mm'})
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                all_dfs.append(df[['date', 'prism_precip_mm']])
        except Exception as e:
            print(f"  Warning: Could not open {zp}: {e}")
    
    if not all_dfs:
        return pd.DataFrame(columns=['date', 'prism_precip_mm'])
    
    combined = pd.concat(all_dfs)
    basin_avg = combined.groupby('date')['prism_precip_mm'].mean().reset_index()
    return basin_avg

def load_local_snotel_precip():
    print(f"Loading local SNOTEL zarrs...")
    all_dfs = []
    zarr_paths = glob.glob(os.path.join(SNOTEL_DATA_DIR, "*SNOTEL_daily_data.zarr"))
    for zp in sorted(zarr_paths):
        try:
            ds = xr.open_zarr(zp)
            if 'PRECIPITATION' in ds.data_vars:
                df = ds['PRECIPITATION'].mean(dim='site').to_dataframe().reset_index()
                df = df.rename(columns={'time': 'date', 'PRECIPITATION': 'snotel_precip_in'})
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                all_dfs.append(df[['date', 'snotel_precip_in']])
        except Exception as e:
            pass
    
    if not all_dfs:
        return pd.DataFrame(columns=['date', 'snotel_precip_in'])
    
    combined = pd.concat(all_dfs)
    basin_avg = combined.groupby('date')['snotel_precip_in'].mean().reset_index()
    return basin_avg

# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Create Base Timeline (1980-2025)
    dates = pd.date_range(start="1980-01-01", end="2025-12-31", freq='D')
    final_df = pd.DataFrame({'date': dates})

    # 2. Load Hydrology
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')
    h_df = load_usgs_rdb(HYDRO_H_PATH, 'gage_height_ft')
    hydro_df = pd.merge(q_df, h_df, on='date', how='outer')
    final_df = pd.merge(final_df, hydro_df, on='date', how='left')

    # 3. Load AR Data by Station
    points = {
        '475_1245': '47.5N_124.5W',
        '480_1245': '48.0N_124.5W',
        '485_1245': '48.5N_124.5W'
    }
    
    for p_id, p_name in points.items():
        fname = os.path.join(EXP_DATA_DIR, f"ar_event_data_{p_id}.csv")
        if os.path.exists(fname):
            print(f"Processing AR data for {p_name}...")
            ar_df = load_ar_events(fname)
            if ar_df.empty:
                final_df[f'ar_scale_{p_name}'] = 0.0
                continue
                
            daily_list = []
            for _, row in ar_df.iterrows():
                if pd.isna(row['start_date']) or pd.isna(row['end_date']): continue
                curr = row['start_date'].date()
                while curr <= row['end_date'].date():
                    daily_list.append({'date': pd.to_datetime(curr), f'ar_scale_{p_name}': row['ar_scale']})
                    curr += timedelta(days=1)
            
            if daily_list:
                daily_max = pd.DataFrame(daily_list).groupby('date')[f'ar_scale_{p_name}'].max().reset_index()
                final_df = pd.merge(final_df, daily_max, on='date', how='left').fillna({f'ar_scale_{p_name}': 0})
            else:
                final_df[f'ar_scale_{p_name}'] = 0.0

    # 4. Load Precipitation
    # PRISM (Breadth 1981-2024)
    prism_df = load_prism_precip()
    final_df = pd.merge(final_df, prism_df, on='date', how='left')
    
    # GridMET (High detail 5 years)
    gridmet_df = load_gridmet_precip(1980, 2025)
    final_df = pd.merge(final_df, gridmet_df, on='date', how='left')
    
    # SNOTEL (Local Archive 2014-2023)
    snotel_precip = load_local_snotel_precip()
    final_df = pd.merge(final_df, snotel_precip, on='date', how='left')

    # 5. Save Final Dataset
    output_file = os.path.join(OUTPUT_DIR, "skagit_daily_integrated_dataset_1980_2025.csv")
    final_df.to_csv(output_file, index=False)
    
    print(f"\nFinal dataset created successfully: {output_file}")
    print(f"Columns: {final_df.columns.tolist()}")
    print(f"Total rows: {len(final_df)}")
    
    # Quick sanity check
    print(f"Data Presence:")
    print(f"  Q: {final_df['discharge_cfs'].count()} rows")
    print(f"  PRISM: {final_df['prism_precip_mm'].count()} rows")
    print(f"  GridMET: {final_df['gridmet_precip_mm'].count()} rows")
    print(f"  SNOTEL: {final_df['snotel_precip_in'].count()} rows")

if __name__ == "__main__":
    main()
