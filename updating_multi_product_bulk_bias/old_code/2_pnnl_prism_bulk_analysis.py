import os
import pandas as pd
import xarray as xr
import numpy as np
import fsspec
from datetime import datetime

# --- Paths ---
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de/multi_product_bulk_bias"
AR_DATASET_PATH = os.path.join(BASE_DIR, "outputs/skagit_daily_integrated_dataset_1980_2025.csv")
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PNNL_PARQUET_PATH = os.path.join(VAULT_DIR, "PNNL/historical/PNNL_historical.parquet")
PRISM_VAULT_DIR = os.path.join(VAULT_DIR, "PRISM")
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs/prism_pnnl_bulk_bias_data.csv")

def get_decade_folder(year: int) -> str:
    start_decade = (year // 10) * 10
    if start_decade == 1980 and year > 1980: return "1981-1990"
    if year % 10 == 0 and year > 1980: return f"{start_decade-9}-{start_decade}"
    if start_decade % 10 == 0: return f"{start_decade+1}-{start_decade+10}"
    return f"{start_decade}-{(year//10)*10+9}"

def run_bulk_analysis():
    print("Initializing Bulk Bias Analysis...")
    
    # 1. Load AR Master Catalogue
    ar_df = pd.read_csv(AR_DATASET_PATH)
    ar_df['date'] = pd.to_datetime(ar_df['date'])
    
    # Aggregated AR Scale logic
    scale_cols = [c for c in ar_df.columns if 'ar_scale_' in c]
    ar_df['ar_scale'] = ar_df[scale_cols].max(axis=1) if scale_cols else ar_df.get('ar_scale', 0.0)
    
    # 2. Sampling Strategy
    # We take all AR Scale 1-5 cases
    ar_events = ar_df[ar_df['ar_scale'] > 0].copy()
    # We take a sample of 2,000 Scale 0 cases
    non_ar_sample = ar_df[ar_df['ar_scale'] == 0].sample(n=min(2000, len(ar_df[ar_df['ar_scale']==0])), random_state=42).copy()
    
    analysis_subset = pd.concat([ar_events, non_ar_sample]).sort_values('date')
    analysis_subset = analysis_subset[(analysis_subset['date'].dt.year >= 1981) & (analysis_subset['date'].dt.year <= 2025)]
    print(f"Total analysis targets: {len(analysis_subset)} (AR: {len(ar_events)}, Sampled Scale 0: {len(non_ar_sample)})")

    # Initialize columns
    for col in ['prism_3d_tot', 'prism_3d_max', 'pnnl_3d_tot', 'pnnl_3d_max']:
        analysis_subset[col] = np.nan

    # 3. Load PNNL Mapper
    try:
        mapper = fsspec.get_mapper("reference://", fo=PNNL_PARQUET_PATH, target_protocol="file")
        ds_pnnl_all = xr.open_zarr(mapper, consolidated=False)
    except Exception as e:
        print(f"Error loading PNNL vault: {e}")
        ds_pnnl_all = None

    results = []

    # 4. Iterate by Year to keep RAM clean
    target_years = sorted(analysis_subset['date'].dt.year.unique())
    
    for year in target_years:
        print(f"  Processing Year {year}...")
        year_subset = analysis_subset[analysis_subset['date'].dt.year == year].copy()
        
        # --- PRISM Part ---
        prism_tot, prism_max = pd.Series(dtype='float64'), pd.Series(dtype='float64')
        decade_folder = get_decade_folder(year)
        zarr_name = f"{year}-01-01_{year}-12-31_daily_4km_PRISM_data.zarr"
        zarr_path = os.path.join(PRISM_VAULT_DIR, decade_folder, zarr_name)
        
        if os.path.exists(zarr_path):
            try:
                ds_p = xr.open_zarr(zarr_path, consolidated=False)
                # Basin mean daily precipitation
                spatial_dims = [d for d in ds_p.dims if d not in ['time', 'date']]
                p_daily = ds_p['ppt'].mean(dim=spatial_dims, skipna=True).to_series()
                # 3-day rolling totals/max for each date in subset
                # Note: for daily continuous we can just take the 3-day sum around the target
                # But to stay consistent with previous event analysis, we'll re-extract windows
                for dt in year_subset['date']:
                    window = p_daily.loc[dt - pd.Timedelta(days=1) : dt + pd.Timedelta(days=1)]
                    if len(window) == 3:
                        year_subset.loc[year_subset['date'] == dt, 'prism_3d_tot'] = window.sum()
                        year_subset.loc[year_subset['date'] == dt, 'prism_3d_max'] = window.max()
                ds_p.close()
            except Exception as e: print(f"    PRISM Error for {year}: {e}")

        # --- PNNL Part ---
        if ds_pnnl_all is not None and year <= 2020:
            try:
                # Optimized: find indices once per year instead of slow boolean scans on 11TB index
                time_arr = pd.to_datetime(ds_pnnl_all.time.values)
                indices = np.where(time_arr.year == year)[0]
                
                if len(indices) > 0:
                    pnnl_year = ds_pnnl_all.isel(time=indices)
                    # Pre-calculate daily averages for the basin
                    spatial_dims = [d for d in pnnl_year.dims if d not in ['time', 'date']]
                    pnnl_basin = pnnl_year['PREC_ACC_NC'].mean(dim=spatial_dims).resample(time='1D').sum().to_series()
                    # Normalize index to avoid misaligned slicing (e.g. 00:00:00 vs No Time)
                    pnnl_basin.index = pnnl_basin.index.normalize()
                    
                    for dt in year_subset['date']:
                        norm_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                        window = pnnl_basin.loc[norm_dt - pd.Timedelta(days=1) : norm_dt + pd.Timedelta(days=1)]
                        if len(window) == 3:
                            year_subset.loc[year_subset['date'] == dt, 'pnnl_3d_tot'] = window.sum()
                            year_subset.loc[year_subset['date'] == dt, 'pnnl_3d_max'] = window.max()
                        else:
                            print(f"    Missing PNNL window for {norm_dt} (found {len(window)} days)")
                else:
                    print(f"    No PNNL records found for year {year}")
            except Exception as e: print(f"    PNNL Error for {year}: {e}")
            
        # Calculate Bias for the year subset
        year_subset['bias_tot'] = year_subset['pnnl_3d_tot'] - year_subset['prism_3d_tot']
        year_subset['bias_max'] = year_subset['pnnl_3d_max'] - year_subset['prism_3d_max']
        
        results.append(year_subset)
        # Checkpoint save - every year for reliable monitoring
        pd.concat(results).to_csv(OUTPUT_CSV, index=False)

    print(f"✓ Bulk dataset successfully generated: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_bulk_analysis()
