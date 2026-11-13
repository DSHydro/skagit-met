"""Generate an integrated daily dataset combining hydrology and atmospheric river (AR) data.

Builds a continuous daily timeline (1980-01-01 to 2025-12-31) and merges:

* USGS gage 12200500 (Skagit River near Mount Vernon, WA): daily mean
  discharge [cfs] and gage height [ft].
* AR event catalogs at three coastal grid points (47.5N, 48.0N and 48.5N
  at 124.5W). Each event is assigned to every calendar day from its start
  date to its end date (inclusive). If two or more events overlap on the
  same day, the minimum AR scale is kept. Days with no AR are set to 0.

Inputs (in ``<BASE_DIR>/data``):
    usgs_12200500_discharge.rdb     USGS daily values, parameter 00060
    usgs_12200500_gage_height.rdb   USGS daily values, parameter 00065
    ar_event_data_<point_id>.csv    AR event catalog, one file per grid point

Output (in ``<BASE_DIR>/analysis/outputs``):
    1_skagit_daily_integrated_dataset_1980_2025.csv
        date, discharge_cfs, gage_height_ft,
        ar_scale_47.5N_124.5W, ar_scale_48.0N_124.5W, ar_scale_48.5N_124.5W

Usage:
    python 1_generate_daily_dataset.py
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
# Project root = parent of the folder containing this script (i.e. skagit_met_for_publication/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DATA_DIR = os.path.join(BASE_DIR, "data")
HYDRO_Q_PATH = os.path.join(EXP_DATA_DIR, "usgs_12200500_discharge.rdb")
HYDRO_H_PATH = os.path.join(EXP_DATA_DIR, "usgs_12200500_gage_height.rdb")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "1_skagit_daily_integrated_dataset_1980_2025.csv")

# ============================================================
# SETTINGS
# ============================================================
START_DATE = "1980-01-01"
END_DATE = "2025-12-31"

# AR event file ID -> grid point label used in output column names
AR_POINTS = {
    '475_1245': '47.5N_124.5W',
    '480_1245': '48.0N_124.5W',
    '485_1245': '48.5N_124.5W'
}

# Column names of the AR event files (files have no header row)
AR_COLUMNS = [
    'start_date_str', 'max_date_str', 'end_date_str', 'duration',
    'max_ivt', 'ar_scale', 'avg_ivt', 'max_ivt_dir', 'avg_ivt_dir',
    'max_iwv', 'avg_iwv', 'avg_uivt', 'avg_vivt'
]

# ============================================================
# LOADERS
# ============================================================

def load_usgs_rdb(filepath, param_name):
    """Read a USGS daily-values RDB (tab-delimited) file.

    Parameters
    ----------
    filepath : str
        Path to the RDB file.
    param_name : str
        Name given to the value column in the returned DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns ['date', param_name]. Non-numeric values become NaN.
    """
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
    """Read an AR event catalog (CSV without header).

    Parameters
    ----------
    filepath : str
        Path to the AR event file; columns are given by AR_COLUMNS.

    Returns
    -------
    pandas.DataFrame
        One row per AR event, with the start/peak/end times (YYYYMMDDHH)
        also parsed into datetime columns 'start_date', 'max_date', 'end_date'.
    """
    try:
        df = pd.read_csv(filepath, names=AR_COLUMNS)
        for c in ['start_date_str', 'max_date_str', 'end_date_str']:
            df[c.replace('_str', '')] = pd.to_datetime(df[c], format='%Y%m%d%H', errors='coerce')
        return df
    except Exception as e:
        print(f"  Error loading AR data {filepath}: {e}")
        return pd.DataFrame()

# ============================================================
# MAIN
# ============================================================

def main():
    """Build the integrated daily dataset and save it to OUTPUT_FILE."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Create Base Timeline
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    final_df = pd.DataFrame({'date': dates})

    # 2. Load Hydrology
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')
    h_df = load_usgs_rdb(HYDRO_H_PATH, 'gage_height_ft')
    hydro_df = pd.merge(q_df, h_df, on='date', how='outer')
    final_df = pd.merge(final_df, hydro_df, on='date', how='left')

    # 3. Load AR Data by Station
    for p_id, p_name in AR_POINTS.items():
        fname = os.path.join(EXP_DATA_DIR, f"ar_event_data_{p_id}.csv")
        if os.path.exists(fname):
            print(f"Processing AR data for {p_name}...")
            ar_df = load_ar_events(fname)
            if ar_df.empty:
                final_df[f'ar_scale_{p_name}'] = np.nan
                continue

            daily_list = []
            for _, row in ar_df.iterrows():
                if pd.isna(row['start_date']) or pd.isna(row['end_date']): continue
                curr = row['start_date'].date()
                while curr <= row['end_date'].date():
                    daily_list.append({'date': pd.to_datetime(curr), f'ar_scale_{p_name}': row['ar_scale']})
                    curr += timedelta(days=1)

            if daily_list:
                # If events overlap on the same day, keep the minimum AR scale
                daily_ar = pd.DataFrame(daily_list).groupby('date')[f'ar_scale_{p_name}'].min().reset_index()
                final_df = pd.merge(final_df, daily_ar, on='date', how='left')
            else:
                final_df[f'ar_scale_{p_name}'] = np.nan

    # Fill 0s for all days (including those with no ar_scale data in any site)
    ar_scale_cols = [col for col in final_df.columns if col.startswith('ar_scale_')]
    for col in ar_scale_cols:
        final_df[col] = final_df[col].fillna(0)

    # 4. Save Final Dataset
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFinal dataset created successfully: {OUTPUT_FILE}")
    print(f"Columns: {final_df.columns.tolist()}")
    print(f"Total rows: {len(final_df)}")
    
    # Quick sanity check
    print(f"Data Presence:")
    print(f"  Q: {final_df['discharge_cfs'].count()} rows")

if __name__ == "__main__":
    main()
