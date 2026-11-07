"""Generate an integrated daily dataset combining hydrology and atmospheric river event data.

Creates a timeline-based dataset (1980-2025) by merging USGS discharge/gage height data
with atmospheric river event classifications from three monitoring stations (47.5N_124.5W,
48.0N_124.5W, 48.5N_124.5W). Outputs a CSV with daily records suitable for bias analysis.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

HYDRO_Q_PATH = Path("/data0/hernanqd/plots_code/skagit-met/data/usgs_12200500_discharge.rdb")


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

q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')


# ============================================================
# BASEFLOW SEPARATION - LYNE-HOLLICK METHOD
# ============================================================

def lyne_hollick_filter(discharge, alpha=0.925, passes=3):
    """
    Separate baseflow from discharge using the Lyne-Hollick digital filter.

    Uses a recursive low-pass filter applied forward and backward to the
    discharge time series to extract baseflow component.

    Parameters
    ----------
    discharge : array-like
        Daily discharge values (cfs)
    alpha : float, default=0.925
        Filter parameter (0 < alpha < 1). Higher values preserve more high-frequency
        variability in baseflow. Default 0.925 is commonly used for daily data.
    passes : int, default=3
        Number of forward-backward pass pairs to apply (increases filter strength)

    Returns
    -------
    baseflow : np.ndarray
        Separated baseflow component (cfs)
    """
    q = np.asarray(discharge, dtype=float)

    if np.all(np.isnan(q)) or len(q) == 0:
        return np.full_like(q, np.nan)

    # Initialize baseflow as the discharge
    bf = q.copy()

    # Apply forward-backward filtering multiple times
    for _ in range(passes):
        # Forward pass
        bf_fwd = np.full_like(bf, np.nan)
        valid_idx = ~np.isnan(bf)

        if np.any(valid_idx):
            first_valid = np.where(valid_idx)[0][0]
            bf_fwd[first_valid] = bf[first_valid]

            for i in range(first_valid + 1, len(bf)):
                if ~np.isnan(bf[i]):
                    bf_fwd[i] = alpha * bf_fwd[i-1] + (1 - alpha) * bf[i]
                else:
                    bf_fwd[i] = np.nan

        # Backward pass
        bf = np.full_like(bf_fwd, np.nan)
        valid_idx = ~np.isnan(bf_fwd)

        if np.any(valid_idx):
            last_valid = np.where(valid_idx)[0][-1]
            bf[last_valid] = bf_fwd[last_valid]

            for i in range(last_valid - 1, -1, -1):
                if ~np.isnan(bf_fwd[i]):
                    bf[i] = alpha * bf[i+1] + (1 - alpha) * bf_fwd[i]
                else:
                    bf[i] = np.nan

    return bf


# Apply Lyne-Hollick baseflow separation
if len(q_df) > 0 and 'discharge_cfs' in q_df.columns:
    print("Applying Lyne-Hollick baseflow separation...")
    q_df['baseflow_cfs'] = lyne_hollick_filter(q_df['discharge_cfs'].values)
    q_df['direct_runoff_cfs'] = q_df['discharge_cfs'] - q_df['baseflow_cfs']
    print(f"  Added baseflow_cfs and direct_runoff_cfs columns")
    print(f"  Sample:")
    print(q_df[['date', 'discharge_cfs', 'baseflow_cfs', 'direct_runoff_cfs']].head(10))

    # ============================================================
    # PLOT 1995 BASEFLOW SEPARATION
    # ============================================================
    q_df['year'] = q_df['date'].dt.year
    df_1995 = q_df[q_df['year'] == 1995].reset_index(drop=True)

    if len(df_1995) > 0:
        print("\n" + "="*60)
        print("Creating baseflow separation plot for 1995...")
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(df_1995['date'], df_1995['discharge_cfs'],
                label='Total Discharge', color='black', linewidth=1.5, zorder=3)
        ax.fill_between(df_1995['date'], 0, df_1995['baseflow_cfs'],
                        label='Baseflow', alpha=0.6, color='steelblue', zorder=2)
        ax.fill_between(df_1995['date'], df_1995['baseflow_cfs'], df_1995['discharge_cfs'],
                        label='Direct Runoff', alpha=0.6, color='coral', zorder=1)

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Discharge (cfs)', fontsize=12, fontweight='bold')
        ax.set_title('Lyne-Hollick Baseflow Separation - Skagit River at Concrete, WA (1995)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = Path("/data0/hernanqd/plots_code/skagit-met/data/baseflow_1995.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {output_path}")

        # Print statistics
        print("\n1995 Baseflow Separation Statistics:")
        print(f"  Mean Total Discharge: {df_1995['discharge_cfs'].mean():.1f} cfs")
        print(f"  Mean Baseflow: {df_1995['baseflow_cfs'].mean():.1f} cfs")
        print(f"  Mean Direct Runoff: {df_1995['direct_runoff_cfs'].mean():.1f} cfs")
        print(f"  Baseflow fraction: {df_1995['baseflow_cfs'].mean() / df_1995['discharge_cfs'].mean() * 100:.1f}%")
        print(f"  Max Discharge: {df_1995['discharge_cfs'].max():.1f} cfs")
        print(f"  Min Discharge: {df_1995['discharge_cfs'].min():.1f} cfs")
        print("="*60)

    # Save full dataset with baseflow columns to CSV
    output_csv = Path("/data0/hernanqd/plots_code/skagit-met/data/usgs_12200500_discharge_with_baseflow.csv")
    q_df_export = q_df[['date', 'discharge_cfs', 'baseflow_cfs', 'direct_runoff_cfs']].copy()
    q_df_export.to_csv(output_csv, index=False)
    print(f"\n✓ Full dataset with baseflow columns saved to:")
    print(f"  {output_csv}")