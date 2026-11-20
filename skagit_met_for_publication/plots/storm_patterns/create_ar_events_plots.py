"""Plot hourly and 3-hour precipitation with USGS discharge for selected AR events.

For each AR event in ``date_ranges`` (about 8-15 days around the event date),
two figures are created:

* ``ar_event_NN_hourly.png``: hourly basin-mean precipitation [mm/h] from
  CONUS404, UCLA (WRF-ERA5 d02) and PNNL (WRF), with USGS daily discharge
  [m^3/s] on a secondary y-axis.
* ``ar_event_NN_3hr_cumulative.png``: the same precipitation summed over
  3-hour blocks [mm], with discharge.

Basin mean: mean over the model grid cells whose centres fall inside the
Skagit basin boundary (SkagitBoundary.json). UCLA hourly precipitation is the
hourly increment of accumulated RAINC + RAINNC. Hourly gaps in CONUS404 and
PNNL are filled with NaN so lines are not drawn across missing hours.

Inputs:
    <BASE_DIR>/data/GIS/SkagitBoundary.json         basin boundary
    <BASE_DIR>/data/usgs_12200500_discharge.rdb     USGS 12200500 daily discharge
    External hourly model data - see "PATHS - external" below.

Outputs (in this script's folder):
    ar_event_NN_hourly.png, ar_event_NN_3hr_cumulative.png  (NN = 01..06)

Usage:
    python create_ar_events_plots.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared hourly loaders (common/prism_day.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import prism_day  # noqa: E402

# ============================================================================
# PATHS - project files (relative to this script)
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent              # plots/storm_patterns/
BASE_DIR = SCRIPT_DIR.parents[1]                          # skagit_met_for_publication/
OUTPUT_DIR = SCRIPT_DIR
BOUNDARY_PATH = BASE_DIR / "data/GIS/SkagitBoundary.json"
DISCHARGE_PATH = BASE_DIR / "data/usgs_12200500_discharge.rdb"


# Publication font sizes (figures are 16 in wide; expect ~0.45x reduction to journal page width)
FONT_TITLE = 20
FONT_LABEL = 20
FONT_TICK = 17
FONT_LEGEND = 17

# Colors
COLOR_CONUS = '#1f77b4'
COLOR_UCLA = '#ff7f0e'
COLOR_PNNL = '#2ca02c'
COLOR_DISCHARGE = '#777777'

# Other settings
CFS_TO_CMS = 0.0283168   # 1 ft^3/s = 0.0283168 m^3/s
DPI = 150                # resolution of saved figures

# Helper functions
def load_usgs_rdb(filepath, param_name):
    """Load USGS RDB files."""
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

# USGS daily values are means over the local calendar day (Pacific Standard Time, UTC-8),
# while the precipitation axis is in UTC: local day D runs from D 08:00 to D+1 08:00 UTC.
USGS_DAY_START_UTC = pd.Timedelta(hours=8)


def discharge_steps(df, start, end):
    """Daily mean discharge as a step line spanning each local day, on the UTC time axis.

    Returns (x, y) for ``ax.step(x, y, where='post')``, clipped to [start, end], or
    (None, None) if no daily value overlaps the window.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    day_start = df['date'] + USGS_DAY_START_UTC
    d = df[(day_start + pd.Timedelta(days=1) > start) & (day_start < end)]
    if d.empty:
        return None, None
    x = list((d['date'] + USGS_DAY_START_UTC).clip(lower=start)) + [end]
    y = list(d['discharge_cms']) + [d['discharge_cms'].iloc[-1]]
    return x, y


def fill_missing_times(times, values):
    """Fill missing time gaps with NaN values to prevent line connections."""
    if len(times) == 0:
        return times, values

    # Create a DataFrame with times and values
    df = pd.DataFrame({'value': values}, index=pd.to_datetime(times))

    # Remove duplicate timestamps (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]

    # Sort by time
    df = df.sort_index()

    # Resample to 1 hour, filling missing hours with NaN
    df_filled = df.asfreq('1h')

    return df_filled.index.values, df_filled['value'].values

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================


# Load discharge data
print("\n" + "=" * 70)
print("LOADING DISCHARGE DATA")
print("=" * 70)
df_discharge_all = load_usgs_rdb(DISCHARGE_PATH, 'discharge_cfs')
# Convert cfs to cms (1 cfs = 0.0283168 cms)
df_discharge_all['discharge_cms'] = df_discharge_all['discharge_cfs'] * CFS_TO_CMS
print(f"Loaded {len(df_discharge_all)} discharge records")

# ============================================================================
# CREATE PLOTS FOR ALL AR EVENTS
# ============================================================================
print("\n" + "=" * 70)
print("CREATING PLOTS FOR ALL AR EVENTS")
print("=" * 70)

# Date ranges for AR events (date_start_ucla, date_start_conus_pnnl, date_end, event_date, ar_category)
date_ranges = [
    (datetime(1990, 11, 20, 23, 0, 0), datetime(1990, 11, 21, 0, 0, 0), datetime(1990, 11, 29, 23, 0, 0), "1990-11-24", "5.0"),
    (datetime(1995, 11, 26, 23, 0, 0), datetime(1995, 11, 27, 0, 0, 0), datetime(1995, 12, 5, 23, 0, 0), "1995-11-30", "4.0"),
    (datetime(1990, 11, 9, 23, 0, 0), datetime(1990, 11, 10, 0, 0, 0), datetime(1990, 11, 18, 23, 0, 0), "1990-11-13", "4.0"),
    (datetime(2006, 11, 3, 23, 0, 0), datetime(2006, 11, 4, 0, 0, 0), datetime(2006, 11, 12, 23, 0, 0), "2006-11-07", "5.0"),
    (datetime(2003, 10, 13, 23, 0, 0), datetime(2003, 10, 14, 0, 0, 0), datetime(2003, 10, 28, 23, 0, 0), "2003-10-21", "5.0"),
    (datetime(2021, 11, 12, 23, 0, 0), datetime(2021, 11, 13, 0, 0, 0), datetime(2021, 11, 20, 23, 0, 0), "2021-11-15", "4.0"),
]

def load_data_for_period(date_start, date_end, date_start_ucla=None):
    """Basin-mean hourly precipitation [mm/h] of CONUS404, UCLA and PNNL for a period.

    Hourly data come from common/prism_day.load_hourly: every value is the precipitation
    of the hour ending at its UTC stamp, for all three products. The basin mean is taken
    over the cells whose centre is inside SkagitBoundary.json. ``date_start_ucla`` is
    accepted for compatibility and not used (load_hourly reads the extra UCLA hour itself).
    Returns (conus, times_conus, ucla, times_ucla, pnnl, times_pnnl); None where no data.
    """
    out = []
    for product in ("conus", "ucla", "pnnl"):
        da = prism_day.load_hourly(product, date_start, date_end)
        if da is None:
            out += [None, None]
            continue
        series = prism_day.basin_mean(da, prism_day.boundary_mask(da))
        out += [series.values, series.index.values]
    return tuple(out)

# Loop through each date range and create plots
for idx, (date_start_ucla, date_start, date_end, event_date, ar_category) in enumerate(date_ranges, 1):
    print(f"\nProcessing event {idx}/{len(date_ranges)}: {event_date} (AR={ar_category})")

    # Load data for this period
    spatial_mean_conus, times_conus, spatial_mean_ucla, times_ucla, spatial_mean_pnnl, times_pnnl = \
        load_data_for_period(date_start, date_end, date_start_ucla)

    # Compute 3-hour cumulatives BEFORE filling missing times (needed for proper resampling)
    cumul_3hr_conus = None
    cumul_3hr_ucla = None
    cumul_3hr_pnnl = None

    if spatial_mean_conus is not None:
        df_conus = pd.DataFrame({'precip': spatial_mean_conus}, index=pd.to_datetime(times_conus))
        cumul_3hr_conus = df_conus['precip'].resample('3h').sum()

    if spatial_mean_ucla is not None:
        df_ucla = pd.DataFrame({'precip': spatial_mean_ucla}, index=pd.to_datetime(times_ucla))
        cumul_3hr_ucla = df_ucla['precip'].resample('3h').sum()

    if spatial_mean_pnnl is not None:
        df_pnnl = pd.DataFrame({'precip': spatial_mean_pnnl}, index=pd.to_datetime(times_pnnl))
        cumul_3hr_pnnl = df_pnnl['precip'].resample('3h').sum()

    # Fill missing times with NaN to prevent line connections across gaps (for hourly plots only)
    # Note: Skip UCLA because it's already processed through diff() which modifies the time structure
    if spatial_mean_conus is not None:
        times_conus, spatial_mean_conus = fill_missing_times(times_conus, spatial_mean_conus)
    if spatial_mean_pnnl is not None:
        times_pnnl, spatial_mean_pnnl = fill_missing_times(times_pnnl, spatial_mean_pnnl)

    # ========================================================================
    # HOURLY PLOT
    # ========================================================================
    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot precipitation (filter out NaN values)
    if spatial_mean_conus is not None:
        mask_conus = ~np.isnan(spatial_mean_conus)
        ax.plot(times_conus[mask_conus], spatial_mean_conus[mask_conus], marker='o', linewidth=2.5, markersize=4,
                color=COLOR_CONUS, label='CONUS404', alpha=0.8)

    if spatial_mean_ucla is not None:
        mask_ucla = ~np.isnan(spatial_mean_ucla)
        ax.plot(times_ucla[mask_ucla], spatial_mean_ucla[mask_ucla], marker='s', linewidth=2.5, markersize=4,
                color=COLOR_UCLA, label='UCLA', alpha=0.8)

    if spatial_mean_pnnl is not None:
        mask_pnnl = ~np.isnan(spatial_mean_pnnl)
        ax.plot(times_pnnl[mask_pnnl], spatial_mean_pnnl[mask_pnnl], marker='^', linewidth=2.5, markersize=4,
                color=COLOR_PNNL, label='PNNL', alpha=0.8)

    # Create secondary y-axis for discharge
    ax2 = ax.twinx()
    q_x, q_y = discharge_steps(df_discharge_all, date_start, date_end + timedelta(hours=1))

    if q_x is not None:
        ax2.step(q_x, q_y, where='post',
                linewidth=2.5, color=COLOR_DISCHARGE, label='USGS Discharge (cms)', alpha=0.7)
        ax2.set_ylabel('Discharge (cms)', fontsize=FONT_LABEL, fontweight='bold', color=COLOR_DISCHARGE)
        ax2.tick_params(axis='y', labelcolor=COLOR_DISCHARGE, labelsize=FONT_TICK)

    # Formatting
    ax.set_xlabel('Time', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Hourly Precipitation (mm/hour)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_title(f'Spatial Mean Hourly Precipitation - Skagit Basin ({event_date}, AR={ar_category})\n{date_start.strftime("%Y-%m-%d")} to {date_end.strftime("%Y-%m-%d")}',
                fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=FONT_LEGEND, loc='upper right', framealpha=0.95)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save hourly plot
    output_path = OUTPUT_DIR / f"ar_event_{idx:02d}_hourly.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"  Hourly plot saved to: {output_path}")
    plt.close()

    # ========================================================================
    # 3-HOUR CUMULATIVE PLOT
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(16, 7))

    # Plot 3-hour cumulatives (already computed before filling missing times)
    if cumul_3hr_conus is not None:
        mask_3h = ~np.isnan(cumul_3hr_conus.values)
        ax2.plot(cumul_3hr_conus.index[mask_3h], cumul_3hr_conus.values[mask_3h], marker='o', linewidth=2.5, markersize=6,
                color=COLOR_CONUS, label='CONUS404', alpha=0.8)

    if cumul_3hr_ucla is not None:
        mask_3h = ~np.isnan(cumul_3hr_ucla.values)
        ax2.plot(cumul_3hr_ucla.index[mask_3h], cumul_3hr_ucla.values[mask_3h], marker='s', linewidth=2.5, markersize=6,
                color=COLOR_UCLA, label='UCLA', alpha=0.8)

    if cumul_3hr_pnnl is not None:
        mask_3h = ~np.isnan(cumul_3hr_pnnl.values)
        ax2.plot(cumul_3hr_pnnl.index[mask_3h], cumul_3hr_pnnl.values[mask_3h], marker='^', linewidth=2.5, markersize=6,
                color=COLOR_PNNL, label='PNNL', alpha=0.8)

    # Create secondary y-axis for discharge
    ax2_2 = ax2.twinx()
    if q_x is not None:
        ax2_2.step(q_x, q_y, where='post',
                  linewidth=2.5, color=COLOR_DISCHARGE, label='USGS Discharge (cms)', alpha=0.7)
        ax2_2.set_ylabel('Discharge (cms)', fontsize=FONT_LABEL, fontweight='bold', color=COLOR_DISCHARGE)
        ax2_2.tick_params(axis='y', labelcolor=COLOR_DISCHARGE, labelsize=FONT_TICK)

    # Formatting
    ax2.set_xlabel('Time', fontsize=FONT_LABEL, fontweight='bold')
    ax2.set_ylabel('3-Hour Cumulative\nPrecipitation (mm)', fontsize=FONT_LABEL, fontweight='bold')
    ax2.set_title(f'3-Hour Cumulative Precipitation - Skagit Basin ({event_date}, AR={ar_category})\n{date_start.strftime("%Y-%m-%d")} to {date_end.strftime("%Y-%m-%d")}',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=FONT_TICK)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=FONT_LEGEND, loc='upper right', framealpha=0.95)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save 3-hour plot
    output_path_3hr = OUTPUT_DIR / f"ar_event_{idx:02d}_3hr_cumulative.png"
    plt.savefig(output_path_3hr, dpi=DPI, bbox_inches='tight')
    print(f"  3-hour plot saved to: {output_path_3hr}")
    plt.close()

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
