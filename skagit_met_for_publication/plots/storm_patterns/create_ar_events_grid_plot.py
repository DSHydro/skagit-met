"""Plot a 3x2 grid of hourly precipitation with USGS discharge for six AR events.

Each panel (a-f) shows one AR event from ``date_ranges`` (about 8-15 days
around the event date): hourly basin-mean precipitation [mm/h] from CONUS404,
UCLA (WRF-ERA5 d02) and PNNL (WRF), with USGS daily discharge [m^3/s] on a
secondary y-axis. One shared legend is placed below the grid.

Same data processing as create_ar_events_plots.py:
basin mean = mean over the model grid cells whose centres fall inside the
Skagit basin boundary (SkagitBoundary.json); UCLA hourly precipitation is the
hourly increment of accumulated RAINC + RAINNC; hourly gaps in CONUS404 and
PNNL are filled with NaN so lines are not drawn across missing hours.

Inputs:
    <BASE_DIR>/data/GIS/SkagitBoundary.json         basin boundary
    <BASE_DIR>/data/usgs_12200500_discharge.rdb     USGS 12200500 daily discharge
    External hourly model data - see "PATHS - external" below.

Output (in this script's folder):
    ar_events_hourly_grid.png

Usage:
    python create_ar_events_grid_plot.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
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

# Other settings
CFS_TO_CMS = 0.0283168   # 1 ft^3/s = 0.0283168 m^3/s
DPI = 300                # resolution of the saved figure

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
print("CREATING 3x2 GRID FOR AR EVENTS")
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

# ============================================================================
# 3x2 GRID OF HOURLY PLOTS
# ============================================================================

# Grid font sizes (figure is 16 in wide; ~0.45x reduction to journal page width)
GRID_FONT_TITLE = 23
GRID_FONT_LABEL = 23
GRID_FONT_TICK = 20
GRID_FONT_LEGEND = 21
DISCHARGE_COLOR = '#777777'

fig, axes = plt.subplots(3, 2, figsize=(16, 16))
panel_letters = 'abcdef'
legend_handles, legend_labels = None, None

for idx, (ax, (date_start_ucla, date_start, date_end, event_date, ar_category)) in enumerate(zip(axes.flat, date_ranges)):
    print(f"\nProcessing event {idx + 1}/{len(date_ranges)}: {event_date} (AR={ar_category})")

    # Load data for this period
    spatial_mean_conus, times_conus, spatial_mean_ucla, times_ucla, spatial_mean_pnnl, times_pnnl = \
        load_data_for_period(date_start, date_end, date_start_ucla)

    # Fill missing times with NaN to prevent line connections across gaps
    # Note: Skip UCLA because it's already processed through diff() which modifies the time structure
    if spatial_mean_conus is not None:
        times_conus, spatial_mean_conus = fill_missing_times(times_conus, spatial_mean_conus)
    if spatial_mean_pnnl is not None:
        times_pnnl, spatial_mean_pnnl = fill_missing_times(times_pnnl, spatial_mean_pnnl)

    # Plot precipitation (filter out NaN values)
    if spatial_mean_conus is not None:
        mask_conus = ~np.isnan(spatial_mean_conus)
        ax.plot(times_conus[mask_conus], spatial_mean_conus[mask_conus], marker='o', linewidth=2, markersize=3,
                color=COLOR_CONUS, label='CONUS404', alpha=0.8)

    if spatial_mean_ucla is not None:
        mask_ucla = ~np.isnan(spatial_mean_ucla)
        ax.plot(times_ucla[mask_ucla], spatial_mean_ucla[mask_ucla], marker='s', linewidth=2, markersize=3,
                color=COLOR_UCLA, label='UCLA', alpha=0.8)

    if spatial_mean_pnnl is not None:
        mask_pnnl = ~np.isnan(spatial_mean_pnnl)
        ax.plot(times_pnnl[mask_pnnl], spatial_mean_pnnl[mask_pnnl], marker='^', linewidth=2, markersize=3,
                color=COLOR_PNNL, label='PNNL', alpha=0.8)

    # Secondary y-axis for discharge
    ax2 = ax.twinx()
    q_x, q_y = discharge_steps(df_discharge_all, date_start, date_end + timedelta(hours=1))
    if q_x is not None:
        ax2.step(q_x, q_y, where='post',
                 linewidth=2.5, color=DISCHARGE_COLOR, label='USGS Discharge', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=DISCHARGE_COLOR, labelsize=GRID_FONT_TICK)

    # Panel title, e.g. "(a) 24 Nov 1990, AR5"
    ev = pd.Timestamp(event_date)
    ax.set_title(f"({panel_letters[idx]}) {ev.strftime('%-d %b %Y')}, AR{int(float(ar_category))}",
                 fontsize=GRID_FONT_TITLE, fontweight='bold', loc='left')

    ax.tick_params(axis='both', labelsize=GRID_FONT_TICK)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Date ticks: one label every 2 days (3 for long events), rotated
    n_days = (date_end - date_start).days + 1
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2 if n_days <= 10 else 3))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

    # Axis labels only on the outer edges of the grid
    if idx % 2 == 0:
        ax.set_ylabel('Precipitation (mm h$^{-1}$)', fontsize=GRID_FONT_LABEL, fontweight='bold')
    else:
        ax2.set_ylabel('Discharge (m$^{3}$ s$^{-1}$)', fontsize=GRID_FONT_LABEL,
                       fontweight='bold', color=DISCHARGE_COLOR)
    if idx >= 4:
        ax.set_xlabel('Date', fontsize=GRID_FONT_LABEL, fontweight='bold')

    if legend_handles is None:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        legend_handles, legend_labels = lines1 + lines2, labels1 + labels2

# Single shared legend below the grid
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.5, w_pad=1.5)
leg = fig.legend(legend_handles, legend_labels, loc='lower center', ncol=4, fontsize=GRID_FONT_LEGEND,
                 frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.5)
# Thicker lines / bigger markers in the legend only, for quick recognition
for line in leg.get_lines():
    line.set_linewidth(5)
    line.set_markersize(9)
    line.set_alpha(1.0)

output_path = OUTPUT_DIR / "ar_events_hourly_grid.png"
fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
print(f"\nGrid plot saved to: {output_path}")
plt.close(fig)

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
