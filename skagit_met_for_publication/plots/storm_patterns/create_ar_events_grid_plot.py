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

from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# ============================================================================
# PATHS - project files (relative to this script)
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent              # plots/storm_patterns/
BASE_DIR = SCRIPT_DIR.parents[1]                          # skagit_met_for_publication/
OUTPUT_DIR = SCRIPT_DIR
BOUNDARY_PATH = BASE_DIR / "data/GIS/SkagitBoundary.json"
DISCHARGE_PATH = BASE_DIR / "data/usgs_12200500_discharge.rdb"

# ============================================================================
# PATHS - external model data (edit to your local copies)
# ============================================================================
VAULT_DIR = Path("/data0/skagit_met/data_transfer/data")
conus_data_path = VAULT_DIR / "CONUS_hourly/hourly_ar_non_ar_events"
ucla_data_path = VAULT_DIR / "ucla_era5_d02_hourly/hourly_ar_non_ar_events"
ucla_coords_file = Path("/data0/hernanqd/instance_2021_data/preparing_datasets/UCLA/wrfinput_d02_coord.nc")
PNNL_DIR = VAULT_DIR / "PNNL/historical"                  # <year>/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.<year>.nc
pnnl_grid_file = PNNL_DIR / "SERDP6km.geo_em.d01.nc"

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

def get_pnnl_data_path(year):
    """Get PNNL data path for a given year."""
    pnnl_path = PNNL_DIR / str(year) / f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc"
    if pnnl_path.exists():
        return pnnl_path
    else:
        return None

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
print("=" * 70)
print("LOADING BOUNDARY AND MASKS")
print("=" * 70)

boundary_gdf = gpd.read_file(BOUNDARY_PATH)
if hasattr(boundary_gdf.to_crs("EPSG:4326"), 'union_all'):
    poly = boundary_gdf.to_crs("EPSG:4326").union_all()
else:
    poly = boundary_gdf.to_crs("EPSG:4326").unary_union
print(f"Boundary loaded: {BOUNDARY_PATH}")

# Load UCLA coordinates
ucla_coords = xr.open_dataset(ucla_coords_file)
lat2d = ucla_coords['lat2d'].values
lon2d = ucla_coords['lon2d'].values
print(f"UCLA coordinates loaded: lat2d shape {lat2d.shape}, lon2d shape {lon2d.shape}")

# Create UCLA mask
df_pts = pd.DataFrame({'lon': lon2d.flatten(), 'lat': lat2d.flatten()})
gdf_pts = gpd.GeoDataFrame(
    df_pts,
    geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
    crs="EPSG:4326"
)
inside = gdf_pts.intersects(poly).values
mask_ucla = inside.reshape(lon2d.shape)
print(f"UCLA mask: {inside.sum()} cells inside watershed")

# Load PNNL coordinates
ds_pnnl_static = xr.open_dataset(pnnl_grid_file)
pnnl_lon_full = ds_pnnl_static.XLONG_M.values[0]
pnnl_lat_full = ds_pnnl_static.XLAT_M.values[0]
ds_pnnl_static.close()
print(f"PNNL coordinates loaded: shape {pnnl_lon_full.shape}")

# Create PNNL mask
df_pts = pd.DataFrame({'lon': pnnl_lon_full.flatten(), 'lat': pnnl_lat_full.flatten()})
gdf_pts = gpd.GeoDataFrame(
    df_pts,
    geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
    crs="EPSG:4326"
)
inside = gdf_pts.intersects(poly).values
mask_pnnl = inside.reshape(pnnl_lon_full.shape)
print(f"PNNL mask: {inside.sum()} cells inside watershed")

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
    """Load CONUS, UCLA and PNNL basin-mean hourly precipitation for a given period."""
    if date_start_ucla is None:
        date_start_ucla = date_start - timedelta(hours=1)

    # CONUS
    conus_files_list = sorted(conus_data_path.glob('*.PREC_ACC_NC.wrf2d_d01_*.nc'))
    conus_files_in_range = []
    for f in conus_files_list:
        try:
            ds_temp = xr.open_dataset(f)
            time = ds_temp.Time.values[0]
            if pd.Timestamp(date_start) <= pd.Timestamp(time) <= pd.Timestamp(date_end):
                conus_files_in_range.append(f)
            ds_temp.close()
        except:
            pass

    spatial_mean_conus = None
    times_conus = None
    if conus_files_in_range:
        ds_conus_list = [xr.open_dataset(f) for f in conus_files_in_range]
        ds_conus = xr.concat(ds_conus_list, dim='Time')

        # Create CONUS mask for this period
        df_pts = pd.DataFrame({'lon': ds_conus.XLONG.values.flatten(),
                               'lat': ds_conus.XLAT.values.flatten()})
        gdf_pts = gpd.GeoDataFrame(
            df_pts,
            geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
            crs="EPSG:4326"
        )
        inside = gdf_pts.intersects(poly).values
        mask_conus_local = inside.reshape(ds_conus.XLONG.values.shape)

        mask_da_conus = xr.DataArray(mask_conus_local, dims=['south_north', 'west_east'])
        data_masked_conus = ds_conus['PREC_ACC_NC'].where(mask_da_conus)
        spatial_mean_conus = data_masked_conus.mean(dim=['south_north', 'west_east'], skipna=True).values
        times_conus = pd.to_datetime(ds_conus.Time.values)

    # UCLA
    current = date_start_ucla
    ucla_filenames = []
    while current <= date_end:
        fn = f"auxhist_d01_{current.strftime('%Y-%m-%d_%H:%M:%S')}.nc"
        ucla_filenames.append(ucla_data_path / fn)
        current += timedelta(hours=1)

    spatial_mean_ucla = None
    times_ucla = None
    ucla_files_exist = [f for f in ucla_filenames if f.exists()]
    if ucla_files_exist:
        ds_ucla = xr.open_mfdataset(ucla_files_exist, concat_dim='Time', combine='nested')

        # Create UCLA mask for this period (using the actual data coordinates)
        df_pts = pd.DataFrame({'lon': lon2d.flatten(), 'lat': lat2d.flatten()})
        gdf_pts = gpd.GeoDataFrame(
            df_pts,
            geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
            crs="EPSG:4326"
        )
        inside = gdf_pts.intersects(poly).values
        mask_ucla_local = inside.reshape(lon2d.shape)

        mask_da_ucla = xr.DataArray(mask_ucla_local, dims=['south_north', 'west_east'])
        ds_ucla_rain = ds_ucla[['RAINC', 'RAINNC']]
        ds_masked_ucla = ds_ucla_rain.where(mask_da_ucla)
        total_rain_ucla = ds_masked_ucla['RAINC'] + ds_masked_ucla['RAINNC']
        total_increments_ucla = total_rain_ucla.diff(dim='Time')
        spatial_mean_ucla = total_increments_ucla.mean(dim=['south_north', 'west_east'], skipna=True).values
        times_raw_ucla = ds_ucla['Times'].values
        times_ucla = []
        for t in times_raw_ucla[1:]:
            if isinstance(t, bytes):
                t_str = t.decode().replace('_', ' ')
            else:
                t_str = str(t).replace('_', ' ')
            times_ucla.append(pd.Timestamp(t_str))
        times_ucla = np.array(times_ucla)

    # PNNL - Load dynamically based on year
    spatial_mean_pnnl = None
    times_pnnl = None
    year = date_start.year
    pnnl_data_path = get_pnnl_data_path(year)

    if pnnl_data_path:
        ds_pnnl_full_period = xr.open_dataset(pnnl_data_path)
        date_start_pd = pd.Timestamp(date_start)
        date_end_pd = pd.Timestamp(date_end)
        ds_pnnl_subset = ds_pnnl_full_period.sel(time=slice(date_start_pd, date_end_pd))

        if len(ds_pnnl_subset.time) > 0:
            # Create PNNL mask (using pre-computed coordinates)
            df_pts = pd.DataFrame({'lon': pnnl_lon_full.flatten(), 'lat': pnnl_lat_full.flatten()})
            gdf_pts = gpd.GeoDataFrame(
                df_pts,
                geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
                crs="EPSG:4326"
            )
            inside = gdf_pts.intersects(poly).values
            mask_pnnl_local = inside.reshape(pnnl_lon_full.shape)

            data_masked_pnnl = ds_pnnl_subset['PREC_ACC_NC'].where(mask_pnnl_local)
            spatial_mean_pnnl = data_masked_pnnl.mean(dim=['x', 'y'], skipna=True).values
            times_pnnl = pd.to_datetime(ds_pnnl_subset.time.values)

    return spatial_mean_conus, times_conus, spatial_mean_ucla, times_ucla, spatial_mean_pnnl, times_pnnl

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
    date_end_plus_1h = date_end + timedelta(hours=1)
    df_discharge_period = df_discharge_all[(df_discharge_all['date'] >= date_start) &
                                            (df_discharge_all['date'] <= date_end_plus_1h)]
    if not df_discharge_period.empty:
        ax2.plot(df_discharge_period['date'], df_discharge_period['discharge_cms'],
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
