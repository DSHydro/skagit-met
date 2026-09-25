"""
Extract and visualize cumulative precipitation from hourly datasets for specific AR events.

This module uses hourly weather products (CONUS404, UCLA, PNNL) to create cumulative
precipitation comparison plots for AR events. Each plot includes peak
discharge measurements from USGS gauges and AR scale classifications.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import geopandas as gpd
import rioxarray
import regionmask
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
# --- Boundary path (commented out - using HUC8 masking instead) ---
# BOUNDARY_PATH = Path("/data0/nksp2/skagit/skagit_2/skagit-met/data/GIS/SkagitBoundary.json")
EVENTS_CSV = os.path.join(BASE_DIR, "multi_product_bulk_bias/outputs/4_clean_bias_table.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "cumulative_precipitation_plot")

# Hydrology data paths
HYDRO_BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
HYDRO_EXP_DATA_DIR = os.path.join(HYDRO_BASE_DIR, "experiments_2/data")
HYDRO_Q_PATH = os.path.join(HYDRO_EXP_DATA_DIR, "usgs_12200500_discharge.rdb")

# Hourly data paths
conus_data_path = Path("/data0/hernanqd/instance_2021_data/preparing_datasets/CONUS404/hourly_ar_non_ar_events")
ucla_data_path = Path("/data0/hernanqd/instance_2021_data/hourly_ar_non_ar_events")
ucla_coords_file = Path("/data0/hernanqd/instance_2021_data/preparing_datasets/UCLA/wrfinput_d02_coord.nc")
pnnl_grid_file = Path("/data0/skagit_met/data_transfer/data/PNNL/historical/SERDP6km.geo_em.d01.nc")

# Specific event dates to plot
SPECIFIC_DATES = [
    {'event_date': '1990-11-24', 'start_date': datetime(1990, 11, 21, 0, 0, 0), 'end_date': datetime(1990, 11, 29, 23, 0, 0)},
    {'event_date': '1995-11-29', 'start_date': datetime(1995, 11, 26, 0, 0, 0), 'end_date': datetime(1995, 12, 4, 23, 0, 0)},
    {'event_date': '1990-11-10', 'start_date': datetime(1990, 11, 7, 0, 0, 0), 'end_date': datetime(1990, 11, 15, 23, 0, 0)},
    {'event_date': '2006-11-07', 'start_date': datetime(2006, 11, 4, 0, 0, 0), 'end_date': datetime(2006, 11, 12, 23, 0, 0)},
    {'event_date': '2003-10-21', 'start_date': datetime(2003, 10, 14, 0, 0, 0), 'end_date': datetime(2003, 10, 28, 23, 0, 0)},
    {'event_date': '2021-11-15', 'start_date': datetime(2021, 11, 12, 0, 0, 0), 'end_date': datetime(2021, 11, 20, 23, 0, 0)},
]


def load_usgs_rdb(filepath, param_name):
    """Load USGS RDB file"""
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


def load_regions():
    """Load HUC8 regions"""
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    return gdf


def get_mask(gdf, lon, lat):
    """Create mask using regionmask with HUC8 sub-basins"""
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    for r in ["Upper Skagit", "Sauk", "Lower Skagit"]:
        idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
        region_indices.append(idx)
    return mask.sel(region=region_indices)

# --- Boundary-based masking (commented out - using HUC8 masking instead) ---
# def get_boundary_mask(lon2d, lat2d, boundary_poly):
#     """Create 2D mask for watershed boundary (same approach as spatial_plots and create_ar_events_plots)"""
#     df_pts = pd.DataFrame({'lon': lon2d.flatten(), 'lat': lat2d.flatten()})
#     gdf_pts = gpd.GeoDataFrame(
#         df_pts,
#         geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
#         crs="EPSG:4326"
#     )
#     inside = gdf_pts.intersects(boundary_poly).values
#     return inside.reshape(lon2d.shape)


def calculate_basin_mean(da, mask_2d):
    """Calculate basin-mean precipitation with proper masking"""
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

        # Get time coordinate - try multiple approaches
        time_coord = None

        # Try to get from the first dimension (usually time)
        try:
            first_dim = da.dims[0]
            if first_dim in da.coords:
                time_coord_candidate = da.coords[first_dim].values
                if len(time_coord_candidate) > 0:
                    time_coord = time_coord_candidate
        except Exception as e:
            pass

        # If that didn't work, try common names
        if time_coord is None:
            for name in ['time', 'Time', 'TIME']:
                if name in da.coords:
                    time_coord = da.coords[name].values
                    break

        # Last resort: use integers
        if time_coord is None:
            time_coord = range(time_steps)

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


def load_masks_huc8():
    """Load HUC8 regions and create masks for UCLA and PNNL products"""
    gdf = load_regions()

    # Load UCLA coordinates and mask using HUC8
    ucla_coords = xr.open_dataset(ucla_coords_file)
    lat2d_ucla = ucla_coords['lat2d'].values
    lon2d_ucla = ucla_coords['lon2d'].values
    mask_ucla_3d = get_mask(gdf, lon2d_ucla, lat2d_ucla)
    mask_ucla = mask_ucla_3d.any(dim='region').values

    # Load PNNL coordinates and mask using HUC8
    ds_pnnl_static = xr.open_dataset(pnnl_grid_file)
    pnnl_lon_full = ds_pnnl_static.XLONG_M.values[0]
    pnnl_lat_full = ds_pnnl_static.XLAT_M.values[0]
    ds_pnnl_static.close()

    mask_pnnl_3d = get_mask(gdf, pnnl_lon_full, pnnl_lat_full)
    mask_pnnl = mask_pnnl_3d.any(dim='region').values

    return gdf, mask_ucla, lon2d_ucla, pnnl_lon_full, pnnl_lat_full, mask_pnnl

# --- Boundary-based mask loading (commented out - using HUC8 masking instead) ---
# def load_masks_boundary():
#     """Load boundary-based masks for UCLA and PNNL products (same approach as spatial_plots and create_ar_events_plots)"""
#     # Load boundary polygon
#     boundary_gdf = gpd.read_file(BOUNDARY_PATH)
#     if hasattr(boundary_gdf.to_crs("EPSG:4326"), 'union_all'):
#         boundary_poly = boundary_gdf.to_crs("EPSG:4326").union_all()
#     else:
#         boundary_poly = boundary_gdf.to_crs("EPSG:4326").unary_union
#
#     # Load UCLA coordinates and create boundary mask
#     ucla_coords = xr.open_dataset(ucla_coords_file)
#     lat2d_ucla = ucla_coords['lat2d'].values
#     lon2d_ucla = ucla_coords['lon2d'].values
#     mask_ucla = get_boundary_mask(lon2d_ucla, lat2d_ucla, boundary_poly)
#
#     # Load PNNL coordinates and create boundary mask
#     ds_pnnl_static = xr.open_dataset(pnnl_grid_file)
#     pnnl_lon_full = ds_pnnl_static.XLONG_M.values[0]
#     pnnl_lat_full = ds_pnnl_static.XLAT_M.values[0]
#     ds_pnnl_static.close()
#
#     mask_pnnl = get_boundary_mask(pnnl_lon_full, pnnl_lat_full, boundary_poly)
#
#     return mask_ucla, lon2d_ucla, pnnl_lon_full, pnnl_lat_full, mask_pnnl


def get_pnnl_data_path(year):
    """Get PNNL data path for a given year."""
    pnnl_path = Path(f"/data0/skagit_met/data_transfer/data/PNNL/historical/{year}/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
    if pnnl_path.exists():
        return pnnl_path
    return None


def load_hourly_data_for_period(date_start, date_end, gdf, mask_ucla, lon2d_ucla, pnnl_lon_full, pnnl_lat_full, mask_pnnl):
    """Load CONUS404, UCLA, and PNNL hourly data for a given period using consistent HUC8 masking."""

    # CONUS404
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

    series_conus = None
    times_conus = None
    if conus_files_in_range:
        ds_conus_list = [xr.open_dataset(f) for f in conus_files_in_range]
        ds_conus = xr.concat(ds_conus_list, dim='Time')

        # Create CONUS mask using HUC8 regions
        mask_conus_3d = get_mask(gdf, ds_conus.XLONG.values, ds_conus.XLAT.values)
        mask_conus = mask_conus_3d.any(dim='region').values

        # Use consistent basin mean calculation
        series_conus = calculate_basin_mean(ds_conus['PREC_ACC_NC'], mask_conus)
        times_conus = pd.to_datetime(ds_conus.Time.values)

    # UCLA
    date_start_ucla = date_start - timedelta(hours=1)
    current = date_start_ucla
    ucla_filenames = []
    while current <= date_end:
        fn = f"auxhist_d01_{current.strftime('%Y-%m-%d_%H:%M:%S')}.nc"
        ucla_filenames.append(ucla_data_path / fn)
        current += timedelta(hours=1)

    series_ucla_increments = None
    times_ucla = None
    ucla_files_exist = [f for f in ucla_filenames if f.exists()]
    if ucla_files_exist:
        ds_ucla = xr.open_mfdataset(ucla_files_exist, concat_dim='Time', combine='nested')

        # Calculate total rain and increments
        total_rain_ucla = ds_ucla['RAINC'] + ds_ucla['RAINNC']
        total_increments_ucla = total_rain_ucla.diff(dim='Time')

        # Convert Times strings to datetime64 and round to nearest hour for alignment
        times_raw_ucla = ds_ucla['Times'].values
        times_ucla_dt = []
        for t in times_raw_ucla[1:]:  # Skip first (before diff)
            if isinstance(t, bytes):
                t_str = t.decode().replace('_', ' ')
            else:
                t_str = str(t).replace('_', ' ')
            ts = pd.Timestamp(t_str)
            ts_rounded = ts.round('1H')  # Round to nearest hour for alignment with other products
            times_ucla_dt.append(ts_rounded)
        times_ucla_dt = np.array(times_ucla_dt, dtype='datetime64[ns]')

        # Restore time coordinate with proper datetime values
        total_increments_ucla = total_increments_ucla.assign_coords(
            Time=('Time', times_ucla_dt)
        )

        # Use consistent basin mean calculation for increments
        series_ucla_increments = calculate_basin_mean(total_increments_ucla, mask_ucla)

        times_ucla = times_ucla_dt

    # PNNL
    series_pnnl = None
    times_pnnl = None
    year = date_start.year
    pnnl_data_path = get_pnnl_data_path(year)

    if pnnl_data_path:
        ds_pnnl_full_period = xr.open_dataset(pnnl_data_path)
        date_start_pd = pd.Timestamp(date_start)
        date_end_pd = pd.Timestamp(date_end)
        ds_pnnl_subset = ds_pnnl_full_period.sel(time=slice(date_start_pd, date_end_pd))

        if len(ds_pnnl_subset.time) > 0:
            # Use consistent basin mean calculation
            series_pnnl = calculate_basin_mean(ds_pnnl_subset['PREC_ACC_NC'], mask_pnnl)
            times_pnnl = pd.to_datetime(ds_pnnl_subset.time.values)

    return series_conus, times_conus, series_ucla_increments, times_ucla, series_pnnl, times_pnnl


def compute_cumulative_from_hourly(spatial_mean, times, resampling_freq='1D'):
    """Compute cumulative precipitation from hourly values."""
    if spatial_mean is None or times is None:
        return None

    df = pd.DataFrame({'precip': spatial_mean}, index=pd.to_datetime(times))
    cumul = df['precip'].resample(resampling_freq).sum()
    return cumul


def save_timeseries_to_csv(all_event_data_with_hourly, output_dir):
    """Save all hourly and cumulative timeseries to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for event_date, event_data in all_event_data_with_hourly.items():
        event_str = event_date.strftime('%Y%m%d')

        # Save hourly incremental precipitation
        hourly_series_list = [f'{product}_hourly' for product in ['conus', 'ucla', 'pnnl']
                              if f'{product}_hourly' in event_data and event_data[f'{product}_hourly'] is not None]

        if hourly_series_list:
            hourly_data_list = []
            for key in hourly_series_list:
                product = key.replace('_hourly', '')
                series = event_data[key]
                # Ensure index is datetime
                series.index = pd.to_datetime(series.index)
                df_temp = series.to_frame(name=f'{product.upper()}_hourly_precip_mm')
                hourly_data_list.append(df_temp)

            if hourly_data_list:
                # Concatenate by index alignment (join='outer' keeps all timestamps)
                hourly_combined = pd.concat(hourly_data_list, axis=1, join='outer')
                hourly_combined = hourly_combined.reset_index().rename(columns={'index': 'datetime'})
                hourly_csv_path = os.path.join(output_dir, f'hourly_precip_{event_str}.csv')
                hourly_combined.to_csv(hourly_csv_path, index=False)
                print(f"  Saved hourly data: {hourly_csv_path}")

        # Save daily cumulative precipitation
        cumul_combined = {}
        for product in ['conus', 'ucla', 'pnnl']:
            if product in event_data and event_data[product] is not None:
                cumul_combined[f'{product.upper()}_daily_cumsum_mm'] = event_data[product].cumsum()

        if cumul_combined:
            cumul_df = pd.DataFrame(cumul_combined)
            cumul_df['datetime'] = cumul_df.index
            cumul_df = cumul_df[['datetime'] + [col for col in cumul_df.columns if col != 'datetime']]
            cumul_csv_path = os.path.join(output_dir, f'cumulative_from_hourly_precip_{event_str}.csv')
            cumul_df.to_csv(cumul_csv_path, index=False)
            print(f"  Saved cumulative data: {cumul_csv_path}")


def plot_hourly_cumulative_ar_events():
    """Extract and plot specific AR events using hourly data"""
    print("Loading HUC8 regions and creating masks...")
    gdf, mask_ucla, lon2d_ucla, pnnl_lon_full, pnnl_lat_full, mask_pnnl = load_masks_huc8()

    print("Loading events data...")
    events_df = pd.read_csv(EVENTS_CSV)
    events_df['date'] = pd.to_datetime(events_df['date'])

    print("Loading discharge data...")
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')
    q_df['discharge_cms'] = q_df['discharge_cfs'] * 0.0283168

    print(f"Extracting and plotting {len(SPECIFIC_DATES)} specific AR events...")

    # Extract data for all events and calculate max cumulative precipitation
    max_cumsum = 0
    all_event_data = {}
    all_event_data_with_hourly = {}

    for event_info in SPECIFIC_DATES:
        event_date = pd.to_datetime(event_info['event_date'])
        start_date = event_info['start_date']
        end_date = event_info['end_date']

        print(f"\nExtracting hourly data for {event_date.strftime('%Y-%m-%d')}...")
        series_conus, times_conus, series_ucla, times_ucla, series_pnnl, times_pnnl = \
            load_hourly_data_for_period(start_date, end_date, gdf, mask_ucla, lon2d_ucla, pnnl_lon_full, pnnl_lat_full, mask_pnnl)

        # Compute daily cumulative values
        cumul_conus = compute_cumulative_from_hourly(series_conus.values if series_conus is not None else None,
                                                      times_conus, '1D') if series_conus is not None else None
        cumul_ucla = compute_cumulative_from_hourly(series_ucla.values if series_ucla is not None else None,
                                                     times_ucla, '1D') if series_ucla is not None else None
        cumul_pnnl = compute_cumulative_from_hourly(series_pnnl.values if series_pnnl is not None else None,
                                                     times_pnnl, '1D') if series_pnnl is not None else None

        # Debug output for problem dates
        if event_date.strftime('%Y-%m-%d') in ['1995-11-29', '1990-11-10', '2021-11-15']:
            print(f"  [DEBUG-HOURLY] {event_date.strftime('%Y-%m-%d')} CONUS:")
            if cumul_conus is not None:
                print(f"    Daily values: {dict(cumul_conus)}")
                print(f"    Cumulative: {dict(cumul_conus.cumsum())}")
            else:
                print(f"    No CONUS data found")
            if series_conus is not None:
                print(f"    Hourly series shape: {series_conus.shape}, first 5 values: {series_conus.values[:5]}")
            else:
                print(f"    No hourly series")

        # Store cumulative and hourly data
        event_cumul_data = {}
        event_hourly_data = {}

        if cumul_conus is not None:
            event_cumul_data['conus'] = cumul_conus
            max_val = cumul_conus.cumsum().max() if len(cumul_conus) > 0 else 0
            max_cumsum = max(max_cumsum, max_val)
        if series_conus is not None:
            event_hourly_data['conus_hourly'] = series_conus

        if cumul_ucla is not None:
            event_cumul_data['ucla'] = cumul_ucla
            max_val = cumul_ucla.cumsum().max() if len(cumul_ucla) > 0 else 0
            max_cumsum = max(max_cumsum, max_val)
        if series_ucla is not None:
            event_hourly_data['ucla_hourly'] = series_ucla

        if cumul_pnnl is not None:
            event_cumul_data['pnnl'] = cumul_pnnl
            max_val = cumul_pnnl.cumsum().max() if len(cumul_pnnl) > 0 else 0
            max_cumsum = max(max_cumsum, max_val)
        if series_pnnl is not None:
            event_hourly_data['pnnl_hourly'] = series_pnnl

        all_event_data[event_date] = event_cumul_data
        all_event_data_with_hourly[event_date] = {**event_cumul_data, **event_hourly_data}

    ylim_max = max_cumsum * 1.05

    # Save timeseries data to CSV files
    print("\nSaving timeseries data to CSV files...")
    csv_output_dir = os.path.join(OUTPUT_DIR, 'timeseries_data')
    save_timeseries_to_csv(all_event_data_with_hourly, csv_output_dir)

    # Create plots - 3x2 grid matching plot_specific_ar_events.py layout
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes = axes.flatten()

    colors = {'conus': '#d62728', 'ucla': '#9467bd', 'pnnl': '#ff7f0e'}
    markers = {'conus': 'D', 'ucla': 'v', 'pnnl': 's'}

    for idx, event_info in enumerate(SPECIFIC_DATES):
        ax = axes[idx]
        event_date = pd.to_datetime(event_info['event_date'])
        start_date = event_info['start_date']
        end_date = event_info['end_date']

        # Get AR scale
        event_row = events_df[events_df['date'] == event_date]
        ar_scale = event_row['ar_scale'].values[0] if len(event_row) > 0 else None

        # Find maximum discharge in event window
        discharge_window = q_df[(q_df['date'] >= start_date) & (q_df['date'] <= end_date)]
        max_discharge_cfs = discharge_window['discharge_cfs'].max() if len(discharge_window) > 0 else None
        discharge_cms = max_discharge_cfs * 0.0283168 if max_discharge_cfs is not None and max_discharge_cfs > 0 else None

        # Plot each product
        event_cumul_data = all_event_data[event_date]
        for product, cumul_data in event_cumul_data.items():
            if cumul_data is not None and len(cumul_data) > 0:
                cumsum = cumul_data.cumsum()
                ax.plot(cumsum.index, cumsum.values, label=product.upper(),
                       marker=markers[product], linewidth=2, markersize=5,
                       color=colors[product], alpha=0.8)

        ar_label = f'(AR = {ar_scale} | Discharge = {discharge_cms:.0f} cms)' if ar_scale is not None else ''
        ax.set_title(f'Event: {event_date.strftime("%Y-%m-%d")} {ar_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=10)
        ax.set_ylim(0, ylim_max)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Cumulative Precipitation from Hourly Data - Skagit Basin AR Events',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'hourly_cumulative_ar_events.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_hourly_cumulative_ar_events()
