"""
Create individual plots for multiple AR events comparing CONUS404, UCLA, and PNNL precipitation with USGS discharge.
"""

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import zipfile
import rioxarray
import regionmask
import os

# Configuration
BOUNDARY_PATH = Path("/data0/nksp2/skagit/skagit_2/skagit-met/data/GIS/SkagitBoundary.json")
DISCHARGE_PATH = Path("/data0/nksp2/skagit/skagit_2/skagit-met/experiments_2/data/usgs_12200500_discharge.rdb")

# Data paths
conus_data_path = Path("/data0/skagit_met/data_transfer/data/CONUS_hourly/hourly_ar_non_ar_events")
ucla_data_path = Path("/data0/skagit_met/data_transfer/data/ucla_era5_d02_hourly/hourly_ar_non_ar_events")
ucla_coords_file = Path("/data0/hernanqd/instance_2021_data/preparing_datasets/UCLA/wrfinput_d02_coord.nc")
pnnl_grid_file = Path("/data0/skagit_met/data_transfer/data/PNNL/historical/SERDP6km.geo_em.d01.nc")

# PRISM data
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
BASE_DIR = "/data0/hernanqd/plots_code/skagit-met"
HUC8_GEO = Path(BASE_DIR) / "data/GIS/SkagitSubBasin_HUC8.geojson"
PRISM_ROOT = Path(VAULT_DIR) / "prism_new_hq"
SNOTEL_DATA_BASE = Path(BASE_DIR) / "data/hourly_snotel"

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
    pnnl_path = Path(f"/data0/skagit_met/data_transfer/data/PNNL/historical/{year}/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
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

def get_decade_folder(year: int) -> str:
    """Get decade folder name for PRISM data organization."""
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
    """Load PRISM data from new format (TIFF) zip file for a specific date."""
    date_str = date.strftime('%Y%m%d')
    year = date.year
    decade_folder = get_decade_folder(year)
    zip_path = Path(prism_root) / decade_folder / f"prism_ppt_us_25m_{date_str}.zip"

    if not zip_path.exists():
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith('.tif'):
                            filepath = Path(root) / f
                            da = rioxarray.open_rasterio(str(filepath))
                            return da.squeeze().copy(deep=True)
    except Exception as e:
        pass

    return None

def calculate_basin_mean(da, mask_2d):
    """Calculate spatial mean over masked basin region."""
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
        time_coord = da.time.values if 'time' in da.coords else range(time_steps)
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

def get_mask(gdf, lon, lat):
    """Create spatial mask for basin regions."""
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    for r in ["Upper Skagit", "Sauk", "Lower Skagit"]:
        idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
        region_indices.append(idx)
    return mask.sel(region=region_indices)

def load_snotel_hourly_data(start_date, end_date):
    """Load SNOTEL hourly precipitation data for an event window."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Search for SNOTEL file matching the event period
    zarr_files = list(SNOTEL_DATA_BASE.glob('*_SNOTEL_hourly_data.zarr'))
    zarr_path = None

    for f in zarr_files:
        # Extract dates from filename (format: YYYY-MM-DD_YYYY-MM-DD_SNOTEL_hourly_data.zarr)
        fname = f.stem.replace('_SNOTEL_hourly_data', '')
        # Split by underscore to separate the two dates
        date_parts = fname.split('_')
        if len(date_parts) >= 2:
            try:
                file_start = pd.Timestamp(date_parts[0])
                file_end = pd.Timestamp(date_parts[1])
                # Check if file overlaps with event period (file_start <= event_end and event_start <= file_end)
                if file_start <= end_ts and start_ts <= file_end:
                    zarr_path = f
                    break
            except:
                continue

    if zarr_path is None:
        print(f"  [WARN] SNOTEL data not found for period {start_ts.date()} to {end_ts.date()}")
        return None

    try:
        ds_snotel = xr.open_zarr(str(zarr_path))
        times = pd.to_datetime(ds_snotel.time.values)
        # Select data within event period
        mask = (times >= pd.Timestamp(start_date)) & (times <= pd.Timestamp(end_date))
        filtered_times = times[mask]
        # Read accumulated precipitation (time, site) and convert inches to mm
        precip_data = ds_snotel['ACCUMULATED PRECIPITATION'].sel(time=filtered_times).values
        precip_mm = precip_data #* 25.4
        n_sites = precip_data.shape[1] if len(precip_data.shape) > 1 else 1
        ds_snotel.close()

        snotel_data = {
            'times': filtered_times,
            'precip': precip_mm,
            'n_stations': n_sites
        }
        print(f"  SNOTEL loaded from {zarr_path.name}: {snotel_data['n_stations']} stations")
        return snotel_data
    except Exception as e:
        print(f"  [WARN] SNOTEL loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

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
df_discharge_all['discharge_cms'] = df_discharge_all['discharge_cfs'] * 0.0283168
print(f"Loaded {len(df_discharge_all)} discharge records")

# ============================================================================
# CREATE PLOTS FOR ALL AR EVENTS
# ============================================================================
print("\n" + "=" * 70)
print("CREATING PLOTS FOR ALL AR EVENTS")
print("=" * 70)

# # Date ranges for AR events (date_start_ucla, date_start_conus_pnnl, date_end, event_date, ar_category)
date_ranges = [
    (datetime(1990, 11, 20, 23, 0, 0), datetime(1990, 11, 21, 0, 0, 0), datetime(1990, 11, 29, 23, 0, 0), "1990-11-24", "5.0"),
    # (datetime(1995, 11, 25, 23, 0, 0), datetime(1995, 11, 26, 0, 0, 0), datetime(1995, 12, 4, 23, 0, 0), "1995-11-29", "4.0"),
    # (datetime(1990, 11, 6, 23, 0, 0), datetime(1990, 11, 7, 0, 0, 0), datetime(1990, 11, 15, 23, 0, 0), "1990-11-10", "4.0"),
    # (datetime(2006, 11, 3, 23, 0, 0), datetime(2006, 11, 4, 0, 0, 0), datetime(2006, 11, 12, 23, 0, 0), "2006-11-07", "5.0"),
    # (datetime(2003, 10, 13, 23, 0, 0), datetime(2003, 10, 14, 0, 0, 0), datetime(2003, 10, 28, 23, 0, 0), "2003-10-21", "5.0"),
    # (datetime(2021, 11, 12, 23, 0, 0), datetime(2021, 11, 13, 0, 0, 0), datetime(2021, 11, 20, 23, 0, 0), "2021-11-15", "4.0"),
]

# Date ranges for AR0 events (date_start_ucla, date_start_conus_pnnl, date_end, event_date, ar_category)
# date_ranges = [
    # (datetime(1995, 11, 28, 23, 0, 0), datetime(1995, 11, 29, 0, 0, 0), datetime(1995, 12, 7, 23, 0, 0), "1995-12-02", "0.0"),
    # (datetime(2011, 1, 14, 23, 0, 0), datetime(2011, 1, 15, 0, 0, 0), datetime(2011, 1, 23, 23, 0, 0), "2011-01-18", "0.0"),
    # (datetime(2015, 11, 10, 23, 0, 0), datetime(2015, 11, 11, 0, 0, 0), datetime(2015, 11, 19, 23, 0, 0), "2015-11-14", "0.0"),
    # (datetime(2007, 3, 9, 23, 0, 0), datetime(2007, 3, 10, 0, 0, 0), datetime(2007, 3, 19, 22, 0, 0), "2007-03-13", "0.0"),
    # (datetime(2021, 11, 29, 23, 0, 0), datetime(2021, 11, 30, 0, 0, 0), datetime(2021, 12, 7, 23, 0, 0), "2021-12-03", "0.0"),
    # (datetime(2010, 12, 10, 23, 0, 0), datetime(2010, 12, 11, 0, 0, 0), datetime(2010, 12, 19, 23, 0, 0), "2010-12-14", "0.0"),
# ]

def load_data_for_period(date_start, date_end, date_start_ucla=None):
    """Load CONUS, UCLA, PNNL, and PRISM data for a given period."""
    if date_start_ucla is None:
        date_start_ucla = date_start - timedelta(hours=1)

    # Generate date range for PRISM extraction (extend by 1 day)
    date_start_ts = pd.Timestamp(date_start)
    date_end_ts = pd.Timestamp(date_end)
    prism_dates = pd.date_range(date_start_ts.date(), date_end_ts.date() + timedelta(days=1), freq='D').tolist()

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

    # Extract PRISM daily data
    spatial_mean_prism = None
    times_prism = None
    try:
        if HUC8_GEO.exists():
            gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
            # prism_data = {}
            #
            # for date in prism_dates:
            #     try:
            #         da = load_prism_from_new_format(date, PRISM_ROOT)
            #         if da is None:
            #             continue
            #         lon = da.x.values
            #         lat = da.y.values
            #         m_prism = get_mask(gdf, lon, lat).any(dim='region')
            #         mean_val = calculate_basin_mean(da, m_prism)
            #         prism_data[date.normalize() if hasattr(date, 'normalize') else pd.Timestamp(date)] = mean_val
            #     except Exception as date_error:
            #         pass
            #
            # if prism_data:
            #     times_prism = np.array(list(prism_data.keys()))
            #     spatial_mean_prism = np.array([prism_data[t] for t in times_prism])
    except Exception as e:
        pass

    return spatial_mean_conus, times_conus, spatial_mean_ucla, times_ucla, spatial_mean_pnnl, times_pnnl, spatial_mean_prism, times_prism

# Loop through each date range and create plots
for idx, (date_start_ucla, date_start, date_end, event_date, ar_category) in enumerate(date_ranges, 1):
    print(f"\nProcessing event {idx}/{len(date_ranges)}: {event_date} (AR={ar_category})")

    # Load data for this period
    spatial_mean_conus, times_conus, spatial_mean_ucla, times_ucla, spatial_mean_pnnl, times_pnnl, spatial_mean_prism, times_prism = \
        load_data_for_period(date_start, date_end, date_start_ucla)

    # Compute 3-hour cumulatives BEFORE filling missing times (needed for proper resampling)
    cumul_3hr_conus = None
    cumul_3hr_ucla = None
    cumul_3hr_pnnl = None
    cumul_3hr_prism = None

    if spatial_mean_conus is not None:
        df_conus = pd.DataFrame({'precip': spatial_mean_conus}, index=pd.to_datetime(times_conus))
        cumul_3hr_conus = df_conus['precip'].resample('3h').sum()

    if spatial_mean_ucla is not None:
        df_ucla = pd.DataFrame({'precip': spatial_mean_ucla}, index=pd.to_datetime(times_ucla))
        cumul_3hr_ucla = df_ucla['precip'].resample('3h').sum()

    if spatial_mean_pnnl is not None:
        df_pnnl = pd.DataFrame({'precip': spatial_mean_pnnl}, index=pd.to_datetime(times_pnnl))
        cumul_3hr_pnnl = df_pnnl['precip'].resample('3h').sum()

    if spatial_mean_prism is not None:
        df_prism = pd.DataFrame({'precip': spatial_mean_prism}, index=pd.to_datetime(times_prism))
        cumul_3hr_prism = df_prism['precip']

    # Load SNOTEL data
    print(f"  Loading SNOTEL data for {event_date}...")
    snotel_data = load_snotel_hourly_data(date_start, date_end)

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
                color='#1f77b4', label='CONUS404', alpha=0.8)

    if spatial_mean_ucla is not None:
        mask_ucla = ~np.isnan(spatial_mean_ucla)
        ax.plot(times_ucla[mask_ucla], spatial_mean_ucla[mask_ucla], marker='s', linewidth=2.5, markersize=4,
                color='#ff7f0e', label='UCLA', alpha=0.8)

    if spatial_mean_pnnl is not None:
        mask_pnnl = ~np.isnan(spatial_mean_pnnl)
        ax.plot(times_pnnl[mask_pnnl], spatial_mean_pnnl[mask_pnnl], marker='^', linewidth=2.5, markersize=4,
                color='#2ca02c', label='PNNL', alpha=0.8)

    if spatial_mean_prism is not None:
        mask_prism = ~np.isnan(spatial_mean_prism)
        ax.plot(times_prism[mask_prism], spatial_mean_prism[mask_prism], marker='o', linewidth=1.5, markersize=8,
                color='#d62728', label='PRISM (daily)', alpha=0.8, linestyle=':')

    # Overlay SNOTEL hourly precipitation
    if snotel_data is not None:
        snotel_precip = snotel_data['precip']
        snotel_times = snotel_data['times']
        # Calculate spatial mean across SNOTEL stations (handle irregular sampling)
        if snotel_precip.ndim > 1:
            snotel_mean = np.nanmean(snotel_precip, axis=1)
        else:
            snotel_mean = snotel_precip
        mask_snotel = ~np.isnan(snotel_mean)
        if mask_snotel.any():
            ax.scatter(snotel_times[mask_snotel], snotel_mean[mask_snotel], s=80, marker='*',
                      color='#2ca02c', edgecolors='darkgreen', linewidths=1.5, alpha=0.8,
                      label=f'SNOTEL (mean of {snotel_data["n_stations"]} stations)', zorder=5)

    # Create secondary y-axis for discharge
    ax2 = ax.twinx()
    date_end_plus_1h = date_end + timedelta(hours=1)
    df_discharge_period = df_discharge_all[(df_discharge_all['date'] >= date_start) &
                                            (df_discharge_all['date'] <= date_end_plus_1h)]

    if not df_discharge_period.empty:
        ax2.plot(df_discharge_period['date'], df_discharge_period['discharge_cms'],
                linewidth=2.5, color='#777777', label='USGS Discharge (cms)', alpha=0.7)
        ax2.set_ylabel('Discharge (cms)', fontsize=13, fontweight='bold', color='#777777')
        ax2.tick_params(axis='y', labelcolor='#777777')

    # Formatting
    ax.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hourly Precipitation (mm/hour)', fontsize=13, fontweight='bold')
    ax.set_title(f'Spatial Mean Hourly Precipitation - Skagit Basin ({event_date}, AR={ar_category})\n{date_start.strftime("%Y-%m-%d")} to {date_end.strftime("%Y-%m-%d")}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper right', framealpha=0.95)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save hourly plot
    output_path = Path(f"/data0/hernanqd/plots_code/skagit_basin_de/houly_timeseries_plots/ar_event_{idx:02d}_hourly.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Hourly plot saved to: {output_path}")
    plt.close()

print("\n" + "=" * 70)
print("COMPLETE!")
print("=" * 70)
