"""Cumulative precipitation of six products during selected atmospheric river events.

For each event in ``SPECIFIC_DATES``, daily basin-mean precipitation over the
Skagit Basin (Upper Skagit, Sauk and Lower Skagit HUC8 sub-basins) is
extracted from six gridded products (PRISM, PNNL, Daymet, CONUS404, UCLA,
GridMET) for a 9-day window (T-3 to T+5) or a custom date range, and the
cumulative totals are plotted in a 3x2 panel figure. Each panel title gives
the event's AR scale and the peak USGS 12200500 discharge in the window. All
panels share the same y-axis range.

Main functions:
  extract_event_window: Extract precipitation window for a specific event across all products
  plot_specific_ar_events: Generate cumulative precipitation comparison plots for selected events

Input
    /data0/skagit_met/data_transfer/data/...      (gridded precipitation products)
    data/GIS/SkagitSubBasin_HUC8.geojson
    data/weather_data/conus404_skagit_precip_daily_full.zarr
    data/usgs_12200500_discharge.rdb
    analysis/outputs/5day_taking_min_and_filling/4_clean_bias_table.csv
Output
    plots/plot_specific_ar_and_non_ar_events/specific_ar_events_cumulative_precipitation.png
    plots/plot_specific_ar_and_non_ar_events/timeseries_data/cumulative_precip_<YYYYMMDD>.csv

Usage
    python plot_specific_ar_events.py
"""

import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import rioxarray
import xarray as xr

# Shared raw-data archive (outside this repository)
VAULT_DIR = "/data0/skagit_met/data_transfer/data"

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
HUC8_GEO = BASE_DIR / "data" / "GIS" / "SkagitSubBasin_HUC8.geojson"
EVENTS_CSV = BASE_DIR / "analysis" / "outputs" / "5day_taking_min_and_filling" / "4_clean_bias_table.csv"
HYDRO_Q_PATH = BASE_DIR / "data" / "usgs_12200500_discharge.rdb"
OUTPUT_DIR = Path(__file__).resolve().parent

# Precipitation product locations
PRISM_DIR = os.path.join(VAULT_DIR, "prism_new_hq")  # daily TIFF zips, by decade folder
DAYMET_DIR = os.path.join(VAULT_DIR, "daymet_new_hq")  # prcp_<year>_subset.nc
PNNL_DIR = os.path.join(VAULT_DIR, "PNNL/historical")  # <year>/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.<year>.nc
PNNL_GEO = os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc")  # PNNL grid coordinates
CONUS_ZARR = BASE_DIR / "data" / "weather_data" / "conus404_skagit_precip_daily_full.zarr"
UCLA_PREC_DIR = os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec")  # prec.daily.era5.d02.<year>.nc
UCLA_STATIC = os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc")  # UCLA grid coordinates
GRIDMET_DIR = os.path.join(VAULT_DIR, "gridmet")  # <year>_daily_4km_gridMET_data.zarr
GRIDMET_SAMPLE = os.path.join(GRIDMET_DIR, "2021_daily_4km_gridMET_data.zarr")  # used only for its grid

# Products, in plotting order, with their line colors and markers
PRODUCTS = ['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']
PRODUCT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
PRODUCT_MARKERS = ['o', 's', '^', 'D', 'v', 'p']
PNNL_LAST_YEAR = 2020  # PNNL historical run ends in 2020

# Default event window: T-3 to T+5 days
WINDOW_DAYS_BEFORE = 3
WINDOW_DAYS_AFTER = 5

CFS_TO_CMS = 0.0283168

# Figure settings
FIGSIZE = (14, 12)
DPI = 300

# Publication font sizes (figure is 14 in wide; expect ~0.5x reduction to journal page width)
FONT_SUPTITLE = 22
FONT_TITLE = 16
FONT_LABEL = 16
FONT_TICK = 14
FONT_LEGEND = 13

# Specific event dates to plot
# Can be a string date or a dict with 'event_date', 'start_date', 'end_date' for custom ranges
SPECIFIC_DATES = [
    '1990-11-24',
    "1995-11-30", #"1995-11-29",
    '1990-11-13', #'1990-11-10',
    '2006-11-07',
    {'event_date': '2003-10-21', 'start_date': '2003-10-14', 'end_date': '2003-10-28'},  # custom range '2003-10-21'
    '2021-11-15',
]


def load_prism_from_new_format(date, prism_root):
    """Load PRISM data from new format (TIFF) zip file for a specific date"""
    date_str = date.strftime('%Y%m%d')
    year = date.year
    decade_folder = get_decade_folder(year)
    zip_path = os.path.join(prism_root, decade_folder, f"prism_ppt_us_25m_{date_str}.zip")

    if not os.path.exists(zip_path):
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith('.tif'):
                            filepath = os.path.join(root, f)
                            da = rioxarray.open_rasterio(filepath)
                            return da.squeeze().copy(deep=True)
    except Exception:
        pass

    return None


def load_daymet_from_netcdf(year):
    """Load Daymet data from netCDF file for a specific year"""
    daymet_nc_path = os.path.join(DAYMET_DIR, f"prcp_{year}_subset.nc")

    if not os.path.exists(daymet_nc_path):
        return None

    try:
        ds = xr.open_dataset(daymet_nc_path)
        return ds
    except Exception as e:
        print(f"  Error loading Daymet netCDF for {year}: {e}")
        return None


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
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    return gdf


def get_mask(gdf, lon, lat):
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    for r in ["Upper Skagit", "Sauk", "Lower Skagit"]:
        idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
        region_indices.append(idx)
    return mask.sel(region=region_indices)


def calculate_basin_mean(da, mask_2d):
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
        # Mask out nodata values (typically -9999 or negative values for TIFF)
        data = data.astype(float)
        data[data < 0] = np.nan  # Treat negative values as nodata

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


def get_decade_folder(year: int) -> str:
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


def extract_event_window(event_date, products_to_extract=PRODUCTS, start_date=None, end_date=None):
    """Extract daily basin-mean precipitation around an event for each product.

    The default window is 9 days (T-3 to T+5); ``start_date``/``end_date``
    give a custom range instead. Returns a DataFrame indexed by date string
    with one column per product.
    """

    if start_date and end_date:
        # Custom date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        window_dates = pd.date_range(start_dt, end_dt, freq='D').tolist()
    else:
        # Default window (T-3 to T+5)
        window_dates = []
        for i in range(-WINDOW_DAYS_BEFORE, WINDOW_DAYS_AFTER + 1):
            window_dates.append(event_date + pd.Timedelta(days=i))

    year = event_date.year
    results = {f'{date.strftime("%Y-%m-%d")}': {} for date in window_dates}

    gdf = load_regions()
    masks_2d = {}

    # Basin masks on each product's grid. PRISM and Daymet masks are built
    # later from the coordinates of the data actually loaded.
    try:
        sample_geo = xr.open_dataset(PNNL_GEO)
        masks_2d['PNNL'] = get_mask(gdf, sample_geo.XLONG_M.values[0], sample_geo.XLAT_M.values[0]).any(dim='region')
    except Exception:
        pass

    try:
        sample_conus = xr.open_zarr(CONUS_ZARR)
        masks_2d['CONUS404'] = get_mask(gdf, sample_conus.lon, sample_conus.lat).any(dim='region')
        sample_conus.close()
    except Exception:
        pass

    try:
        ucla_static = xr.open_dataset(UCLA_STATIC)
        masks_2d['UCLA'] = get_mask(gdf, ucla_static.lon2d.values, ucla_static.lat2d.values).any(dim='region')
    except Exception:
        pass

    try:
        sample_gridmet = xr.open_zarr(GRIDMET_SAMPLE)
        masks_2d['GridMET'] = get_mask(gdf, sample_gridmet.lon, sample_gridmet.lat).any(dim='region')
    except Exception:
        pass

    # Extract PRISM from new format (TIFF)
    if 'prism' in products_to_extract:
        prism_data = {}

        for date in window_dates:
            try:
                da = load_prism_from_new_format(date, PRISM_DIR)
                if da is None:
                    continue
                lon = da.x.values
                lat = da.y.values
                m_prism = get_mask(gdf, lon, lat).any(dim='region')
                mean_val = calculate_basin_mean(da, m_prism)
                prism_data[date.normalize()] = mean_val
            except Exception:
                pass

        # Add extracted values to results
        for date in window_dates:
            normalized_date = date.normalize()
            if normalized_date in prism_data:
                results[f'{date.strftime("%Y-%m-%d")}']['prism'] = float(prism_data[normalized_date])

        # Debug: Print what was found
        prism_count = sum(1 for d in prism_data.values() if not np.isnan(d))
        if prism_count > 0:
            print(f"  ✓ PRISM: {prism_count}/{len(window_dates)} dates extracted")

    # Extract PNNL (hourly, summed to daily)
    if 'pnnl' in products_to_extract and year <= PNNL_LAST_YEAR:
        try:
            pnnl_path = os.path.join(PNNL_DIR, str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
            if os.path.exists(pnnl_path):
                ds = xr.open_dataset(pnnl_path)
                hourly_dates = []
                for d in window_dates:
                    hourly_dates.extend(pd.date_range(d, d + pd.Timedelta(hours=23), freq='h'))
                available_hourly = [d for d in hourly_dates if d in ds.time.values]
                if available_hourly and 'PNNL' in masks_2d:
                    da_subset = ds['PREC_ACC_NC'].sel(time=available_hourly)
                    s_hourly = calculate_basin_mean(da_subset, masks_2d['PNNL'])
                    pnnl_daily = s_hourly.resample('1D').sum()
                    pnnl_daily.index = pnnl_daily.index.normalize()
                    for date in window_dates:
                        normalized_date = date.normalize()
                        if normalized_date in pnnl_daily.index:
                            val = pnnl_daily[normalized_date]
                            results[f'{date.strftime("%Y-%m-%d")}']['pnnl'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ds.close()
        except Exception as e:
            print(f"Error loading PNNL for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract Daymet (new format - netCDF files)
    if 'daymet' in products_to_extract:
        try:
            ds = load_daymet_from_netcdf(year)
            if ds is not None:
                ds['time'] = pd.to_datetime(ds.time.values).normalize()
                da = ds['prcp']

                # Create Daymet mask dynamically if not cached
                if 'Daymet' not in masks_2d:
                    lon_2d = ds['lon'].values
                    lat_2d = ds['lat'].values
                    masks_2d['Daymet'] = get_mask(gdf, lon_2d, lat_2d).any(dim='region')

                da_window = da.sel(time=window_dates, method='nearest')
                daymet_daily = calculate_basin_mean(da_window, masks_2d['Daymet'])
                daymet_daily.index = daymet_daily.index.normalize()
                daymet_daily = daymet_daily[~daymet_daily.index.duplicated(keep='first')]
                for date in window_dates:
                    normalized_date = date.normalize()
                    if normalized_date in daymet_daily.index:
                        val = daymet_daily[normalized_date]
                        results[f'{date.strftime("%Y-%m-%d")}']['daymet'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ds.close()
        except Exception as e:
            print(f"Error loading Daymet for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract CONUS404
    if 'conus' in products_to_extract:
        try:
            if os.path.exists(CONUS_ZARR):
                ds = xr.open_zarr(CONUS_ZARR)
                ds['time'] = pd.to_datetime(ds.time.values).normalize()
                da = ds['precip_daily'].sel(time=window_dates, method='nearest')
                if 'CONUS404' in masks_2d:
                    conus_daily = calculate_basin_mean(da, masks_2d['CONUS404'])
                    conus_daily.index = conus_daily.index.normalize()
                    conus_daily = conus_daily[~conus_daily.index.duplicated(keep='first')]
                    for date in window_dates:
                        normalized_date = date.normalize()
                        if normalized_date in conus_daily.index:
                            val = conus_daily[normalized_date]
                            results[f'{date.strftime("%Y-%m-%d")}']['conus'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ds.close()
        except Exception as e:
            print(f"Error loading CONUS404 for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract UCLA
    if 'ucla' in products_to_extract:
        try:
            u_paths = [
                os.path.join(UCLA_PREC_DIR, f"prec.daily.era5.d02.{year-1}.nc"),
                os.path.join(UCLA_PREC_DIR, f"prec.daily.era5.d02.{year}.nc")
            ]
            series_list = []
            for p in u_paths:
                if os.path.exists(p):
                    try:
                        ds = xr.open_dataset(p)
                        u_var = "prec" if "prec" in ds.data_vars else "pr"
                        if 'day' in ds.dims:
                            ds = ds.rename({'day': 'time'})
                        ds['time'] = pd.to_datetime(ds.time.values).normalize()
                        available_dates = [d for d in window_dates if d in ds.time.values]
                        if available_dates and 'UCLA' in masks_2d:
                            da_subset = ds[u_var].sel(time=available_dates)
                            s = calculate_basin_mean(da_subset, masks_2d['UCLA'])
                            s.index = s.index.normalize()
                            series_list.append(s)
                        ds.close()
                    except Exception:
                        pass
            if series_list:
                ucla_daily = pd.concat(series_list).sort_index()
                ucla_daily = ucla_daily[~ucla_daily.index.duplicated(keep='first')]
                for date in window_dates:
                    normalized_date = date.normalize()
                    if normalized_date in ucla_daily.index:
                        val = ucla_daily[normalized_date]
                        results[f'{date.strftime("%Y-%m-%d")}']['ucla'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
        except Exception as e:
            print(f"Error loading UCLA for {event_date.strftime('%Y-%m-%d')}: {e}")

    # Extract GridMET
    if 'gridmet' in products_to_extract:
        try:
            gridmet_path = os.path.join(GRIDMET_DIR, f"{year}_daily_4km_gridMET_data.zarr")
            if os.path.exists(gridmet_path):
                ds = xr.open_zarr(gridmet_path)
                if 'day' in ds.dims:
                    ds = ds.rename({'day': 'time'})
                ds['time'] = pd.to_datetime(ds.time.values).normalize()
                var = 'prcp' if 'prcp' in ds.data_vars else 'precipitation_amount'
                da = ds[var].sel(time=window_dates, method='nearest')
                if 'GridMET' in masks_2d:
                    gridmet_daily = calculate_basin_mean(da, masks_2d['GridMET'])
                    gridmet_daily.index = gridmet_daily.index.normalize()
                    gridmet_daily = gridmet_daily[~gridmet_daily.index.duplicated(keep='first')]
                    for date in window_dates:
                        normalized_date = date.normalize()
                        if normalized_date in gridmet_daily.index:
                            val = gridmet_daily[normalized_date]
                            results[f'{date.strftime("%Y-%m-%d")}']['gridmet'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ds.close()
        except Exception as e:
            print(f"Error loading GridMET for {event_date.strftime('%Y-%m-%d')}: {e}")

    return pd.DataFrame(results).T


def save_timeseries_to_csv(all_event_data, output_dir):
    """Save all cumulative precipitation timeseries to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for event_date, event_df in all_event_data.items():
        event_str = event_date.strftime('%Y%m%d')
        csv_path = os.path.join(output_dir, f'cumulative_precip_{event_str}.csv')

        # Compute cumulative sum for each product
        cumul_df = event_df.cumsum()
        cumul_df.index.name = 'datetime'
        cumul_df = cumul_df.reset_index()

        cumul_df.to_csv(csv_path, index=False)
        print(f"  Saved cumulative data: {csv_path}")


def plot_specific_ar_events():
    """Extract and plot specific AR events"""
    print("Loading events data...")
    events_df = pd.read_csv(EVENTS_CSV)
    events_df['date'] = pd.to_datetime(events_df['date'])

    print("Loading discharge data...")
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    # Convert specific dates to datetime and extract date/range info
    events_info = []
    for d in SPECIFIC_DATES:
        if isinstance(d, dict):
            events_info.append(d)
        else:
            events_info.append({'event_date': d})

    event_dates = [pd.to_datetime(info['event_date']) for info in events_info]

    print(f"Extracting and plotting {len(event_dates)} specific AR events...")

    # Extract data for all events and calculate max cumulative precipitation
    products = PRODUCTS
    max_cumsum = 0
    all_event_data = {}

    for event_date, event_info in zip(event_dates, events_info):
        print(f"Extracting data for {event_date.strftime('%Y-%m-%d')}...")
        start_date = event_info.get('start_date')
        end_date = event_info.get('end_date')
        window_df = extract_event_window(event_date, products_to_extract=products, start_date=start_date, end_date=end_date)
        all_event_data[event_date] = window_df

        for product in products:
            if product in window_df.columns:
                cumsum = window_df[product].cumsum()
                valid_values = cumsum[cumsum.notna()].max() if len(cumsum[cumsum.notna()]) > 0 else np.nan
                if not np.isnan(valid_values):
                    max_cumsum = max(max_cumsum, valid_values)

    ylim_max = max_cumsum * 1.05

    # Save timeseries data to CSV files
    print("\nSaving timeseries data to CSV files...")
    csv_output_dir = os.path.join(OUTPUT_DIR, 'timeseries_data')
    save_timeseries_to_csv(all_event_data, csv_output_dir)

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE)
    axes = axes.flatten()

    for idx, event_date in enumerate(event_dates):
        ax = axes[idx]

        # Get data for this event
        event_data = all_event_data[event_date].copy()
        event_data.index = pd.to_datetime(event_data.index)
        event_data = event_data.sort_index()

        # Get AR scale
        event_row = events_df[events_df['date'] == event_date]
        ar_scale = event_row['ar_scale'].values[0] if len(event_row) > 0 else None

        # Find maximum discharge in event window
        event_start = event_data.index.min()
        event_end = event_data.index.max()
        discharge_window = q_df[(q_df['date'] >= event_start) & (q_df['date'] <= event_end)]
        max_discharge_cfs = discharge_window['discharge_cfs'].max() if len(discharge_window) > 0 else None
        discharge_cms = max_discharge_cfs * CFS_TO_CMS if max_discharge_cfs is not None and max_discharge_cfs > 0 else None

        # Plot each product
        for prod_idx, product in enumerate(products):
            if product in event_data.columns:
                cumsum = event_data[product].cumsum()
                ax.plot(cumsum.index, cumsum, label=product.upper(),
                       marker=PRODUCT_MARKERS[prod_idx], linewidth=2, markersize=5,
                       color=PRODUCT_COLORS[prod_idx], alpha=0.8)

        ar_label = f'(AR = {ar_scale} | Discharge = {discharge_cms:.0f} cms)' if ar_scale is not None else ''
        ax.set_title(f'{event_date.strftime("%Y-%m-%d")} {ar_label}', fontsize=FONT_TITLE, fontweight='bold')
        if idx >= len(axes) - 2:
            ax.set_xlabel('Date', fontsize=FONT_LABEL)
        if idx % 2 == 0:
            ax.set_ylabel('Cumulative\nPrecipitation (mm)', fontsize=FONT_LABEL)
        ax.set_ylim(0, ylim_max)
        ax.legend(loc='upper left', fontsize=FONT_LEGEND, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Cumulative Precipitation during Top Discharge AR Events', fontsize=FONT_SUPTITLE, fontweight='bold')
    plt.tight_layout(h_pad=2.0)

    output_path = os.path.join(OUTPUT_DIR, 'specific_ar_events_cumulative_precipitation.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_specific_ar_events()
