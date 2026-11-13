"""Cumulative precipitation of six products over selected months, with events marked.

For each month in ``SPECIFIC_DATES``, daily basin-mean precipitation over the
Skagit Basin (Upper Skagit, Sauk and Lower Skagit HUC8 sub-basins) is
extracted from six gridded products (PRISM, PNNL, Daymet, CONUS404, UCLA,
GridMET) and the cumulative totals are plotted in a 1x3 panel figure.
USGS 12200500 discharge is drawn in grey on a secondary y-axis, and every
sampled event in the month is marked by a dashed vertical line labelled with
its AR scale (red: AR event, scale > 0; blue: non-AR event, scale 0). All
panels share the same y-axis ranges.

Main functions:
  extract_event_window: Extract precipitation for a date range across all products
  plot_specific_months: Generate the cumulative precipitation figure for the selected months

Input
    /data0/skagit_met/data_transfer/data/...      (gridded precipitation products)
    data/GIS/SkagitSubBasin_HUC8.geojson
    data/weather_data/conus404_skagit_precip_daily_full.zarr
    data/usgs_12200500_discharge.rdb
    analysis/outputs/5day_taking_min_and_filling/4_clean_bias_table.csv
Output
    plots/plot_specific_ar_and_non_ar_events/specific_months_cumulative_precipitation.png
    plots/plot_specific_ar_and_non_ar_events/timeseries_data/cumulative_precip_<YYYYMMDD>.csv
        (one per month, named after the 15th of the month)

Usage
    python plot_specific_months.py
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

# Default event window when no date range is given: T-3 to T+5 days
WINDOW_DAYS_BEFORE = 3
WINDOW_DAYS_AFTER = 5

CFS_TO_CMS = 0.0283168

# Specific months to plot - each with start and end dates
# Will plot all precipitation data for the month and mark AR events within that period
SPECIFIC_DATES = [
    {'start_date': '2006-11-01', 'end_date': '2006-11-30'},
    {'start_date': '1995-11-01', 'end_date': '1995-11-30'},
    {'start_date': '1997-03-01', 'end_date': '1997-03-30'},
]

# Event marker colors: AR events (scale > 0) and non-AR events (scale 0)
AR_LINE_COLOR = 'red'
NON_AR_LINE_COLOR = 'blue'

# Figure settings (1 row x 3 columns, one panel per month)
FIGSIZE = (16, 5.5)
DPI = 300

# Publication font sizes (figure is 16 in wide; expect ~0.45x reduction to journal page width)
FONT_SUPTITLE = 22
FONT_TITLE = 18
FONT_LABEL = 16
FONT_TICK = 14
FONT_LEGEND = 12
FONT_AR_SCALE = 14  # AR scale numbers above the dashed lines


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
    """Extract daily basin-mean precipitation for each product.

    ``start_date``/``end_date`` give the date range (here, a whole month);
    without them a 9-day window (T-3 to T+5) around ``event_date`` is used.
    ``event_date`` also sets the year of the files that are read. Returns a
    DataFrame indexed by date string with one column per product.
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


def plot_specific_months():
    """Extract and plot specific months with AR event markers"""
    print("Loading events data...")
    events_df = pd.read_csv(EVENTS_CSV)
    events_df['date'] = pd.to_datetime(events_df['date'])

    print("Loading discharge data...")
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    # Process date ranges for each month
    months_info = []
    for date_range in SPECIFIC_DATES:
        start_date = pd.to_datetime(date_range['start_date'])
        end_date = pd.to_datetime(date_range['end_date'])
        # Use middle of month as reference event date for extraction
        event_date = start_date + pd.Timedelta(days=14)
        months_info.append({
            'event_date': event_date,
            'start_date': start_date,
            'end_date': end_date
        })

    print(f"Extracting and plotting {len(months_info)} months...")

    # Extract data for all months and calculate max cumulative precipitation
    products = PRODUCTS
    max_cumsum = 0
    all_event_data = {}

    for month_info in months_info:
        event_date = month_info['event_date']
        start_date = month_info['start_date']
        end_date = month_info['end_date']
        print(f"Extracting data for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

        window_df = extract_event_window(event_date, products_to_extract=products,
                                        start_date=start_date.strftime('%Y-%m-%d'),
                                        end_date=end_date.strftime('%Y-%m-%d'))
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

    # Calculate max discharge across all months for consistent y-axis
    max_discharge = 0
    for month_info in months_info:
        start_date = month_info['start_date']
        end_date = month_info['end_date']
        discharge_window = q_df[(q_df['date'] >= start_date) & (q_df['date'] <= end_date)]
        if len(discharge_window) > 0:
            max_discharge = max(max_discharge, discharge_window['discharge_cfs'].max() * CFS_TO_CMS)
    discharge_ylim_max = max_discharge * 1.05 if max_discharge > 0 else 1

    # Create plots - 1 row x 3 columns for 3 months
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    for idx, month_info in enumerate(months_info):
        ax = axes[idx]
        event_date = month_info['event_date']
        start_date = month_info['start_date']
        end_date = month_info['end_date']

        # Get data for this month
        event_data = all_event_data[event_date].copy()
        event_data.index = pd.to_datetime(event_data.index)
        event_data = event_data.sort_index()

        # Plot each product on primary axis
        for prod_idx, product in enumerate(products):
            if product in event_data.columns:
                cumsum = event_data[product].cumsum()
                ax.plot(cumsum.index, cumsum, label=product.upper(),
                       marker=PRODUCT_MARKERS[prod_idx], linewidth=2, markersize=5,
                       color=PRODUCT_COLORS[prod_idx], alpha=0.8)

        # Create secondary axis for streamflow
        ax2 = ax.twinx()
        discharge_window = q_df[(q_df['date'] >= start_date) & (q_df['date'] <= end_date)]
        if len(discharge_window) > 0:
            ax2.plot(discharge_window['date'], discharge_window['discharge_cfs'] * CFS_TO_CMS,
                    color='grey', linewidth=2, alpha=0.7)
            if idx == len(months_info) - 1:
                ax2.set_ylabel('Discharge (cms)', fontsize=FONT_LABEL, color='grey')
            ax2.tick_params(axis='y', labelcolor='grey', labelsize=FONT_TICK)
            ax2.set_ylim(0, discharge_ylim_max)

        # Add vertical lines for AR events within this month
        ar_events_in_month = events_df[(events_df['date'] >= start_date) & (events_df['date'] <= end_date)]
        for _, ar_event in ar_events_in_month.iterrows():
            ar_date = ar_event['date']
            ar_scale = int(ar_event['ar_scale'])

            if ar_scale == 0:
                # Dashed line for non-AR events (AR scale 0)
                ax.axvline(x=ar_date, color=NON_AR_LINE_COLOR, linestyle='--', alpha=0.6, linewidth=1.5)
                line_color = NON_AR_LINE_COLOR
            else:
                # Dashed line for AR events (AR scale > 0)
                ax.axvline(x=ar_date, color=AR_LINE_COLOR, linestyle='--', alpha=0.6, linewidth=1.5)
                line_color = AR_LINE_COLOR

            ax.text(ar_date, ylim_max * 1.02, f'{ar_scale}', rotation=0,
                   verticalalignment='bottom', horizontalalignment='center',
                   color=line_color, fontsize=FONT_AR_SCALE, fontweight='bold', alpha=0.8, clip_on=False)

        month_label = start_date.strftime('%B %Y')
        ax.set_title(f'{month_label}', fontsize=FONT_TITLE, fontweight='bold', y=1.08)
        ax.set_xlabel('Date', fontsize=FONT_LABEL)
        if idx == 0:
            ax.set_ylabel('Cumulative\nPrecipitation (mm)', fontsize=FONT_LABEL)
        ax.set_ylim(0, ylim_max)
        ax.legend(loc='upper left', fontsize=FONT_LEGEND, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Cumulative Precipitation with AR Events', fontsize=FONT_SUPTITLE, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'specific_months_cumulative_precipitation.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_specific_months()
