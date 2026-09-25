"""Extract and plot cumulative precipitation for specific non-AR events in the Skagit Basin.

Loads precipitation data from multiple sources (PRISM, PNNL, Daymet, CONUS404, UCLA, GridMET),
computes basin-averaged values for an 8-day window around each event date (T-3 to T+5),
and generates comparison plots showing cumulative precipitation curves overlaid with discharge data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import geopandas as gpd
import regionmask
import zipfile
import tempfile
import rioxarray

# Configuration
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
EVENTS_CSV = os.path.join(BASE_DIR, "multi_product_bulk_bias/outputs/4_clean_bias_table.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "cumulative_precipitation_plot")

# Hydrology data paths
HYDRO_BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
HYDRO_EXP_DATA_DIR = os.path.join(HYDRO_BASE_DIR, "experiments_2/data")
HYDRO_Q_PATH = os.path.join(HYDRO_EXP_DATA_DIR, "usgs_12200500_discharge.rdb")

# Specific event dates to plot
SPECIFIC_DATES = [
    # '2021-11-16',
    '1995-12-02',
    # '1990-11-14',
    '2011-01-18',
    '2015-11-14',
    '2007-03-13',
    # '2021-12-03',
    '2010-12-14',
    '2009-11-27',
]

# 11/16/21
# 12/2/95
# 1/18/11
# 11/14/15
# 11/19/21
# 3/13/07
# 12/3/21
# 11/26/95
# 11/28/11
# 12/14/10
# 11/27/09
# 12/12/04
# 12/4/95

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
    except Exception as e:
        pass

    return None


def load_daymet_from_netcdf(year):
    """Load Daymet data from netCDF file for a specific year"""
    daymet_nc_path = os.path.join(VAULT_DIR, "daymet_new_hq", f"prcp_{year}_subset.nc")

    if not os.path.exists(daymet_nc_path):
        return None

    try:
        ds = xr.open_dataset(daymet_nc_path)
        return ds
    except Exception as e:
        print(f"  Error loading Daymet netCDF for {year}: {e}")
        return None


def get_ornl_zarr(year):
    if 1981 <= year <= 2011:
        p = os.path.join(VAULT_DIR, "climate_sets/1981_2011_ORNL_data.zarr")
        if os.path.exists(p):
            return p
    p1 = os.path.join(VAULT_DIR, f"ornl/{year}_{year}_ref_DaymetV4_ORNL_data.zarr")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(VAULT_DIR, f"DaymetV4/{year}_{year}_ORNL_data.zarr")
    if os.path.exists(p2):
        return p2
    return None


def extract_event_window(event_date, products_to_extract=['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']):
    """Extract 8-day precipitation window (T-2 to T+5) for an event"""

    window_dates = []
    for i in range(-3, 6):
        window_dates.append(event_date + pd.Timedelta(days=i))

    year = event_date.year
    results = {f'{date.strftime("%Y-%m-%d")}': {} for date in window_dates}

    gdf = load_regions()
    masks_2d = {}

    try:
        sample_prism = xr.open_zarr(os.path.join(VAULT_DIR, "PRISM/1991-2000/1996-01-01_1996-12-31_daily_4km_PRISM_data.zarr"), consolidated=False)
        masks_2d['PRISM'] = get_mask(gdf, sample_prism.lon, sample_prism.lat).any(dim='region')
        sample_prism.close()
    except:
        pass

    try:
        sample_geo = xr.open_dataset(os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc"))
        masks_2d['PNNL'] = get_mask(gdf, sample_geo.XLONG_M.values[0], sample_geo.XLAT_M.values[0]).any(dim='region')
    except:
        pass

    # Daymet mask will be created dynamically in extraction (from actual data coordinates)

    try:
        sample_conus = xr.open_zarr(os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr"))
        masks_2d['CONUS404'] = get_mask(gdf, sample_conus.lon, sample_conus.lat).any(dim='region')
        sample_conus.close()
    except:
        pass

    try:
        ucla_static = xr.open_dataset(os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc"))
        masks_2d['UCLA'] = get_mask(gdf, ucla_static.lon2d.values, ucla_static.lat2d.values).any(dim='region')
    except:
        pass

    try:
        sample_gridmet = xr.open_zarr(os.path.join(VAULT_DIR, "gridmet/2021_daily_4km_gridMET_data.zarr"))
        masks_2d['GridMET'] = get_mask(gdf, sample_gridmet.lon, sample_gridmet.lat).any(dim='region')
    except:
        pass

    # Extract PRISM from new format (TIFF)
    if 'prism' in products_to_extract:
        prism_root = os.path.join(VAULT_DIR, "prism_new_hq")
        prism_data = {}

        for date in window_dates:
            try:
                da = load_prism_from_new_format(date, prism_root)
                if da is None:
                    continue
                lon = da.x.values
                lat = da.y.values
                m_prism = get_mask(gdf, lon, lat).any(dim='region')
                mean_val = calculate_basin_mean(da, m_prism)
                prism_data[date.normalize()] = mean_val
            except Exception as date_error:
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

    # Extract PNNL
    if 'pnnl' in products_to_extract and year <= 2020:
        try:
            pnnl_path = os.path.join(VAULT_DIR, "PNNL/historical", str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
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
            conus_path = os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
            if os.path.exists(conus_path):
                ds = xr.open_zarr(conus_path)
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
                os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year-1}.nc"),
                os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year}.nc")
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
                    except:
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
            gridmet_path = os.path.join(VAULT_DIR, f"gridmet/{year}_daily_4km_gridMET_data.zarr")
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


def plot_specific_ar_events():
    """Extract and plot specific AR events"""
    print("Loading events data...")
    events_df = pd.read_csv(EVENTS_CSV)
    events_df['date'] = pd.to_datetime(events_df['date'])

    print("Loading discharge data...")
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    # Convert specific dates to datetime
    event_dates = [pd.to_datetime(d) for d in SPECIFIC_DATES]

    print(f"Extracting and plotting {len(event_dates)} specific AR events...")

    # Extract data for all events and calculate max cumulative precipitation
    products = ['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']
    max_cumsum = 0
    all_event_data = {}

    for event_date in event_dates:
        print(f"Extracting data for {event_date.strftime('%Y-%m-%d')}...")
        window_df = extract_event_window(event_date, products_to_extract=products)
        all_event_data[event_date] = window_df

        for product in products:
            if product in window_df.columns:
                cumsum = window_df[product].cumsum()
                max_cumsum = max(max_cumsum, cumsum.max())

    ylim_max = max_cumsum * 1.05

    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

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
        discharge_cms = max_discharge_cfs * 0.0283168 if max_discharge_cfs is not None and max_discharge_cfs > 0 else None

        # Plot each product
        for prod_idx, product in enumerate(products):
            if product in event_data.columns:
                cumsum = event_data[product].cumsum()
                ax.plot(cumsum.index, cumsum, label=product.upper(),
                       marker=markers[prod_idx], linewidth=2, markersize=5,
                       color=colors[prod_idx], alpha=0.8)

        ar_label = f'(AR = {ar_scale} | Discharge = {discharge_cms:.0f} cms)' if ar_scale is not None else ''
        ax.set_title(f'Event: {event_date.strftime("%Y-%m-%d")} {ar_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=10)
        ax.set_ylim(0, ylim_max)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Cumulative Precipitation during Top Discharge non-AR Events', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'specific_non_ar_events_cumulative_precipitation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_specific_ar_events()
