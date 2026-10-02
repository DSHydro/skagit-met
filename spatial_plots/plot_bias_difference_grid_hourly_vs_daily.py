"""Generate 6x5 grid comparing hourly vs daily bias differences for AR events.

Creates a grid showing the DIFFERENCE in bias calculations (hourly - daily) for all products.
For products like Daymet, PNNL, GridMET, the difference is zero since they use the same approach.
For CONUS404 and UCLA, shows the actual difference from switching to hourly data.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from pathlib import Path
from datetime import timedelta
import warnings
import tempfile
import zipfile
import rioxarray

warnings.filterwarnings('ignore')

# --- Configuration & Paths ---
BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
DATA_DIR = "/data0/hernanqd/plots_code/skagit-met/data"
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PRISM_ROOT = os.path.join(VAULT_DIR, "prism_new_hq")
DAYMET_ROOT = os.path.join(VAULT_DIR, "daymet_new_hq")
BOUNDARY_PATH = os.path.join(BASE_DIR, "data/GIS/SkagitBoundary.json")
SUBBASIN_PATH = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
OUT_DIR = "/data0/hernanqd/plots_code/skagit-met/spatial_plots/plots"
os.makedirs(OUT_DIR, exist_ok=True)

AR_EVENTS = [
    {"label": "November_1990_AR5",   "start": "1990-11-22", "end": "1990-11-29", "row_label": "Nov 22–29, 1990\n(AR5)"},
    # {"label": "November_1995_AR4",  "start": "1995-11-27", "end": "1995-12-04", "row_label": "Nov 27–Dec 4, 1995\n(AR4)"},
    # {"label": "November_1990_AR4",  "start": "1990-11-08", "end": "1990-11-15", "row_label": "Nov 8–15, 1990\n(AR4)"},
    # {"label": "November_2006_AR5",  "start": "2006-11-05", "end": "2006-11-12", "row_label": "Nov 5–12, 2006\n(AR5)"},
    # {"label": "October_2003_AR5",  "start": "2003-10-19", "end": "2003-10-26", "row_label": "Oct 19–26, 2003\n(AR5)"},
    # {"label": "November_2021_AR4",  "start": "2021-11-13", "end": "2021-11-20", "row_label": "Nov 13–20, 2021\n(AR4)"}
]

PRODUCTS = ['PRISM', 'Daymet', 'PNNL', 'CONUS404', 'UCLA', 'GridMET']
BIAS_PRODUCTS = [p for p in PRODUCTS if p != 'PRISM']

BB = (-122.5, -120.5, 47.8, 49.5)
MAP_EXTENT = [-122.35, -120.65, 47.90, 49.35]


def bb_mask(lon2d, lat2d):
    return (lon2d >= BB[0]) & (lon2d <= BB[1]) & (lat2d >= BB[2]) & (lat2d <= BB[3])


def crop_rows_cols(lon2d, lat2d):
    m = bb_mask(lon2d, lat2d)
    rows, cols = np.where(m)
    return rows.min(), rows.max(), cols.min(), cols.max()


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


def load_prism_day(date, prism_root):
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
                            da = da.squeeze().copy(deep=True)
                            da = da.rename({'x': 'lon', 'y': 'lat'})
                            da.load()
                            return da
    except Exception as e:
        print(f"    Error loading PRISM: {e}")
    return None


def load_daymet_from_netcdf(year, daymet_root):
    daymet_nc_path = os.path.join(daymet_root, f"prcp_{year}_subset.nc")
    if not os.path.exists(daymet_nc_path):
        return None
    try:
        return xr.open_dataset(daymet_nc_path)
    except Exception as e:
        print(f"    Error loading Daymet: {e}")
        return None


def load_conus_daily(start, end):
    try:
        conus_path = os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
        ds = xr.open_zarr(conus_path)
        da = ds['precip_daily'].sel(time=slice(start, end)).sum(dim='time', skipna=False).compute()
        ds.close()
        return da
    except:
        return None


def load_conus_hourly(start, end):
    try:
        conus_data_path = Path("/data0/hernanqd/instance_2021_data/preparing_datasets/CONUS404/hourly_ar_non_ar_events")
        conus_files_list = sorted(conus_data_path.glob('*.PREC_ACC_NC.wrf2d_d01_*.nc'))
        conus_files_in_range = []
        for f in conus_files_list:
            try:
                ds_temp = xr.open_dataset(f)
                time = ds_temp.Time.values[0]
                if pd.Timestamp(start) <= pd.Timestamp(time) <= pd.Timestamp(end):
                    conus_files_in_range.append(f)
                ds_temp.close()
            except:
                pass
        if conus_files_in_range:
            ds_conus_list = [xr.open_dataset(f) for f in conus_files_in_range]
            ds_conus = xr.concat(ds_conus_list, dim='Time')
            da = ds_conus['PREC_ACC_NC'].sum(dim='Time', skipna=False).compute()
            da = da.assign_coords(lon=(('south_north', 'west_east'), ds_conus.XLONG.values),
                                  lat=(('south_north', 'west_east'), ds_conus.XLAT.values))
            return da
    except:
        pass
    return None


def load_ucla_daily(start, end):
    try:
        year = int(start[:4])
        da_parts = []
        for yr_off in [year-1, year]:
            p = os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{yr_off}.nc")
            if os.path.exists(p):
                file_start = f"{yr_off}-09-01"
                file_end = f"{yr_off+1}-08-31"
                s_start = max(start, file_start)
                s_end = min(end, file_end)
                if s_start <= s_end:
                    ds_u = xr.open_dataset(p)
                    u_var = "prec" if "prec" in ds_u.data_vars else "pr"
                    da_part = ds_u[u_var].sel(day=slice(s_start, s_end)).compute()
                    da_parts.append(da_part)
                    ds_u.close()
        if da_parts:
            da_year = xr.concat(da_parts, dim='day')

            # Load UCLA coordinates for proper regridding
            ds_ucla_static = xr.open_dataset("/data0/hernanqd/instance_2021_data/preparing_datasets/UCLA/wrfinput_d02_coord.nc")
            ucla_lon_full = ds_ucla_static.lon2d.values
            ucla_lat_full = ds_ucla_static.lat2d.values
            ds_ucla_static.close()

            # Crop to bounding box BEFORE summing
            bb_mask_ucla = (ucla_lon_full >= BB[0]) & (ucla_lon_full <= BB[1]) & (ucla_lat_full >= BB[2]) & (ucla_lat_full <= BB[3])
            rows, cols = np.where(bb_mask_ucla)
            if len(rows) > 0:
                rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
                # Daily UCLA uses lat2d/lon2d as coordinate variables
                da_year = da_year.isel(lat2d=slice(rm, rx+1), lon2d=slice(cm, cx+1))
                da = da_year.sum(dim='day', skipna=False)
                lat_c = ucla_lat_full[rm:rx+1, cm:cx+1]
                lon_c = ucla_lon_full[rm:rx+1, cm:cx+1]
                da = da.assign_coords(lat=(('lat2d', 'lon2d'), lat_c),
                                     lon=(('lat2d', 'lon2d'), lon_c))
                return da
    except Exception as e:
        print(f"    [DEBUG] UCLA daily error: {e}")
    return None


def load_ucla_hourly(start, end):
    try:
        from pathlib import Path
        ucla_data_path = Path("/data0/hernanqd/instance_2021_data/hourly_ar_non_ar_events")
        date_start_ucla = pd.Timestamp(start) - timedelta(hours=1)
        date_end = pd.Timestamp(end)
        current = date_start_ucla
        ucla_filenames = []
        while current <= date_end:
            fn = f"auxhist_d01_{current.strftime('%Y-%m-%d_%H:%M:%S')}.nc"
            ucla_filenames.append(ucla_data_path / fn)
            current += timedelta(hours=1)
        ucla_files_exist = [f for f in ucla_filenames if f.exists()]
        if ucla_files_exist:
            ds_ucla = xr.open_mfdataset(ucla_files_exist, concat_dim='Time', combine='nested')
            total_rain_ucla = ds_ucla['RAINC'] + ds_ucla['RAINNC']
            total_increments_ucla = total_rain_ucla.diff(dim='Time')
            da = total_increments_ucla.sum(dim='Time', skipna=False).compute()

            # Load UCLA coordinates for proper regridding
            ds_ucla_static = xr.open_dataset("/data0/hernanqd/instance_2021_data/preparing_datasets/UCLA/wrfinput_d02_coord.nc")
            ucla_lon_full = ds_ucla_static.lon2d.values
            ucla_lat_full = ds_ucla_static.lat2d.values
            ds_ucla_static.close()

            # Crop to bounding box
            bb_mask_ucla = (ucla_lon_full >= BB[0]) & (ucla_lon_full <= BB[1]) & (ucla_lat_full >= BB[2]) & (ucla_lat_full <= BB[3])
            rows, cols = np.where(bb_mask_ucla)
            if len(rows) > 0:
                rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
                da_cropped = da.isel(south_north=slice(rm, rx+1), west_east=slice(cm, cx+1))
                lat_c = ucla_lat_full[rm:rx+1, cm:cx+1]
                lon_c = ucla_lon_full[rm:rx+1, cm:cx+1]
                da_cropped = da_cropped.assign_coords(lat=(('south_north', 'west_east'), lat_c),
                                                       lon=(('south_north', 'west_east'), lon_c))
                ds_ucla.close()
                return da_cropped
            ds_ucla.close()
    except Exception as e:
        print(f"    [DEBUG] UCLA hourly error: {e}")
    return None


def regrid_to_reference(da_src, ref_lon, ref_lat, mask_2d=None):
    if da_src is None:
        return np.full(ref_lat.shape, np.nan)
    vals = da_src.values
    if np.all(np.isnan(vals)):
        return np.full(ref_lat.shape, np.nan)

    lon_name = next((c for c in da_src.coords if 'lon' in c or 'longitude' in c), None)
    lat_name = next((c for c in da_src.coords if 'lat' in c or 'latitude' in c), None)
    if lon_name is None or lat_name is None:
        return np.full(ref_lat.shape, np.nan)

    lon_vals = da_src.coords[lon_name].values
    lat_vals = da_src.coords[lat_name].values

    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        lon_g, lat_g = np.meshgrid(lon_vals, lat_vals)
    else:
        lon_g, lat_g = lon_vals, lat_vals

    pts = np.column_stack((lon_g.flatten(), lat_g.flatten()))
    v   = vals.flatten()
    ok  = ~np.isnan(v)
    if not ok.any():
        return np.full(ref_lat.shape, np.nan)

    grid_z = griddata(pts[ok], v[ok], (ref_lon, ref_lat), method='linear')
    if mask_2d is not None:
        grid_z[~mask_2d] = np.nan
    return grid_z


def main():
    print("Generating 6x5 bias difference grid (hourly - daily)\n")

    # Load static coordinates
    print("Loading static grid coordinates...")
    ds_pnnl_static = xr.open_dataset(os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc"))
    pnnl_lon_full = ds_pnnl_static.XLONG_M.values[0]
    pnnl_lat_full = ds_pnnl_static.XLAT_M.values[0]
    ds_pnnl_static.close()

    ds_ucla_static = xr.open_dataset("/data0/hernanqd/instance_2021_data/preparing_datasets/UCLA/wrfinput_d02_coord.nc")
    ucla_lon = ds_ucla_static.lon2d.values
    ucla_lat = ds_ucla_static.lat2d.values
    ds_ucla_static.close()

    # Load GIS boundaries
    boundary_gdf = gpd.read_file(BOUNDARY_PATH)
    subbasin_gdf = gpd.read_file(SUBBASIN_PATH) if os.path.exists(SUBBASIN_PATH) else None

    # Load reference PRISM grid
    print("Loading reference PRISM grid...")
    prism_sample_date = pd.Timestamp("2020-01-01")
    da_prism_sample = load_prism_day(prism_sample_date, PRISM_ROOT)
    if da_prism_sample is None:
        print("Error: Could not load PRISM")
        return

    prism_lon_full = da_prism_sample['lon'].values
    prism_lat_full = da_prism_sample['lat'].values

    if prism_lon_full.ndim == 1 and prism_lat_full.ndim == 1:
        prism_lon_2d, prism_lat_2d = np.meshgrid(prism_lon_full, prism_lat_full)
    else:
        prism_lon_2d, prism_lat_2d = prism_lon_full, prism_lat_full

    row_min_p, row_max_p, col_min_p, col_max_p = crop_rows_cols(prism_lon_2d, prism_lat_2d)
    ref_lon = prism_lon_2d[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
    ref_lat = prism_lat_2d[row_min_p:row_max_p+1, col_min_p:col_max_p+1]

    # Load watershed mask
    boundary_gdf = gpd.read_file(BOUNDARY_PATH)
    if hasattr(boundary_gdf.to_crs("EPSG:4326"), 'union_all'):
        poly = boundary_gdf.to_crs("EPSG:4326").union_all()
    else:
        poly = boundary_gdf.to_crs("EPSG:4326").unary_union

    df_pts = pd.DataFrame({'lon': ref_lon.flatten(), 'lat': ref_lat.flatten()})
    gdf_pts = gpd.GeoDataFrame(df_pts, geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat), crs="EPSG:4326")
    inside = gdf_pts.intersects(poly).values
    mask_2d = inside.reshape(ref_lon.shape)

    # Create 6x5 grid
    fig, axes = plt.subplots(
        len(AR_EVENTS), len(BIAS_PRODUCTS),
        figsize=(15, 17.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor='#ffffff'
    )
    # Ensure axes is always 2D
    if axes.ndim == 1:
        axes = axes.reshape(len(AR_EVENTS), len(BIAS_PRODUCTS))

    # Collect all differences to determine dynamic colorbar limits
    all_differences = []

    print("Processing events...\n")
    bias_grids_daily = {}
    bias_grids_hourly = {}

    for event in AR_EVENTS:
        start = event["start"]
        end = event["end"]
        label = event["label"]
        year = int(start[:4])

        print(f"  {label}...")
        bias_grids_daily[label] = {}
        bias_grids_hourly[label] = {}

        # Load PRISM
        date_range = pd.date_range(start, end, freq='D')
        prism_das = []
        for d in date_range:
            da_day = load_prism_day(d, PRISM_ROOT)
            if da_day is not None:
                da_day = da_day.where(da_day >= 0)
                prism_das.append(da_day.values)

        prism_data = np.nansum(prism_das, axis=0) if prism_das else None
        if prism_data is not None:
            da_prism = xr.DataArray(
                prism_data,
                coords={'lon': (['y', 'x'], prism_lon_2d), 'lat': (['y', 'x'], prism_lat_2d)},
                dims=['y', 'x']
            )
            prism_regrid = regrid_to_reference(da_prism, ref_lon, ref_lat, mask_2d)
        else:
            prism_regrid = np.full(ref_lat.shape, np.nan)

        # Load other products (DAILY approach)
        # Daymet
        ds_daymet = load_daymet_from_netcdf(year, DAYMET_ROOT)
        if ds_daymet is not None:
            ds_daymet['time'] = pd.to_datetime(ds_daymet.time.values).normalize()
            da_daymet = ds_daymet['prcp'].sel(time=slice(start, end)).sum(dim='time', skipna=True).compute()
            daymet_regrid = regrid_to_reference(da_daymet, ref_lon, ref_lat, mask_2d)
            bias_grids_daily[label]['Daymet'] = daymet_regrid - prism_regrid
            bias_grids_hourly[label]['Daymet'] = daymet_regrid - prism_regrid
            ds_daymet.close()

        # PNNL
        pnnl_file = os.path.join(VAULT_DIR, "PNNL/historical", str(year),
                                 f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        if os.path.exists(pnnl_file):
            ds = xr.open_dataset(pnnl_file, chunks={'time': 720})
            da_daily = ds['PREC_ACC_NC'].resample(time='1D').sum()
            mask_c = bb_mask(pnnl_lon_full, pnnl_lat_full)
            rows, cols = np.where(mask_c)
            if len(rows) > 0:
                rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
                da_daily = da_daily.isel(x=slice(rm, rx+1), y=slice(cm, cx+1))
                lat_c = pnnl_lat_full[rm:rx+1, cm:cx+1]
                lon_c = pnnl_lon_full[rm:rx+1, cm:cx+1]
                da = da_daily.sel(time=slice(start, end)).sum(dim='time', skipna=True).compute()
                da = da.assign_coords(lat=(('x', 'y'), lat_c), lon=(('x', 'y'), lon_c))
                pnnl_regrid = regrid_to_reference(da, ref_lon, ref_lat, mask_2d)
                bias_grids_daily[label]['PNNL'] = pnnl_regrid - prism_regrid
                bias_grids_hourly[label]['PNNL'] = pnnl_regrid - prism_regrid
            ds.close()

        # CONUS404
        conus_daily = load_conus_daily(start, end)
        conus_hourly = load_conus_hourly(start, end)
        if conus_daily is not None:
            conus_daily_regrid = regrid_to_reference(conus_daily, ref_lon, ref_lat, mask_2d)
            bias_grids_daily[label]['CONUS404'] = conus_daily_regrid - prism_regrid
        if conus_hourly is not None:
            conus_hourly_regrid = regrid_to_reference(conus_hourly, ref_lon, ref_lat, mask_2d)
            bias_grids_hourly[label]['CONUS404'] = conus_hourly_regrid - prism_regrid

        # UCLA
        ucla_daily = load_ucla_daily(start, end)
        ucla_hourly = load_ucla_hourly(start, end)
        if ucla_daily is not None:
            ucla_daily_regrid = regrid_to_reference(ucla_daily, ref_lon, ref_lat, mask_2d)
            bias_grids_daily[label]['UCLA'] = ucla_daily_regrid - prism_regrid
        if ucla_hourly is not None:
            ucla_hourly_regrid = regrid_to_reference(ucla_hourly, ref_lon, ref_lat, mask_2d)
            bias_grids_hourly[label]['UCLA'] = ucla_hourly_regrid - prism_regrid

        # GridMET
        gridmet_path = os.path.join(VAULT_DIR, "gridmet", f"{year}_daily_4km_gridMET_data.zarr")
        if os.path.exists(gridmet_path):
            ds = xr.open_zarr(gridmet_path)
            if 'day' in ds.dims:
                ds = ds.rename({'day': 'time'})
            da = ds['prcp'].sel(time=slice(start, end)).sum(dim='time', skipna=True).compute()
            gridmet_regrid = regrid_to_reference(da, ref_lon, ref_lat, mask_2d)
            bias_grids_daily[label]['GridMET'] = gridmet_regrid - prism_regrid
            bias_grids_hourly[label]['GridMET'] = gridmet_regrid - prism_regrid
            ds.close()

    # Compute differences and spatial means
    print("Computing differences...\n")
    bias_differences = {}
    spatial_means_list = []
    for event in AR_EVENTS:
        label = event["label"]
        bias_differences[label] = {}
        start = event["start"]
        end = event["end"]
        print(f"  {label}:")
        for prod in BIAS_PRODUCTS:
            if prod in bias_grids_daily[label] and prod in bias_grids_hourly[label]:
                diff = bias_grids_hourly[label][prod] - bias_grids_daily[label][prod]
                bias_differences[label][prod] = diff
                all_differences.append(diff[~np.isnan(diff)])

                # Compute and print mean bias difference
                valid_mask = ~np.isnan(diff)
                if valid_mask.any():
                    mean_diff = np.nanmean(diff[valid_mask])
                    print(f"    {prod}: mean difference = {mean_diff:.3f} mm")

            # Compute spatial means for daily and hourly
            if prod in ['CONUS404', 'UCLA']:
                daily_grid = bias_grids_daily[label].get(prod)
                hourly_grid = bias_grids_hourly[label].get(prod)
                daily_mean = np.nanmean(daily_grid) if daily_grid is not None and not np.all(np.isnan(daily_grid)) else np.nan
                hourly_mean = np.nanmean(hourly_grid) if hourly_grid is not None and not np.all(np.isnan(hourly_grid)) else np.nan
                spatial_means_list.append({
                    'event': label,
                    'start_date': start,
                    'end_date': end,
                    'product': prod,
                    'daily_bias_mean_mm': daily_mean,
                    'hourly_bias_mean_mm': hourly_mean,
                    'difference_mean_mm': hourly_mean - daily_mean if not np.isnan(hourly_mean) and not np.isnan(daily_mean) else np.nan
                })

    # Determine colorbar limits
    all_diffs_array = np.concatenate(all_differences) if all_differences else np.array([0])
    vmax_diff = float(np.ceil(np.quantile(np.abs(all_diffs_array), 0.95)))
    vmin_diff = -vmax_diff

    print(f"Difference color range: {vmin_diff}–{vmax_diff} mm\n")

    # Apply font settings
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
    })

    # Plot grid
    im = None
    for row_idx, event in enumerate(AR_EVENTS):
        label = event["label"]
        for col_idx, prod in enumerate(BIAS_PRODUCTS):
            ax = axes[row_idx, col_idx]

            if prod in bias_differences[label]:
                diff_grid = bias_differences[label][prod]

                if diff_grid is not None and not np.all(np.isnan(diff_grid)):
                    im = ax.pcolormesh(
                        ref_lon, ref_lat, diff_grid,
                        transform=ccrs.PlateCarree(),
                        cmap="RdBu_r", vmin=vmin_diff, vmax=vmax_diff,
                        shading='auto'
                    )

            ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, edgecolor='#444444')
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='#000000')
            ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle='--', linewidth=0.4, edgecolor='#666666')

            # Add subbasin and boundary overlays
            if subbasin_gdf is not None:
                subbasin_gdf.plot(ax=ax, facecolor="none", edgecolor="#888888",
                                  lw=0.5, linestyle=":", transform=ccrs.PlateCarree())
            boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#000000",
                              lw=1.3, transform=ccrs.PlateCarree())

            # Row label
            if col_idx == 0:
                ax.text(-0.25, 0.5, event['row_label'], transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='center', ha='right')

        # Column titles
        if row_idx == 0:
            for col_idx, prod in enumerate(BIAS_PRODUCTS):
                axes[0, col_idx].set_title(f"{prod} Difference", fontsize=16, fontweight='bold', pad=12)

    # Colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.30, 0.05, 0.40, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                            label="Bias Difference (Hourly - Daily) (mm)")
        cbar.ax.tick_params(labelsize=13)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.05, hspace=0.08)

    plt.suptitle(
        "Bias Difference: Hourly vs Daily Approach\n"
        "Multi-Product Comparison (Bias Difference relative to PRISM)\n"
        "(Positive = Hourly has higher bias, Negative = Daily has higher bias)",
        fontsize=22, fontweight='bold', y=0.97
    )

    out_png = os.path.join(OUT_DIR, "bias_difference_hourly_vs_daily_grid.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

    # Save spatial means to CSV
    if spatial_means_list:
        df_means = pd.DataFrame(spatial_means_list)
        csv_path = os.path.join(OUT_DIR, "hourly_vs_daily_spatial_means_conus_ucla.csv")
        df_means.to_csv(csv_path, index=False)
        print(f"Saved spatial means: {csv_path}")


if __name__ == "__main__":
    main()
