"""Generate spatial maps of AR event precipitation across weather products (8-day windows).

Extracts 8-day cumulative precipitation (T-2 to T+5) for 6 selected atmospheric river
(AR) events from multiple weather datasets (PRISM, Daymet, PNNL, CONUS404, UCLA,
GridMET). Regrids all products to the PRISM 4km reference grid and generates a
6 × 6 grid visualization where each row represents an AR event and each column shows
raw precipitation values for one product (baseline and alternatives).

Outputs one high-resolution PNG figure to spatial_plots/plots/
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from scipy.interpolate import griddata
import subprocess
import zipfile
import tempfile
import rioxarray

warnings.filterwarnings('ignore')

# --- Configuration & Paths ---
BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
DATA_DIR = "/data0/hernanqd/plots_code/skagit_basin_de/data"
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PRISM_ROOT = os.path.join(VAULT_DIR, "prism_new_hq")
DAYMET_ROOT = os.path.join(VAULT_DIR, "daymet_new_hq")
BOUNDARY_PATH = os.path.join(BASE_DIR, "data/GIS/SkagitBoundary.json")
SUBBASIN_PATH  = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
OUT_DIR = os.path.join("/data0/hernanqd/plots_code/skagit_basin_de/spatial_plots/daily_versions/plots")
DAILY_MEANS_DIR = os.path.join("/data0/hernanqd/plots_code/skagit_basin_de/spatial_plots/daily_versions/daily_means")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DAILY_MEANS_DIR, exist_ok=True)

# --- AR Event windows (exact dates from cumulative precipitation plot) ---

# AR_EVENTS = [
#     {"label": "November_1990_AR5",   "start": "1990-11-23", "end": "1990-11-25", "row_label": "Nov 23–25, 1990\n(AR5)"},
#     {"label": "December_2010_AR3",  "start": "2010-12-11", "end": "2010-12-13", "row_label": "Dec 11–13, 2010\n(AR3)"},
#     {"label": "January_1984_AR4",  "start": "1984-01-03", "end": "1984-01-5", "row_label": "Jan 3–5, 1984\n(AR4)"},
#     {"label": "November_2017_AR4",  "start": "2017-11-21", "end": "2017-11-23", "row_label": "Nov 21–23, 2017\n(AR4)"},
#     {"label": "November_1999_AR3",  "start": "1999-11-11", "end": "1999-11-13", "row_label": "Nov 11–13, 1999\n(AR3)"},
#     {"label": "January_2011_AR2",  "start": "2011-01-15", "end": "2011-01-17", "row_label": "Jan 15–17, 2011\n(AR2)"}
# ]

AR_EVENTS = [
    {"label": "November_1990_AR5",   "start": "1990-11-22", "end": "1990-11-29", "row_label": "Nov 22–29, 1990\n(AR5)"},
    # {"label": "November_1995_AR4",  "start": "1995-11-27", "end": "1995-12-04", "row_label": "Nov 27–Dec 4, 1995\n(AR4)"},
    # {"label": "November_1990_AR4",  "start": "1990-11-08", "end": "1990-11-15", "row_label": "Nov 8–15, 1990\n(AR4)"},
    # {"label": "November_2006_AR5",  "start": "2006-11-05", "end": "2006-11-12", "row_label": "Nov 5–12, 2006\n(AR5)"},
    # {"label": "October_2003_AR5",  "start": "2003-10-19", "end": "2003-10-26", "row_label": "Oct 19–26, 2003\n(AR5)"},
    # {"label": "November_2021_AR4",  "start": "2021-11-13", "end": "2021-11-20", "row_label": "Nov 13–20, 2021\n(AR4)"}
]

PRODUCTS = ['PRISM', 'Daymet', 'PNNL', 'CONUS404', 'UCLA', 'GridMET'] #'ORNL (Daymet)', 'HRRR'

# Crop bounding box: Skagit domain
BB = (-122.5, -120.5, 47.8, 49.5)  # lon_min, lon_max, lat_min, lat_max
MAP_EXTENT = [-122.35, -120.65, 47.90, 49.35]

# --- Helper functions ---
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
    """Load a single day of PRISM precipitation from the new TIFF/zip source.
    Returns a DataArray with 1D 'lon'/'lat' coords (renamed from x/y), or None if missing."""
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
        print(f"  Error loading PRISM (new format) for {date_str}: {e}")

    return None


def load_daymet_from_netcdf(year, daymet_root):
    """Load Daymet data from the new netCDF source for a specific year."""
    daymet_nc_path = os.path.join(daymet_root, f"prcp_{year}_subset.nc")

    if not os.path.exists(daymet_nc_path):
        return None

    try:
        return xr.open_dataset(daymet_nc_path)
    except Exception as e:
        print(f"  Error loading Daymet netCDF for {year}: {e}")
        return None

# --- Load static coordinates once ---
print("Loading static grid coordinates...")

# Load PNNL 6km coordinates (for PNNL product, not reference)
ds_pnnl_static = xr.open_dataset(os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc"))
pnnl_lon_full = ds_pnnl_static.XLONG_M.values[0]
pnnl_lat_full = ds_pnnl_static.XLAT_M.values[0]
ds_pnnl_static.close()

# Load PRISM reference grid (new TIFF/zip source)
prism_sample_date = pd.Timestamp("2020-01-01")
da_prism_sample = load_prism_day(prism_sample_date, PRISM_ROOT)
if da_prism_sample is not None:
    prism_lon_full = da_prism_sample['lon'].values
    prism_lat_full = da_prism_sample['lat'].values
else:
    raise FileNotFoundError(f"Could not find PRISM data for reference date {prism_sample_date.date()}")

ds_ucla_static = xr.open_dataset(os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc"))
ucla_lon = ds_ucla_static.lon2d.values
ucla_lat = ds_ucla_static.lat2d.values
ds_ucla_static.close()

# HRRR coordinates
hrrr_lon = None
hrrr_lat = None
for y in range(2014, 2022):
    for m in range(1, 13):
        p = os.path.join(VAULT_DIR, "weather_data", f"{y}-{m:02d}_HRRR_data.zarr")
        if os.path.exists(p):
            try:
                ds_h = xr.open_zarr(p, consolidated=False)
                hrrr_lon = ds_h.longitude.values
                hrrr_lat = ds_h.latitude.values
                ds_h.close()
                break
            except Exception:
                pass
    if hrrr_lon is not None:
        break

# # Crop reference PNNL 6km grid (commented out; using PRISM 4km instead)
# row_min_p, row_max_p, col_min_p, col_max_p = crop_rows_cols(pnnl_lon_full, pnnl_lat_full)
# ref_lon = pnnl_lon_full[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
# ref_lat = pnnl_lat_full[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
# print(f"Reference 6km grid shape: {ref_lon.shape}")

# Crop reference PRISM grid
if prism_lon_full.ndim == 1 and prism_lat_full.ndim == 1:
    prism_lon_2d, prism_lat_2d = np.meshgrid(prism_lon_full, prism_lat_full)
else:
    prism_lon_2d, prism_lat_2d = prism_lon_full, prism_lat_full
row_min_p, row_max_p, col_min_p, col_max_p = crop_rows_cols(prism_lon_2d, prism_lat_2d)
ref_lon = prism_lon_2d[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
ref_lat = prism_lat_2d[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
print(f"Reference PRISM grid shape: {ref_lon.shape}")

# Crop UCLA & HRRR coordinate arrays
row_min_u, row_max_u, col_min_u, col_max_u = crop_rows_cols(ucla_lon, ucla_lat)
ucla_lon_c = ucla_lon[row_min_u:row_max_u+1, col_min_u:col_max_u+1]
ucla_lat_c = ucla_lat[row_min_u:row_max_u+1, col_min_u:col_max_u+1]

if hrrr_lon is not None:
    row_min_h, row_max_h, col_min_h, col_max_h = crop_rows_cols(hrrr_lon, hrrr_lat)
    hrrr_lon_c = hrrr_lon[row_min_h:row_max_h+1, col_min_h:col_max_h+1]
    hrrr_lat_c = hrrr_lat[row_min_h:row_max_h+1, col_min_h:col_max_h+1]
else:
    hrrr_lon_c = hrrr_lat_c = None

# Load GIS boundaries
boundary_gdf = gpd.read_file(BOUNDARY_PATH)
subbasin_gdf = gpd.read_file(SUBBASIN_PATH) if os.path.exists(SUBBASIN_PATH) else None

# Precompute watershed mask on the reference grid
if hasattr(boundary_gdf.to_crs("EPSG:4326"), 'union_all'):
    poly = boundary_gdf.to_crs("EPSG:4326").union_all()
else:
    poly = boundary_gdf.to_crs("EPSG:4326").unary_union

df_pts = pd.DataFrame({'lon': ref_lon.flatten(), 'lat': ref_lat.flatten()})
gdf_pts = gpd.GeoDataFrame(df_pts, geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat), crs="EPSG:4326")
inside = gdf_pts.intersects(poly).values
mask_2d = inside.reshape(ref_lon.shape)


def regrid_to_reference(da_src, mask_2d=None):
    """Regrid any product to the reference PRISM grid via griddata."""
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


def load_event_grids(event):
    """Load daily-mean precipitation from each product for the event window.
    Also returns daily spatial means for CONUS404 and UCLA."""
    start = event["start"]
    end   = event["end"]
    year  = int(start[:4])
    print(f"\n  Loading data for {event['label']} ({start} → {end})...")

    grids = {p: None for p in PRODUCTS}
    date_range = pd.date_range(start, end, freq='D')
    daily_means = {'date': date_range.strftime('%Y-%m-%d').tolist(), 'CONUS404': [], 'UCLA': []}

    # 1. PRISM (new TIFF/zip source) - load each day and sum over the event window
    try:
        date_range = pd.date_range(start, end, freq='D')
        daily_das = []
        for d in date_range:
            da_day = load_prism_day(d, PRISM_ROOT)
            if da_day is not None:
                da_day = da_day.where(da_day >= 0)  # mask nodata (e.g. -9999)
                da_day = da_day.expand_dims(time=[d])
                daily_das.append(da_day)
        if daily_das:
            da = xr.concat(daily_das, dim='time').sum(dim='time', skipna=True)
            grids['PRISM'] = da
            print(f"    PRISM OK ({len(daily_das)}/{len(date_range)} days)")
        else:
            print("    [WARN] PRISM: no daily files found for this window")
    except Exception as e:
        print(f"    [WARN] PRISM: {e}")

    # 2. Daymet (new netCDF source) & ORNL (same source)
    try:
        ds = load_daymet_from_netcdf(year, DAYMET_ROOT)
        if ds is not None:
            ds['time'] = pd.to_datetime(ds.time.values).normalize()
            da = ds['prcp'].sel(time=slice(start, end)).sum(dim='time').compute()
            grids['Daymet']        = da
            grids['ORNL (Daymet)'] = da
            ds.close()
            print("    Daymet/ORNL OK")
        else:
            print(f"    [WARN] Daymet: file not found for year {year}")
    except Exception as e:
        print(f"    [WARN] Daymet/ORNL: {e}")

    # 3. PNNL WRF
    try:
        pnnl_file = os.path.join(VAULT_DIR, "PNNL/historical", str(year),
                                 f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        if os.path.exists(pnnl_file):
            ds = xr.open_dataset(pnnl_file, chunks={'time': 720})
            da_daily = ds['PREC_ACC_NC'].resample(time='1D').sum()
            # Crop
            mask_c = bb_mask(pnnl_lon_full, pnnl_lat_full)
            rows, cols = np.where(mask_c)
            rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
            da_daily = da_daily.isel(x=slice(rm, rx+1), y=slice(cm, cx+1))
            lat_c = pnnl_lat_full[rm:rx+1, cm:cx+1]
            lon_c = pnnl_lon_full[rm:rx+1, cm:cx+1]
            da = da_daily.sel(time=slice(start, end)).sum(dim='time').compute()
            da = da.assign_coords(lat=(('x', 'y'), lat_c), lon=(('x', 'y'), lon_c))
            grids['PNNL'] = da
            ds.close()
            print("    PNNL OK")
    except Exception as e:
        print(f"    [WARN] PNNL: {e}")

    # 4. CONUS404
    try:
        conus_path = os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
        ds = xr.open_zarr(conus_path)

        # Extract daily spatial means (regridded & masked) while loading
        for d in date_range:
            try:
                da_day = ds['precip_daily'].sel(time=d).compute()
                regridded = regrid_to_reference(da_day, mask_2d)
                if not np.all(np.isnan(regridded)):
                    daily_means['CONUS404'].append(float(np.nanmean(regridded)))
                else:
                    daily_means['CONUS404'].append(np.nan)
            except Exception as e_conus:
                print(f"      [DEBUG CONUS] Failed for {d.strftime('%Y-%m-%d')}: {e_conus}")
                daily_means['CONUS404'].append(np.nan)

        # For cumulative, crop to bbox as before
        mask_c = bb_mask(ds.lon.values, ds.lat.values)
        rows, cols = np.where(mask_c)
        if len(rows):
            rm, rx, cm, cx = rows.min(), rows.max(), cols.min(), cols.max()
            ds = ds.isel(y=slice(rm, rx+1), x=slice(cm, cx+1))

        da = ds['precip_daily'].sel(time=slice(start, end)).sum(dim='time').compute()
        grids['CONUS404'] = da
        ds.close()
        print("    CONUS404 OK")
    except Exception as e:
        print(f"    [WARN] CONUS404: {e}")
        daily_means['CONUS404'] = [np.nan] * len(date_range)

    # 5. UCLA ERA5 d02
    try:
        da_parts = []
        for yr_off in [year-1, year]:
            p = os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec",
                             f"prec.daily.era5.d02.{yr_off}.nc")
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
            # Reassign lat2d/lon2d coordinates that may have been lost during concatenation
            da_year = da_year.assign_coords(lat2d=(('lat2d', 'lon2d'), ucla_lat),
                                            lon2d=(('lat2d', 'lon2d'), ucla_lon))

            # Extract daily spatial means (regridded & masked) while loading
            for d in date_range:
                try:
                    da_day = da_year.sel(day=d.strftime('%Y-%m-%d')).compute()
                    regridded = regrid_to_reference(da_day, mask_2d)
                    if not np.all(np.isnan(regridded)):
                        daily_means['UCLA'].append(float(np.nanmean(regridded)))
                    else:
                        daily_means['UCLA'].append(np.nan)
                except Exception as e_ucla:
                    print(f"      [DEBUG UCLA] Failed for {d.strftime('%Y-%m-%d')}: {e_ucla}")
                    daily_means['UCLA'].append(np.nan)

            # For cumulative, crop and use cropped version
            da_year = da_year.isel(lat2d=slice(row_min_u, row_max_u+1),
                                   lon2d=slice(col_min_u, col_max_u+1))
            da = da_year.sum(dim='day')
            da = da.assign_coords(lat=(('lat2d', 'lon2d'), ucla_lat_c),
                                  lon=(('lat2d', 'lon2d'), ucla_lon_c))
            grids['UCLA'] = da
            print("    UCLA OK")
        else:
            daily_means['UCLA'] = [np.nan] * len(date_range)
    except Exception as e:
        print(f"    [WARN] UCLA: {e}")
        daily_means['UCLA'] = [np.nan] * len(date_range)

    # 6. GridMET
    try:
        gridmet_path = os.path.join(VAULT_DIR, "gridmet", f"{year}_daily_4km_gridMET_data.zarr")
        if os.path.exists(gridmet_path):
            ds = xr.open_zarr(gridmet_path)
            if 'day' in ds.dims:
                ds = ds.rename({'day': 'time'})
            da = ds['prcp'].sel(time=slice(start, end)).sum(dim='time').compute()
            grids['GridMET'] = da
            ds.close()
            print("    GridMET OK")
    except Exception as e:
        print(f"    [WARN] GridMET: {e}")

    # 7. HRRR  (available 2014–)
    if year >= 2014 and hrrr_lon_c is not None:
        try:
            start_dt = pd.Timestamp(start)
            end_dt   = pd.Timestamp(end)
            months_needed = set()
            cur = start_dt
            while cur <= end_dt:
                months_needed.add((cur.year, cur.month))
                cur += pd.DateOffset(months=1)
            # Also include the next month in case event spans a boundary
            months_needed.add(((end_dt + pd.DateOffset(months=1)).year,
                                (end_dt + pd.DateOffset(months=1)).month))

            da_months = []
            for (yr_m, mo_m) in sorted(months_needed):
                for suffix in ["", "_fixed_time"]:
                    p = os.path.join(VAULT_DIR, "weather_data",
                                     f"{yr_m}-{mo_m:02d}_HRRR_data{suffix}.zarr")
                    if os.path.exists(p):
                        ds_h = xr.open_zarr(p, consolidated=False)
                        ds_h = ds_h.isel(y=slice(row_min_h, row_max_h+1),
                                         x=slice(col_min_h, col_max_h+1))
                        if "forecast_hour" in ds_h.coords:
                            ds_h = ds_h.where(ds_h.forecast_hour == 1, drop=True)
                        elif "step" in ds_h.coords:
                            if np.issubdtype(ds_h.step.dtype, np.timedelta64):
                                ds_h = ds_h.where(ds_h.step == np.timedelta64(1, "h"), drop=True)
                            else:
                                ds_h = ds_h.where(ds_h.step == 1, drop=True)
                        ds_h = ds_h.sortby('time').drop_duplicates('time')
                        v_h = 'APCP_sfc' if 'APCP_sfc' in ds_h.data_vars else 'tp'
                        da_d = ds_h[v_h].resample(time='1D').sum().compute()
                        da_d = da_d.drop_vars(['latitude', 'longitude'], errors='ignore')
                        da_months.append(da_d)
                        ds_h.close()
                        break   # prefer non-fixed unless only fixed exists

            if da_months:
                da_all = xr.concat(da_months, dim='time').sortby('time').drop_duplicates('time')
                da = da_all.sel(time=slice(start, end)).sum(dim='time')
                da = da.assign_coords(lat=(('y', 'x'), hrrr_lat_c),
                                      lon=(('y', 'x'), hrrr_lon_c))
                grids['HRRR'] = da
                print("    HRRR OK")
        except Exception as e:
            print(f"    [WARN] HRRR: {e}")
    else:
        print("    HRRR skipped (pre-2014 event or no grid)")

    return grids, daily_means


def main():
    # Load and regrid all events
    event_regridded_grids = {}
    event_snotel_data = {}

    for event in AR_EVENTS:
        start = event["start"]
        end = event["end"]
        print(f"\nProcessing: {event['label']}")
        grids, daily_means = load_event_grids(event)

        # Save daily spatial means to CSV
        df_means = pd.DataFrame(daily_means)
        csv_path = os.path.join(DAILY_MEANS_DIR, f"{event['label']}_daily_spatial_means.csv")
        df_means.to_csv(csv_path, index=False)
        print(f"  Saved daily spatial means to: {csv_path}")

        # Load SNOTEL data
        zarr_path = os.path.join(DATA_DIR, "weather_data", f"{start}_{end}_SNOTEL_daily_data.zarr")
        if not os.path.exists(zarr_path):
            print(f"  Downloading SNOTEL daily data for {start} to {end}...")
            import sys
            try:
                subprocess.run([
                    sys.executable,
                    os.path.join(BASE_DIR, "scripts/snotel_downloader_hernan.py"),
                    "--startDate", start,
                    "--endDate", end,
                    "--geojson", os.path.join(BASE_DIR, "data/GIS/SkagitBoundary.json"),
                    "--frequency", "daily",
                    "--outputDir", os.path.join(BASE_DIR, "data/")
                ], check=True)
            except subprocess.CalledProcessError:
                print(f"  [WARN] SNOTEL download failed (metloom module may not be available)")

        # Load SNOTEL data and filter to event period (start to end)
        try:
            ds_snotel = xr.open_zarr(zarr_path)
            snotel_lats = ds_snotel.lat.values
            snotel_lons = ds_snotel.lon.values
            snotel_cumulative_inches = ds_snotel.PRECIPITATION.sel(time=slice(start, end)).sum(dim='time').values
            snotel_cumulative_mm = snotel_cumulative_inches * 25.4

            event_snotel_data[event['label']] = {
                'lons': snotel_lons,
                'lats': snotel_lats,
                'vals': snotel_cumulative_mm
            }
            print(f"  SNOTEL loaded: {len(snotel_lons)} stations.")
            ds_snotel.close()
        except Exception as e:
            print(f"  [WARN] SNOTEL processing failed: {e}")
            event_snotel_data[event['label']] = None
        
        # Regrid each product
        event_regridded_grids[event['label']] = {}
        for prod in PRODUCTS:
            print(f"  Regridding {prod} for {event['label']}...")
            event_regridded_grids[event['label']][prod] = regrid_to_reference(grids[prod], mask_2d)

    # Dynamic vmax: 99.5th percentile across all non-NaN values in all products/events
    print("\nCalculating dynamic colorbar limits...")
    all_arrays = []
    for elabel in event_regridded_grids:
        for prod in PRODUCTS:
            g = event_regridded_grids[elabel][prod]
            if g is not None:
                v = g[~np.isnan(g)]
                all_arrays.append(v)
    all_vals = np.concatenate(all_arrays) if all_arrays else np.array([])
    if len(all_vals):
        vmax = max(float(np.ceil(np.quantile(all_vals, 0.995))), 1.0)
    else:
        vmax = 60.0
    vmin = 0.0
    print(f"Global color range: {vmin}–{vmax} mm/day")

    # Plot grid: 5 rows × 8 columns
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
    })

    fig, axes = plt.subplots(
        len(AR_EVENTS), len(PRODUCTS),
        figsize=(17.5, 17.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor='#ffffff'
    )

    # Ensure axes is always 2D
    if len(AR_EVENTS) == 1:
        axes = axes.reshape(1, -1)

    im = None
    for row_idx, event in enumerate(AR_EVENTS):
        elabel = event['label']
        for col_idx, prod in enumerate(PRODUCTS):
            ax = axes[row_idx, col_idx]
            grid = event_regridded_grids[elabel][prod]

            if grid is not None and not np.all(np.isnan(grid)):
                im = ax.pcolormesh(
                    ref_lon, ref_lat, grid,
                    transform=ccrs.PlateCarree(),
                    cmap="Greens", vmin=vmin, vmax=vmax,
                    shading='auto'
                )

                # Overlay SNOTEL precipitation points
                snotel = event_snotel_data[elabel]
                if snotel is not None:
                    ax.scatter(
                        snotel['lons'], snotel['lats'],
                        c=snotel['vals'], cmap="Greens",
                        vmin=vmin, vmax=vmax,
                        edgecolors="black", linewidths=1.2,
                        s=60, transform=ccrs.PlateCarree(),
                        zorder=10
                    )
            else:
                # Grey placeholder if product missing
                ax.set_facecolor('#dddddd')
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='#555555')

            ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, edgecolor='#444444')
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='#000000')
            ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle='--', linewidth=0.4, edgecolor='#666666')

            if subbasin_gdf is not None:
                subbasin_gdf.plot(ax=ax, facecolor="none", edgecolor="#888888",
                                  lw=0.5, linestyle=":", transform=ccrs.PlateCarree())
            boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#000000",
                              lw=1.3, transform=ccrs.PlateCarree())

            # Row label on the first column
            if col_idx == 0:
                ax.text(-0.25, 0.5, event['row_label'], transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='center', ha='right')

        # Column titles on the first row
        if row_idx == 0:
            for col_idx, prod in enumerate(PRODUCTS):
                label = "ORNL (Daymet)" if prod == "ORNL" else prod
                axes[0, col_idx].set_title(label, fontsize=16, fontweight='bold', pad=12)

    # Colorbar at the bottom of the grid
    if im is not None:
        cbar_ax = fig.add_axes([0.30, 0.05, 0.40, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                            label="Cumulative Precipitation (mm)")
        cbar.ax.tick_params(labelsize=13)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.05, hspace=0.08)

    # # Overall title
    # plt.suptitle(
    #     "Spatial Distribution of Precipitation during Atmospheric River Events\n"
    #     "Multi-Product Comparison (Cumulative Precipitation)",
    #     fontsize=22, fontweight='bold', y=0.96
    # )

    plt.suptitle(
        "Spatial Distribution of Precipitation during Atmospheric River Events\n"
        "Multi-Product Comparison (Cumulative Precipitation)\n"
        "(AR Event Day: 2 days after period start)",
        fontsize=22, fontweight='bold', y=0.97
    )

    out_png = os.path.join(OUT_DIR, "ar_events_cumulative_spatial_grid_8_days.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined comparison figure to: {out_png}")


if __name__ == "__main__":
    main()
