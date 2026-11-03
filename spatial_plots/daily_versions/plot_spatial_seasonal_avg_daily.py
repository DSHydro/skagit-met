"""Generate spatial maps of seasonal average precipitation and biases for Skagit Basin.

Extracts seasonal average precipitation (Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec) from
multiple weather datasets (PRISM, Daymet, PNNL, CONUS404, UCLA, GridMET) for years
1981–2019. Regrids all products to a common PRISM 4km reference grid, computes
multi-year averages, and calculates biases relative to PRISM baseline. Generates
two publication-quality maps: one showing raw seasonal average values for each
product, and one showing product-minus-PRISM biases.

Outputs two high-resolution PNG figures to spatial_plots/plots/plot_spatial_seasonal_avg_prism/
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
from joblib import Parallel, delayed
import rioxarray
import tempfile
import zipfile

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration & Paths ---
BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
BOUNDARY_PATH = os.path.join(BASE_DIR, "data/GIS/SkagitBoundary.json")
SUBBASIN_PATH = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
OUT_DIR = os.path.join("/data0/hernanqd/plots_code/skagit-met/spatial_plots/plots/plot_spatial_seasonal_avg_prism")

# Output image paths
OUT_PRODUCTS_IMG = os.path.join(OUT_DIR, "spatial_seasonal_products.png")
OUT_BIASES_IMG = os.path.join(OUT_DIR, "spatial_seasonal_biases.png")

# Seasons definition (Months)
SEASONS = {
    "Jan-Mar": [1, 2, 3],
    "Apr-Jun": [4, 5, 6],
    "Jul-Sep": [7, 8, 9],
    "Oct-Dec": [10, 11, 12]
}

PRISM_ROOT = os.path.join(VAULT_DIR, "prism_new_hq")
DAYMET_ROOT = os.path.join(VAULT_DIR, "daymet_new_hq")

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
    """Load a single day of PRISM precipitation from TIFF/zip source."""
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
        print(f"  Error loading PRISM for {date_str}: {e}")

    return None

def load_daymet_from_netcdf(year, daymet_root):
    """Load Daymet data from netCDF source for a specific year."""
    daymet_nc_path = os.path.join(daymet_root, f"prcp_{year}_subset.nc")

    if not os.path.exists(daymet_nc_path):
        return None

    try:
        return xr.open_dataset(daymet_nc_path)
    except Exception as e:
        print(f"  Error loading Daymet netCDF for {year}: {e}")
        return None

# Load static coordinates globally to avoid repeated opening in workers
print("Loading static grid coordinates...")

# Load PRISM reference grid (new TIFF/zip source)
prism_sample_date = pd.Timestamp("2020-01-01")
da_prism_sample = load_prism_day(prism_sample_date, PRISM_ROOT)
if da_prism_sample is not None:
    prism_lon_full = da_prism_sample['lon'].values
    prism_lat_full = da_prism_sample['lat'].values
else:
    raise FileNotFoundError(f"Could not find PRISM data for reference date {prism_sample_date.date()}")

# PNNL (still needed for loading PNNL data)
ds_pnnl_static = xr.open_dataset(os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc"))
pnnl_lon = ds_pnnl_static.XLONG_M.values[0]
pnnl_lat = ds_pnnl_static.XLAT_M.values[0]
ds_pnnl_static.close()

# UCLA
ds_ucla_static = xr.open_dataset(os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc"))
ucla_lon = ds_ucla_static.lon2d.values
ucla_lat = ds_ucla_static.lat2d.values
ds_ucla_static.close()

# # HRRR (commented out - not plotting)
# hrrr_lon = None
# hrrr_lat = None
# for y in range(2014, 2022):
#     for m in range(1, 13):
#         p = os.path.join(VAULT_DIR, "weather_data", f"{y}-{m:02d}_HRRR_data.zarr")
#         if os.path.exists(p):
#             try:
#                 ds_h = xr.open_zarr(p, consolidated=False)
#                 hrrr_lon = ds_h.longitude.values
#                 hrrr_lat = ds_h.latitude.values
#                 ds_h.close()
#                 break
#             except:
#                 pass
#     if hrrr_lon is not None:
#         break
#
# # Pre-find HRRR crop range globally
# hrrr_row_min = hrrr_row_max = hrrr_col_min = hrrr_col_max = None
# if hrrr_lon is not None and hrrr_lat is not None:
#     mask_crop_hrrr = (hrrr_lon >= -122.5) & (hrrr_lon <= -120.5) & (hrrr_lat >= 47.8) & (hrrr_lat <= 49.5)
#     rows_h, cols_h = np.where(mask_crop_hrrr)
#     if len(rows_h) > 0:
#         hrrr_row_min, hrrr_row_max = rows_h.min(), rows_h.max()
#         hrrr_col_min, hrrr_col_max = cols_h.min(), cols_h.max()

# Set HRRR variables to None since we're not using it
hrrr_lon = None
hrrr_lat = None
hrrr_row_min = hrrr_row_max = hrrr_col_min = hrrr_col_max = None

def process_year(year):
    """Processes all products for a single year and extracts seasonal daily means in their native grids."""
    print(f"Processing Year: {year}...")
    year_results = {}
    
    # Initialize seasonal maps
    for prod in ['PRISM', 'Daymet', 'PNNL', 'CONUS404', 'UCLA', 'GridMET']:
        year_results[prod] = {s: None for s in SEASONS}
        
    # 1. PRISM (new TIFF/zip source)
    try:
        date_range = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')
        daily_das = []
        for d in date_range:
            da_day = load_prism_day(d, PRISM_ROOT)
            if da_day is not None:
                da_day = da_day.where(da_day >= 0)
                da_day = da_day.expand_dims(time=[d])
                daily_das.append(da_day)
        if daily_das:
            da = xr.concat(daily_das, dim='time')
            times = pd.to_datetime(da.time.values)
            for s_name, months in SEASONS.items():
                mask = np.isin(times.month, months)
                mean_s = da.isel(time=mask).mean(dim='time').compute()
                year_results['PRISM'][s_name] = mean_s
    except Exception as e:
        print(f"  [ERROR] PRISM {year}: {e}")

    # 2. Daymet (new netCDF source)
    try:
        ds = load_daymet_from_netcdf(year, DAYMET_ROOT)
        if ds is not None:
            ds['time'] = pd.to_datetime(ds.time.values).normalize()
            times = pd.to_datetime(ds.time.values)
            var = 'prcp' if 'prcp' in ds.data_vars else 'ppt'
            for s_name, months in SEASONS.items():
                mask = np.isin(times.month, months)
                mean_s = ds[var].isel(time=mask).mean(dim='time').compute()
                year_results['Daymet'][s_name] = mean_s
            ds.close()
    except Exception as e:
        print(f"  [ERROR] Daymet {year}: {e}")
        
    # 3. PNNL WRF
    try:
        pnnl_file = os.path.join(VAULT_DIR, "PNNL/historical", str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        if os.path.exists(pnnl_file):
            ds = xr.open_dataset(pnnl_file, chunks={'time': 720})
            da_daily = ds['PREC_ACC_NC'].resample(time='1D').sum()
            
            # Crop PNNL WRF to Skagit bounding box
            mask_crop = (pnnl_lon >= -122.5) & (pnnl_lon <= -120.5) & (pnnl_lat >= 47.8) & (pnnl_lat <= 49.5)
            rows, cols = np.where(mask_crop)
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            da_daily = da_daily.isel(y=slice(row_min, row_max+1), x=slice(col_min, col_max+1))
            pnnl_lat_cropped = pnnl_lat[row_min:row_max+1, col_min:col_max+1]
            pnnl_lon_cropped = pnnl_lon[row_min:row_max+1, col_min:col_max+1]
            
            times = pd.to_datetime(da_daily.time.values)
            for s_name, months in SEASONS.items():
                mask = np.isin(times.month, months)
                mean_s = da_daily.isel(time=mask).mean(dim='time').compute()
                # Attach static coords
                mean_s = mean_s.assign_coords(lat=(('y', 'x'), pnnl_lat_cropped), lon=(('y', 'x'), pnnl_lon_cropped))
                year_results['PNNL'][s_name] = mean_s
            ds.close()
    except Exception as e:
        print(f"  [ERROR] PNNL {year}: {e}")
        
    # 4. CONUS404
    try:
        conus_path = os.path.join(BASE_DIR, "data/weather_data/conus404_skagit_precip_daily_full.zarr")
        ds = xr.open_zarr(conus_path)
        
        # Crop CONUS404 to Skagit bounding box
        mask_crop = (ds.lon >= -122.5) & (ds.lon <= -120.5) & (ds.lat >= 47.8) & (ds.lat <= 49.5)
        rows, cols = np.where(mask_crop.values)
        if len(rows) > 0:
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            ds = ds.isel(y=slice(row_min, row_max+1), x=slice(col_min, col_max+1))
            
        ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        times = pd.to_datetime(ds_year.time.values)
        for s_name, months in SEASONS.items():
            mask = np.isin(times.month, months)
            mean_s = ds_year['precip_daily'].isel(time=mask).mean(dim='time').compute()
            year_results['CONUS404'][s_name] = mean_s
        ds.close()
    except Exception as e:
        print(f"  [ERROR] CONUS404 {year}: {e}")
        
    # 5. UCLA
    try:
        u_path_prev = os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year-1}.nc")
        u_path_curr = os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec", f"prec.daily.era5.d02.{year}.nc")
        da_parts = []
        if os.path.exists(u_path_prev):
            ds_prev = xr.open_dataset(u_path_prev)
            u_var = "prec" if "prec" in ds_prev.data_vars else "pr"
            da_part1 = ds_prev[u_var].sel(day=slice(f"{year}-01-01", f"{year}-08-31"))
            da_parts.append(da_part1.compute())
            ds_prev.close()
        if os.path.exists(u_path_curr):
            ds_curr = xr.open_dataset(u_path_curr)
            u_var = "prec" if "prec" in ds_curr.data_vars else "pr"
            da_part2 = ds_curr[u_var].sel(day=slice(f"{year}-09-01", f"{year}-12-31"))
            da_parts.append(da_part2.compute())
            ds_curr.close()
            
        if da_parts:
            da_year = xr.concat(da_parts, dim='day')
            
            # Crop UCLA to Skagit bounding box
            mask_crop = (ucla_lon >= -122.5) & (ucla_lon <= -120.5) & (ucla_lat >= 47.8) & (ucla_lat <= 49.5)
            rows, cols = np.where(mask_crop)
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            da_year = da_year.isel(lat2d=slice(row_min, row_max+1), lon2d=slice(col_min, col_max+1))
            ucla_lon_cropped = ucla_lon[row_min:row_max+1, col_min:col_max+1]
            ucla_lat_cropped = ucla_lat[row_min:row_max+1, col_min:col_max+1]
            
            times = pd.to_datetime(da_year.day.values)
            for s_name, months in SEASONS.items():
                mask = np.isin(times.month, months)
                mean_s = da_year.isel(day=mask).mean(dim='day')
                mean_s = mean_s.assign_coords(lat=(('lat2d', 'lon2d'), ucla_lat_cropped), lon=(('lat2d', 'lon2d'), ucla_lon_cropped))
                year_results['UCLA'][s_name] = mean_s
    except Exception as e:
        print(f"  [ERROR] UCLA {year}: {e}")
        
    # 6. GridMET
    try:
        gridmet_path = os.path.join(VAULT_DIR, "gridmet", f"{year}_daily_4km_gridMET_data.zarr")
        if os.path.exists(gridmet_path):
            ds = xr.open_zarr(gridmet_path)
            if 'day' in ds.dims:
                ds = ds.rename({'day': 'time'})
            times = pd.to_datetime(ds.time.values)
            for s_name, months in SEASONS.items():
                mask = np.isin(times.month, months)
                mean_s = ds['prcp'].isel(time=mask).mean(dim='time').compute()
                year_results['GridMET'][s_name] = mean_s
            ds.close()
    except Exception as e:
        print(f"  [ERROR] GridMET {year}: {e}")
        
    # # 7. HRRR (commented out - not plotting)
    # if year >= 2014:
    #     try:
    #         hrrr_paths = []
    #         for m in range(1, 13):
    #             p = os.path.join(VAULT_DIR, "weather_data", f"{year}-{m:02d}_HRRR_data.zarr")
    #             if os.path.exists(p): hrrr_paths.append(p)
    #             p_fixed = os.path.join(VAULT_DIR, "weather_data", f"{year}-{m:02d}_HRRR_data_fixed_time.zarr")
    #             if os.path.exists(p_fixed): hrrr_paths.append(p_fixed)
    #         next_jan = os.path.join(VAULT_DIR, "weather_data", f"{year+1}-01_HRRR_data.zarr")
    #         if os.path.exists(next_jan): hrrr_paths.append(next_jan)
    #
    #         da_months = []
    #         for p in hrrr_paths:
    #             ds_h = xr.open_zarr(p, consolidated=False)
    #             # Crop to bounding box first (huge speedup)
    #             if hrrr_row_min is not None:
    #                 ds_h_cropped = ds_h.isel(y=slice(hrrr_row_min, hrrr_row_max+1), x=slice(hrrr_col_min, hrrr_col_max+1))
    #             else:
    #                 ds_h_cropped = ds_h
    #             ds_h_cropped = ds_h_cropped.sortby('time').drop_duplicates('time')
    #             var_h = 'APCP_sfc' if 'APCP_sfc' in ds_h_cropped.data_vars else 'tp'
    #             da_h_daily = (ds_h_cropped[var_h] / 6.0).resample(time='1D').sum().compute()
    #             da_h_daily = da_h_daily.drop_vars(['latitude', 'longitude'], errors='ignore')
    #             da_months.append(da_h_daily)
    #             ds_h.close()
    #
    #         if da_months:
    #             da_year = xr.concat(da_months, dim='time').sortby('time').drop_duplicates('time')
    #             da_year = da_year.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    #             if da_year.sizes['time'] > 0:
    #                 hrrr_lon_cropped = hrrr_lon[hrrr_row_min:hrrr_row_max+1, hrrr_col_min:hrrr_col_max+1]
    #                 hrrr_lat_cropped = hrrr_lat[hrrr_row_min:hrrr_row_max+1, hrrr_col_min:hrrr_col_max+1]
    #
    #                 times = pd.to_datetime(da_year.time.values)
    #                 for s_name, months in SEASONS.items():
    #                     mask = np.isin(times.month, months)
    #                     if np.any(mask):
    #                         mean_s = da_year.isel(time=mask).mean(dim='time')
    #                         mean_s = mean_s.assign_coords(lat=(('y', 'x'), hrrr_lat_cropped), lon=(('y', 'x'), hrrr_lon_cropped))
    #                         year_results['HRRR'][s_name] = mean_s
    #     except Exception as e:
    #         print(f"  [ERROR] HRRR {year}: {e}")
            
    return year_results

def align_coords(g, template):
    coords_dict = {}
    for d in g.dims:
        if d in template.coords:
            coords_dict[d] = template.coords[d]
    return g.assign_coords(coords_dict)

def regrid_to_reference(da_src, ref_lat, ref_lon, mask_2d=None):
    """Regrids native 1D/2D grid fields to the 2D cropped PRISM reference grid using scipy griddata.
    This implementation handles duplicate non-monotonic coordinates (e.g. Daymet) and prevents NaN propagation.
    Optionally masks/clips the regridded data using precomputed mask_2d.
    """
    if da_src is None or np.all(np.isnan(da_src.values)):
        return np.full(ref_lat.shape, np.nan)

    # Detect coord names
    lon_name = next((c for c in da_src.coords if 'lon' in c or 'longitude' in c), None)
    lat_name = next((c for c in da_src.coords if 'lat' in c or 'latitude' in c), None)

    if lon_name is None or lat_name is None:
        return np.full(ref_lat.shape, np.nan)

    lon_vals = da_src.coords[lon_name].values
    lat_vals = da_src.coords[lat_name].values
    values = da_src.values

    # Create 2D coordinates for unstructured griddata interpolation
    if lon_vals.ndim == 1 and lat_vals.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    else:
        lon_grid, lat_grid = lon_vals, lat_vals

    points = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))
    vals = values.flatten()

    # Filter out NaNs to optimize griddata speed and prevent NaN propagation
    mask = ~np.isnan(vals)
    points = points[mask]
    vals = vals[mask]

    if len(vals) == 0:
        return np.full(ref_lat.shape, np.nan)

    grid_z = griddata(points, vals, (ref_lon, ref_lat), method='linear')

    if mask_2d is not None:
        grid_z[~mask_2d] = np.nan

    return grid_z

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load reference PRISM 4km grid
    if prism_lon_full.ndim == 1 and prism_lat_full.ndim == 1:
        prism_lon_2d, prism_lat_2d = np.meshgrid(prism_lon_full, prism_lat_full)
    else:
        prism_lon_2d, prism_lat_2d = prism_lon_full, prism_lat_full

    # Crop reference grid to Skagit map extent bounds: [-122.5, -120.5, 47.8, 49.5]
    mask_crop = (prism_lon_2d >= -122.5) & (prism_lon_2d <= -120.5) & (prism_lat_2d >= 47.8) & (prism_lat_2d <= 49.5)
    rows, cols = np.where(mask_crop)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    ref_lon = prism_lon_2d[row_min:row_max+1, col_min:col_max+1]
    ref_lat = prism_lat_2d[row_min:row_max+1, col_min:col_max+1]
    print(f"Cropped PRISM 4km reference grid shape: {ref_lon.shape}")
    
    # Pre-crop UCLA coordinates for regridding coordination
    mask_crop_u = (ucla_lon >= -122.5) & (ucla_lon <= -120.5) & (ucla_lat >= 47.8) & (ucla_lat <= 49.5)
    rows_u, cols_u = np.where(mask_crop_u)
    row_min_u, row_max_u = rows_u.min(), rows_u.max()
    col_min_u, col_max_u = cols_u.min(), cols_u.max()
    ucla_lon_cropped = ucla_lon[row_min_u:row_max_u+1, col_min_u:col_max_u+1]
    ucla_lat_cropped = ucla_lat[row_min_u:row_max_u+1, col_min_u:col_max_u+1]

    # # Pre-crop HRRR coordinates for regridding coordination (commented out - not using HRRR)
    # mask_crop_h = (hrrr_lon >= -122.5) & (hrrr_lon <= -120.5) & (hrrr_lat >= 47.8) & (hrrr_lat <= 49.5)
    # rows_h, cols_h = np.where(mask_crop_h)
    # row_min_h, row_max_h = rows_h.min(), rows_h.max()
    # col_min_h, col_max_h = cols_h.min(), cols_h.max()
    # hrrr_lon_cropped = hrrr_lon[row_min_h:row_max_h+1, col_min_h:col_max_h+1]
    # hrrr_lat_cropped = hrrr_lat[row_min_h:row_max_h+1, col_min_h:col_max_h+1]
    
    # Load GIS boundary files early for regridding/clipping
    boundary_gdf = gpd.read_file(BOUNDARY_PATH)
    
    # Precompute 2D mask on the cropped curvilinear grid
    if hasattr(boundary_gdf.to_crs("EPSG:4326"), 'union_all'):
        poly = boundary_gdf.to_crs("EPSG:4326").union_all()
    else:
        poly = boundary_gdf.to_crs("EPSG:4326").unary_union
    df_pts = pd.DataFrame({'lon': ref_lon.flatten(), 'lat': ref_lat.flatten()})
    gdf_pts = gpd.GeoDataFrame(df_pts, geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat), crs="EPSG:4326")
    inside = gdf_pts.intersects(poly).values
    mask_2d = inside.reshape(ref_lon.shape)
    
    # Check cache to avoid re-extracting large raw datasets
    cache_path = os.path.join(OUT_DIR, "seasonal_avg_cache.pkl")
    years = range(1981, 2020) # 1981 to 2019
    if os.path.exists(cache_path):
        print(f"Loading cached seasonal averages from {cache_path}...")
        import pickle
        with open(cache_path, 'rb') as f:
            yearly_dicts = pickle.load(f)
    else:
        # Run years in parallel (n_jobs=12)
        print(f"Starting parallel data extraction across {len(years)} years with n_jobs=12...")
        yearly_dicts = Parallel(n_jobs=12)(delayed(process_year)(y) for y in years)
        print(f"Caching seasonal averages to {cache_path}...")
        import pickle
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(yearly_dicts, f)
        except Exception as e:
            print(f"  [WARNING] Failed to write cache: {e}")
    
    # 2. Combine and Average maps in their native grids with coordinate alignment
    print("Averaging multi-year seasonal averages...")
    products = ['PRISM', 'Daymet', 'PNNL', 'CONUS404', 'UCLA', 'GridMET']
    native_averages = {prod: {s: None for s in SEASONS} for prod in products}
    
    for prod in products:
        for s_name in SEASONS:
            grids = []
            template = None
            for y_dict in yearly_dicts:
                g = y_dict[prod][s_name]
                if g is not None:
                    if template is None:
                        template = g
                        grids.append(g)
                    else:
                        if g.shape == template.shape:
                            g_aligned = align_coords(g, template)
                            grids.append(g_aligned)
                        else:
                            # Mismatched shape (e.g. Daymet 2012-2013)
                            g_aligned = g.interp(lat=template.lat, lon=template.lon, method='linear')
                            grids.append(g_aligned)
            if grids:
                native_averages[prod][s_name] = xr.concat(grids, dim='year').mean(dim='year')
                
    # # Calculate PRISM 2014-2019 sub-average for HRRR bias comparison (not needed since HRRR is commented out)
    # prism_14_19_averages = {s: None for s in SEASONS}
    # for s_name in SEASONS:
    #     grids_14_19 = []
    #     template = None
    #     for y, y_dict in zip(years, yearly_dicts):
    #         if y >= 2014:
    #             g = y_dict['PRISM'][s_name]
    #             if g is not None:
    #                 if template is None:
    #                     template = g
    #                     grids_14_19.append(g)
    #                 else:
    #                     g_aligned = align_coords(g, template)
    #                     grids_14_19.append(g_aligned)
    #     if grids_14_19:
    #         prism_14_19_averages[s_name] = xr.concat(grids_14_19, dim='year').mean(dim='year')
                
    # 3. Regrid averages to the reference PRISM grid
    print("Regridding seasonal averages to PRISM 4km grid...")
    regridded_averages = {prod: {s: None for s in SEASONS} for prod in products}
    for prod in products:
        for s_name in SEASONS:
            da_src = native_averages[prod][s_name]
            if da_src is not None:
                if prod == 'UCLA':
                    da_src = da_src.assign_coords(lat=(('lat2d', 'lon2d'), ucla_lat_cropped), lon=(('lat2d', 'lon2d'), ucla_lon_cropped))
            regridded_averages[prod][s_name] = regrid_to_reference(da_src, ref_lat, ref_lon, mask_2d)
            
    # 4. Calculate differences relative to PRISM
    biases = ['Daymet', 'PNNL', 'CONUS404', 'UCLA', 'GridMET']
    regridded_biases = {b: {s: None for s in SEASONS} for b in biases}

    for b in biases:
        for s_name in SEASONS:
            product_grid = regridded_averages[b][s_name]
            prism_grid = regridded_averages['PRISM'][s_name]
            regridded_biases[b][s_name] = product_grid - prism_grid
            
    # 5. Determine dynamic color limits for maps
    all_precip = []
    all_diff = []
    for s_name in SEASONS:
        for prod in products:
            grid = regridded_averages[prod][s_name]
            if grid is not None:
                all_precip.extend(grid.flatten())
        for b in biases:
            grid = regridded_biases[b][s_name]
            if grid is not None:
                all_diff.extend(grid.flatten())
                
    all_precip = np.array(all_precip)
    all_precip = all_precip[~np.isnan(all_precip)]
    all_diff = np.array(all_diff)
    all_diff = all_diff[~np.isnan(all_diff)]
    
    # 99.5th percentile for limits to exclude extreme outliers
    vmax_precip = int(np.ceil(np.percentile(all_precip, 99.5)))
    vmax_precip = max(vmax_precip, 1)
    vmin_precip = 0
    
    vmax_diff = np.percentile(np.abs(all_diff), 99.5)
    vmax_diff = np.ceil(vmax_diff * 2) / 2.0  # round to nearest 0.5
    vmax_diff = max(vmax_diff, 0.5)
    vmin_diff = -vmax_diff
    
    print(f"Precipitation color range: {vmin_precip} to {vmax_precip} mm/day")
    print(f"Difference color range: {vmin_diff} to {vmax_diff} mm/day")
    
    # 6. Load geographical files
    subbasin_gdf = gpd.read_file(SUBBASIN_PATH) if os.path.exists(SUBBASIN_PATH) else None
    map_extent = [-122.35, -120.65, 47.90, 49.35]
    
    # 7. Generate Figure 1: Products
    print("Plotting direct products...")
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial']
    })
    
    fig1, axes1 = plt.subplots(4, 6, figsize=(18, 14),
                               subplot_kw={"projection": ccrs.PlateCarree()},
                               facecolor='#ffffff')
    
    im_precip = None
    for row_idx, s_name in enumerate(SEASONS):
        for col_idx, prod in enumerate(products):
            ax = axes1[row_idx, col_idx]
            grid = regridded_averages[prod][s_name]
            
            # plot using pcolormesh
            im_precip = ax.pcolormesh(ref_lon, ref_lat, grid, 
                                      transform=ccrs.PlateCarree(),
                                      cmap="Greens", vmin=vmin_precip, vmax=vmax_precip,
                                      shading='auto')
            
            # Styling
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, edgecolor='#444444')
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='#000000')
            ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle='--', linewidth=0.4, edgecolor='#666666')
            
            if subbasin_gdf is not None:
                subbasin_gdf.plot(ax=ax, facecolor="none", edgecolor="#888888", lw=0.5, linestyle=":", transform=ccrs.PlateCarree())
            boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#000000", lw=1.3, transform=ccrs.PlateCarree())
            
            # Row label on first column
            if col_idx == 0:
                ax.text(-0.25, 0.5, s_name, transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='center', ha='right')
                
        # Column titles
        if row_idx == 0:
            for col_idx, prod in enumerate(products):
                label = "ORNL (Daymet)" if prod == "ORNL" else prod
                axes1[0, col_idx].set_title(f"{label}\n(1981–2019)", fontsize=15, fontweight='bold', pad=12)
                
    # Add colorbar at bottom of Fig 1
    cbar_ax1 = fig1.add_axes([0.30, 0.05, 0.40, 0.025])
    cbar1 = fig1.colorbar(im_precip, cax=cbar_ax1, orientation='horizontal',
                         label="Average Seasonal Precipitation (mm/day)")
    cbar1.ax.tick_params(labelsize=12)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.10, wspace=0.05, hspace=0.08)
    plt.suptitle("Spatial Distribution of Seasonal Average Precipitation\nMulti-Product Comparison (1981–2019; PRISM 4km reference)",
                 fontsize=22, fontweight='bold', y=0.97)

    print(f"Saving Products map to {OUT_PRODUCTS_IMG}...")
    plt.savefig(OUT_PRODUCTS_IMG, dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Generate Figure 2: Biases
    print("Plotting biases (Product - PRISM)...")
    fig2, axes2 = plt.subplots(4, 5, figsize=(16, 14),
                               subplot_kw={"projection": ccrs.PlateCarree()},
                               facecolor='#ffffff')

    im_diff = None
    for row_idx, s_name in enumerate(SEASONS):
        for col_idx, b in enumerate(biases):
            ax = axes2[row_idx, col_idx]
            grid = regridded_biases[b][s_name]

            # plot using pcolormesh (diverging RdBu_r colormap centered at 0)
            im_diff = ax.pcolormesh(ref_lon, ref_lat, grid,
                                    transform=ccrs.PlateCarree(),
                                    cmap="RdBu_r", vmin=vmin_diff, vmax=vmax_diff,
                                    shading='auto')

            # Styling
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.6, edgecolor='#444444')
            ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, edgecolor='#000000')
            ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle='--', linewidth=0.4, edgecolor='#666666')

            if subbasin_gdf is not None:
                subbasin_gdf.plot(ax=ax, facecolor="none", edgecolor="#888888", lw=0.5, linestyle=":", transform=ccrs.PlateCarree())
            boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#000000", lw=1.3, transform=ccrs.PlateCarree())

            # Row label on first column
            if col_idx == 0:
                ax.text(-0.25, 0.5, s_name, transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='center', ha='right')

        # Column titles
        if row_idx == 0:
            for col_idx, b in enumerate(biases):
                label = f"{b} - PRISM"
                axes2[0, col_idx].set_title(f"{label}\n(1981–2019)", fontsize=15, fontweight='bold', pad=12)

    # Add colorbar at bottom of Fig 2
    cbar_ax2 = fig2.add_axes([0.30, 0.05, 0.40, 0.025])
    cbar2 = fig2.colorbar(im_diff, cax=cbar_ax2, orientation='horizontal',
                         label="Precipitation Bias relative to PRISM (Product - PRISM, mm/day)")
    cbar2.ax.tick_params(labelsize=12)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.10, wspace=0.05, hspace=0.08)
    plt.suptitle("Spatial Distribution of Seasonal Average Precipitation Biases\nProduct minus PRISM Baseline (1981–2019; PRISM 4km reference)",
                 fontsize=22, fontweight='bold', y=0.97)

    print(f"Saving Biases map to {OUT_BIASES_IMG}...")
    plt.savefig(OUT_BIASES_IMG, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Multi-product maps successfully generated!")

if __name__ == "__main__":
    main()
