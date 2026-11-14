"""Spatial maps of extreme daily precipitation percentiles (P95, P99) and their bias.

For each calendar year 1981-2019 and each product (PRISM, Daymet, PNNL,
CONUS404, UCLA, GridMET), the 95th and 99th percentiles of daily
precipitation are computed at every grid cell on the product's native grid.
The yearly percentile maps are averaged over all years, regridded (linear) to
the PRISM 4 km grid and masked to the Skagit watershed. Two figures are made:

    1. Percentile maps of every product (rows: P95, P99; columns: products).
    2. Bias maps relative to PRISM (product - PRISM). The colour ranges are
       set by the 99.5th percentile of all mapped values.

The yearly extraction is slow (39 years, run in parallel), so its result is
cached in plots/percentiles_cache.pkl and reused on later runs. Delete the
cache to re-extract (e.g. after changing YEARS, PERCENTILES or the data).

Input
    /data0/skagit_met/data_transfer/data/...   (gridded products, outside this repository)
    data/weather_data/conus404_skagit_precip_daily_full.zarr
    data/GIS/SkagitBoundary.json, data/GIS/SkagitSubBasin_HUC8.geojson
Output
    plots/spatial_bias_distribution/plots/spatial_percentiles_products.png
    plots/spatial_bias_distribution/plots/spatial_percentiles_biases.png
    plots/spatial_bias_distribution/plots/percentiles_cache.pkl   (cache)

Usage
    python plot_spatial_percentiles.py
"""

import os
import pickle
import tempfile
import warnings
import zipfile
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from joblib import Parallel, delayed
from scipy.interpolate import griddata

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration & Paths ---
# Shared raw-data archive (outside this repository)
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PRISM_ROOT = os.path.join(VAULT_DIR, "prism_new_hq")
DAYMET_ROOT = os.path.join(VAULT_DIR, "daymet_new_hq")
PNNL_GEO = os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc")  # PNNL grid coordinates
PNNL_DIR = os.path.join(VAULT_DIR, "PNNL/historical")  # <year>/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.<year>.nc
UCLA_STATIC = os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc")  # UCLA grid coordinates
UCLA_PREC_DIR = os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec")  # prec.daily.era5.d02.<year>.nc
GRIDMET_DIR = os.path.join(VAULT_DIR, "gridmet")  # <year>_daily_4km_gridMET_data.zarr

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
CONUS_ZARR = BASE_DIR / "data" / "weather_data" / "conus404_skagit_precip_daily_full.zarr"
BOUNDARY_PATH = BASE_DIR / "data" / "GIS" / "SkagitBoundary.json"
SUBBASIN_PATH = BASE_DIR / "data" / "GIS" / "SkagitSubBasin_HUC8.geojson"
OUT_DIR = Path(__file__).resolve().parent / "plots"

# Output paths
OUT_PRODUCTS_IMG = OUT_DIR / "spatial_percentiles_products.png"
OUT_BIASES_IMG = OUT_DIR / "spatial_percentiles_biases.png"
CACHE_PATH = OUT_DIR / "percentiles_cache.pkl"

# Analysis settings
YEARS = range(1981, 2020)  # calendar years 1981-2019
N_JOBS = 12  # years processed in parallel when (re)building the cache
PERCENTILES = {  # percentile levels of daily precipitation
    "P95": 0.95,
    "P99": 0.99
}
PRODUCTS = ['PRISM', 'Daymet', 'PNNL', 'CONUS404', 'UCLA', 'GridMET']
BIAS_PRODUCTS = [p for p in PRODUCTS if p != 'PRISM']

# Crop box around the Skagit basin (lon_min, lon_max, lat_min, lat_max) and map extent
BB = (-122.5, -120.5, 47.8, 49.5)
MAP_EXTENT = [-122.35, -120.65, 47.90, 49.35]
COLOR_LIMIT_PERCENTILE = 99.5  # colour range = this percentile of all mapped values

# Figure settings. Publication font sizes (figures are 16-18 in wide; expect ~0.4x reduction to journal page width)
FIGSIZE_PRODUCTS = (18, 8)
FIGSIZE_BIASES = (16, 8)
DPI = 300
CMAP_PRECIP = "Greens"
CMAP_BIAS = "RdBu_r"
FONT_SUPTITLE = 24
FONT_TITLE = 18       # column titles
FONT_ROW_LABEL = 20
FONT_CBAR_LABEL = 18
FONT_CBAR_TICK = 16

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
ds_pnnl_static = xr.open_dataset(PNNL_GEO)
pnnl_lon = ds_pnnl_static.XLONG_M.values[0]
pnnl_lat = ds_pnnl_static.XLAT_M.values[0]
ds_pnnl_static.close()

# UCLA
ds_ucla_static = xr.open_dataset(UCLA_STATIC)
ucla_lon = ds_ucla_static.lon2d.values
ucla_lat = ds_ucla_static.lat2d.values
ds_ucla_static.close()


def in_bb(lon, lat):
    """True where (lon, lat) is inside the crop box BB."""
    return (lon >= BB[0]) & (lon <= BB[1]) & (lat >= BB[2]) & (lat <= BB[3])


def process_year(year):
    """Processes all products for a single year and extracts daily percentiles (P95, P99) in their native grids."""
    print(f"Processing Year: {year}...")
    year_results = {}

    # Initialize percentile maps
    for prod in PRODUCTS:
        year_results[prod] = {p: None for p in PERCENTILES}
        
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
            for p_name, p_value in PERCENTILES.items():
                perc_s = da.quantile(p_value, dim='time').compute()
                year_results['PRISM'][p_name] = perc_s
    except Exception as e:
        print(f"  [ERROR] PRISM {year}: {e}")

    # 2. Daymet (new netCDF source)
    try:
        ds = load_daymet_from_netcdf(year, DAYMET_ROOT)
        if ds is not None:
            ds['time'] = pd.to_datetime(ds.time.values).normalize()
            var = 'prcp' if 'prcp' in ds.data_vars else 'ppt'
            for p_name, p_value in PERCENTILES.items():
                perc_s = ds[var].quantile(p_value, dim='time').compute()
                year_results['Daymet'][p_name] = perc_s
            ds.close()
    except Exception as e:
        print(f"  [ERROR] Daymet {year}: {e}")
        
    # 3. PNNL WRF
    try:
        pnnl_file = os.path.join(PNNL_DIR, str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        if os.path.exists(pnnl_file):
            ds = xr.open_dataset(pnnl_file, chunks={'time': 720})
            da_daily = ds['PREC_ACC_NC'].resample(time='1D').sum()

            # Crop PNNL WRF to Skagit bounding box
            mask_crop = in_bb(pnnl_lon, pnnl_lat)
            rows, cols = np.where(mask_crop)
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            da_daily = da_daily.isel(y=slice(row_min, row_max+1), x=slice(col_min, col_max+1))
            pnnl_lat_cropped = pnnl_lat[row_min:row_max+1, col_min:col_max+1]
            pnnl_lon_cropped = pnnl_lon[row_min:row_max+1, col_min:col_max+1]

            for p_name, p_value in PERCENTILES.items():
                perc_s = da_daily.quantile(p_value, dim='time').compute()
                perc_s = perc_s.assign_coords(lat=(('y', 'x'), pnnl_lat_cropped), lon=(('y', 'x'), pnnl_lon_cropped))
                year_results['PNNL'][p_name] = perc_s
            ds.close()
    except Exception as e:
        print(f"  [ERROR] PNNL {year}: {e}")
        
    # 4. CONUS404
    try:
        ds = xr.open_zarr(CONUS_ZARR)

        # Crop CONUS404 to Skagit bounding box
        mask_crop = in_bb(ds.lon, ds.lat)
        rows, cols = np.where(mask_crop.values)
        if len(rows) > 0:
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            ds = ds.isel(y=slice(row_min, row_max+1), x=slice(col_min, col_max+1))
            
        ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        for p_name, p_value in PERCENTILES.items():
            perc_s = ds_year['precip_daily'].quantile(p_value, dim='time').compute()
            year_results['CONUS404'][p_name] = perc_s
        ds.close()
    except Exception as e:
        print(f"  [ERROR] CONUS404 {year}: {e}")
        
    # 5. UCLA (each file runs Sep to Aug, so a calendar year spans two files)
    try:
        u_path_prev = os.path.join(UCLA_PREC_DIR, f"prec.daily.era5.d02.{year-1}.nc")
        u_path_curr = os.path.join(UCLA_PREC_DIR, f"prec.daily.era5.d02.{year}.nc")
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
            mask_crop = in_bb(ucla_lon, ucla_lat)
            rows, cols = np.where(mask_crop)
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            da_year = da_year.isel(lat2d=slice(row_min, row_max+1), lon2d=slice(col_min, col_max+1))
            ucla_lon_cropped = ucla_lon[row_min:row_max+1, col_min:col_max+1]
            ucla_lat_cropped = ucla_lat[row_min:row_max+1, col_min:col_max+1]

            for p_name, p_value in PERCENTILES.items():
                perc_s = da_year.quantile(p_value, dim='day')
                perc_s = perc_s.assign_coords(lat=(('lat2d', 'lon2d'), ucla_lat_cropped), lon=(('lat2d', 'lon2d'), ucla_lon_cropped))
                year_results['UCLA'][p_name] = perc_s
    except Exception as e:
        print(f"  [ERROR] UCLA {year}: {e}")
        
    # 6. GridMET
    try:
        gridmet_path = os.path.join(GRIDMET_DIR, f"{year}_daily_4km_gridMET_data.zarr")
        if os.path.exists(gridmet_path):
            ds = xr.open_zarr(gridmet_path)
            if 'day' in ds.dims:
                ds = ds.rename({'day': 'time'})
            for p_name, p_value in PERCENTILES.items():
                perc_s = ds['prcp'].quantile(p_value, dim='time').compute()
                year_results['GridMET'][p_name] = perc_s
            ds.close()
    except Exception as e:
        print(f"  [ERROR] GridMET {year}: {e}")

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

    # Crop reference grid to the Skagit bounding box BB
    mask_crop = in_bb(prism_lon_2d, prism_lat_2d)
    rows, cols = np.where(mask_crop)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    ref_lon = prism_lon_2d[row_min:row_max+1, col_min:col_max+1]
    ref_lat = prism_lat_2d[row_min:row_max+1, col_min:col_max+1]
    print(f"Cropped PRISM 4km reference grid shape: {ref_lon.shape}")
    
    # Pre-crop UCLA coordinates for regridding coordination
    mask_crop_u = in_bb(ucla_lon, ucla_lat)
    rows_u, cols_u = np.where(mask_crop_u)
    row_min_u, row_max_u = rows_u.min(), rows_u.max()
    col_min_u, col_max_u = cols_u.min(), cols_u.max()
    ucla_lon_cropped = ucla_lon[row_min_u:row_max_u+1, col_min_u:col_max_u+1]
    ucla_lat_cropped = ucla_lat[row_min_u:row_max_u+1, col_min_u:col_max_u+1]

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
    cache_path = CACHE_PATH
    if os.path.exists(cache_path):
        print(f"Loading cached yearly percentile maps from {cache_path}...")
        with open(cache_path, 'rb') as f:
            yearly_dicts = pickle.load(f)
    else:
        # Run years in parallel
        print(f"Starting parallel data extraction across {len(YEARS)} years with n_jobs={N_JOBS}...")
        yearly_dicts = Parallel(n_jobs=N_JOBS)(delayed(process_year)(y) for y in YEARS)
        print(f"Caching yearly percentile maps to {cache_path}...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(yearly_dicts, f)
        except Exception as e:
            print(f"  [WARNING] Failed to write cache: {e}")

    # 2. Combine and Average percentile maps in their native grids with coordinate alignment
    print("Averaging multi-year percentile maps...")
    products = PRODUCTS
    native_averages = {prod: {p: None for p in PERCENTILES} for prod in products}

    for prod in products:
        for p_name in PERCENTILES:
            grids = []
            template = None
            for y_dict in yearly_dicts:
                g = y_dict[prod][p_name]
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
                native_averages[prod][p_name] = xr.concat(grids, dim='year').mean(dim='year')

    # 3. Regrid percentile maps to the reference PRISM grid
    print("Regridding percentile maps to PRISM 4km grid...")
    regridded_averages = {prod: {p: None for p in PERCENTILES} for prod in products}
    for prod in products:
        for p_name in PERCENTILES:
            da_src = native_averages[prod][p_name]
            if da_src is not None:
                if prod == 'UCLA':
                    da_src = da_src.assign_coords(lat=(('lat2d', 'lon2d'), ucla_lat_cropped), lon=(('lat2d', 'lon2d'), ucla_lon_cropped))
            regridded_averages[prod][p_name] = regrid_to_reference(da_src, ref_lat, ref_lon, mask_2d)

    # 4. Calculate differences relative to PRISM
    biases = BIAS_PRODUCTS
    regridded_biases = {b: {p: None for p in PERCENTILES} for b in biases}

    for b in biases:
        for p_name in PERCENTILES:
            product_grid = regridded_averages[b][p_name]
            prism_grid = regridded_averages['PRISM'][p_name]
            regridded_biases[b][p_name] = product_grid - prism_grid
            
    # 5. Determine dynamic color limits for maps
    all_precip = []
    all_diff = []
    for p_name in PERCENTILES:
        for prod in products:
            grid = regridded_averages[prod][p_name]
            if grid is not None:
                all_precip.extend(grid.flatten())
        for b in biases:
            grid = regridded_biases[b][p_name]
            if grid is not None:
                all_diff.extend(grid.flatten())
                
    all_precip = np.array(all_precip)
    all_precip = all_precip[~np.isnan(all_precip)]
    all_diff = np.array(all_diff)
    all_diff = all_diff[~np.isnan(all_diff)]
    
    # 99.5th percentile for limits to exclude extreme outliers
    vmax_precip = int(np.ceil(np.percentile(all_precip, COLOR_LIMIT_PERCENTILE)))
    vmax_precip = max(vmax_precip, 1)
    vmin_precip = 0

    vmax_diff = np.percentile(np.abs(all_diff), COLOR_LIMIT_PERCENTILE)
    vmax_diff = np.ceil(vmax_diff * 2) / 2.0  # round to nearest 0.5
    vmax_diff = max(vmax_diff, 0.5)
    vmin_diff = -vmax_diff
    
    print(f"Precipitation color range: {vmin_precip} to {vmax_precip} mm/day")
    print(f"Difference color range: {vmin_diff} to {vmax_diff} mm/day")
    
    # 6. Load geographical files
    subbasin_gdf = gpd.read_file(SUBBASIN_PATH) if os.path.exists(SUBBASIN_PATH) else None
    map_extent = MAP_EXTENT

    # 7. Generate Figure 1: Products
    print("Plotting direct products...")
    plt.rcParams.update({
        'font.size': FONT_CBAR_LABEL,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial']
    })

    fig1, axes1 = plt.subplots(len(PERCENTILES), len(products), figsize=FIGSIZE_PRODUCTS,
                               subplot_kw={"projection": ccrs.PlateCarree()},
                               facecolor='#ffffff')

    im_precip = None
    for row_idx, p_name in enumerate(PERCENTILES):
        for col_idx, prod in enumerate(products):
            ax = axes1[row_idx, col_idx]
            grid = regridded_averages[prod][p_name]

            # plot using pcolormesh
            im_precip = ax.pcolormesh(ref_lon, ref_lat, grid,
                                      transform=ccrs.PlateCarree(),
                                      cmap=CMAP_PRECIP, vmin=vmin_precip, vmax=vmax_precip,
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
                ax.text(-0.25, 0.5, p_name, transform=ax.transAxes,
                        fontsize=FONT_ROW_LABEL, fontweight='bold', va='center', ha='right')

        # Column titles
        if row_idx == 0:
            for col_idx, prod in enumerate(products):
                axes1[0, col_idx].set_title(f"{prod}", fontsize=FONT_TITLE, fontweight='bold', pad=12)
                
    # Add colorbar at bottom of Fig 1
    cbar_ax1 = fig1.add_axes([0.30, 0.05, 0.40, 0.025])
    cbar1 = fig1.colorbar(im_precip, cax=cbar_ax1, orientation='horizontal',
                         label="Daily Precipitation Percentile (mm/day)")
    cbar1.ax.tick_params(labelsize=FONT_CBAR_TICK)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.05, hspace=0.15)
    plt.suptitle("Spatial Distribution of Daily Precipitation Percentiles (P95, P99)\nMulti-Product Comparison (1981–2019; PRISM 4km reference)",
                 fontsize=FONT_SUPTITLE, fontweight='bold', y=1.06)

    print(f"Saving Products map to {OUT_PRODUCTS_IMG}...")
    plt.savefig(OUT_PRODUCTS_IMG, dpi=DPI, bbox_inches='tight')
    plt.close()

    # 8. Generate Figure 2: Biases
    print("Plotting biases (Product - PRISM)...")
    fig2, axes2 = plt.subplots(len(PERCENTILES), len(biases), figsize=FIGSIZE_BIASES,
                               subplot_kw={"projection": ccrs.PlateCarree()},
                               facecolor='#ffffff')

    im_diff = None
    for row_idx, p_name in enumerate(PERCENTILES):
        for col_idx, b in enumerate(biases):
            ax = axes2[row_idx, col_idx]
            grid = regridded_biases[b][p_name]

            # plot using pcolormesh (diverging RdBu_r colormap centered at 0)
            im_diff = ax.pcolormesh(ref_lon, ref_lat, grid,
                                    transform=ccrs.PlateCarree(),
                                    cmap=CMAP_BIAS, vmin=vmin_diff, vmax=vmax_diff,
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
                ax.text(-0.25, 0.5, p_name, transform=ax.transAxes,
                        fontsize=FONT_ROW_LABEL, fontweight='bold', va='center', ha='right')

        # Column titles
        if row_idx == 0:
            for col_idx, b in enumerate(biases):
                label = f"{b} - PRISM"
                axes2[0, col_idx].set_title(f"{label}", fontsize=FONT_TITLE, fontweight='bold', pad=12)

    # Add colorbar at bottom of Fig 2
    cbar_ax2 = fig2.add_axes([0.30, 0.05, 0.40, 0.025])
    cbar2 = fig2.colorbar(im_diff, cax=cbar_ax2, orientation='horizontal',
                         label="Bias in Precipitation Percentile (Product - PRISM, mm/day)")
    cbar2.ax.tick_params(labelsize=FONT_CBAR_TICK)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.05, hspace=0.15)
    plt.suptitle("Spatial Distribution of Daily Precipitation Percentile Biases (P95, P99)\nProduct minus PRISM Baseline (1981–2019; PRISM 4km reference)",
                 fontsize=FONT_SUPTITLE, fontweight='bold', y=1.06)

    print(f"Saving Biases map to {OUT_BIASES_IMG}...")
    plt.savefig(OUT_BIASES_IMG, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print("Multi-product maps successfully generated!")

if __name__ == "__main__":
    main()
