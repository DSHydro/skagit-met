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

Daily data come from common/prism_day.py; with ALIGN = True all products are on the
PRISM day (12 UTC on D-1 to 12 UTC on D, GitHub issue #61). The cache file name
includes the alignment, so aligned and unaligned runs never share a cache.

Input
    Precipitation datasets - see the paths in common/prism_day.py
    data/GIS/SkagitBoundary.json, data/GIS/SkagitSubBasin_HUC8.geojson
Output
    plots/spatial_bias_distribution/plots/spatial_percentiles_products.png
    plots/spatial_bias_distribution/plots/spatial_percentiles_biases.png
    plots/spatial_bias_distribution/plots/percentiles_cache[_prism_day].pkl   (cache; _prism_day when ALIGN)

Usage
    python plot_spatial_percentiles.py
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.interpolate import griddata

# Shared daily loaders (common/prism_day.py); PYTHONPATH also reaches the parallel workers
_REPO_DIR = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO_DIR)
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [_REPO_DIR, os.environ.get("PYTHONPATH")]))
from common import prism_day  # noqa: E402

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration & Paths ---

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
BOUNDARY_PATH = BASE_DIR / "data" / "GIS" / "SkagitBoundary.json"
SUBBASIN_PATH = BASE_DIR / "data" / "GIS" / "SkagitSubBasin_HUC8.geojson"
OUT_DIR = Path(__file__).resolve().parent / "plots"

# Day alignment of the products (GitHub issue #61): True = all products on the PRISM
# day (12 UTC on D-1 to 12 UTC on D); False = each product's own date labels
ALIGN = True

# Product names used here -> names in common/prism_day.py
PRODUCT_KEYS = {'PRISM': 'prism', 'Daymet': 'daymet', 'PNNL': 'pnnl',
                'CONUS404': 'conus', 'UCLA': 'ucla', 'GridMET': 'gridmet'}

# Output paths
OUT_PRODUCTS_IMG = OUT_DIR / "spatial_percentiles_products.png"
OUT_BIASES_IMG = OUT_DIR / "spatial_percentiles_biases.png"
CACHE_PATH = OUT_DIR / f"percentiles_cache{'_prism_day' if ALIGN else ''}.pkl"

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

# Reference grid: PRISM 4 km (cropped to BB in main)
print("Loading reference grid...")
prism_sample_date = pd.Timestamp("2020-01-01")
da_prism_sample = prism_day.load_prism_day(prism_sample_date)
if da_prism_sample is None:
    raise FileNotFoundError(f"Could not find PRISM data for reference date {prism_sample_date.date()}")
prism_lon_full = da_prism_sample['x'].values
prism_lat_full = da_prism_sample['y'].values


def in_bb(lon, lat):
    """True where (lon, lat) is inside the crop box BB."""
    return (lon >= BB[0]) & (lon <= BB[1]) & (lat >= BB[2]) & (lat <= BB[3])


def process_year(year):
    """Daily P95/P99 at every native grid cell, for one calendar year and every product.

    Daily data come from common/prism_day.py (on the PRISM day when ALIGN is True),
    cropped to BB. Missing days are left out of the percentiles.
    """
    print(f"Processing Year: {year}...")
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq='D')
    year_results = {prod: {p: None for p in PERCENTILES} for prod in PRODUCTS}
    for prod in PRODUCTS:
        try:
            da = prism_day.load_daily(PRODUCT_KEYS[prod], dates, align=ALIGN, crop=BB)
            if da is None:
                continue
            for p_name, p_value in PERCENTILES.items():
                year_results[prod][p_name] = da.quantile(p_value, dim='time').drop_vars('quantile')
        except Exception as e:
            print(f"  [ERROR] {prod} {year}: {e}")
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
                            raise ValueError(f"{prod}: grid shape {g.shape} differs from {template.shape}")
            if grids:
                native_averages[prod][p_name] = xr.concat(grids, dim='year').mean(dim='year')

    # 3. Regrid percentile maps to the reference PRISM grid
    print("Regridding percentile maps to PRISM 4km grid...")
    regridded_averages = {prod: {p: None for p in PERCENTILES} for prod in products}
    for prod in products:
        for p_name in PERCENTILES:
            da_src = native_averages[prod][p_name]
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
