"""Spatial bias maps of 8-day cumulative precipitation during non-AR events.

For 6 selected non-AR precipitation events (AR scale 0), the cumulative
precipitation of each product over an 8-day window (T-2 to T+5, where T is the event day)
is computed from PRISM, Daymet, PNNL, CONUS404, UCLA and GridMET, regridded
(linear) to the PRISM 4 km grid and masked to the Skagit watershed
(SkagitBoundary.json). Daily data come from common/prism_day.py; with ALIGN =
True all products are on the PRISM day (12 UTC on D-1 to 12 UTC on D, GitHub
issue #61). Cells without data stay blank (e.g. gridMET does not cover the
Canadian part of the basin). The figure is a 6 x 5 grid of maps: one row per event and one
column per product, showing the bias relative to PRISM (product - PRISM).
SNOTEL station totals are overlaid as dots coloured by their own bias
(SNOTEL - PRISM at the station). The colour range is symmetric, set by the
99.5th percentile of the absolute bias over all maps.

Input
    Precipitation datasets - see the paths in common/prism_day.py
    data/GIS/SkagitBoundary.json, data/GIS/SkagitSubBasin_HUC8.geojson
    data/weather_data/<start>_<end>_SNOTEL_daily_data.zarr   (one per event)
Output
    plots/spatial_bias_distribution/plots/non_ar_events_cumulative_spatial_bias_grid_8_days.png

Usage
    python plot_non_ar_events_cumulative_spatial_bias_grid_8_days_updated.py
"""

import os
import subprocess
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
from scipy.interpolate import griddata

# Shared daily loaders (common/prism_day.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import prism_day  # noqa: E402

warnings.filterwarnings('ignore')

# --- Configuration & Paths ---

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
BOUNDARY_PATH = BASE_DIR / "data" / "GIS" / "SkagitBoundary.json"
SUBBASIN_PATH = BASE_DIR / "data" / "GIS" / "SkagitSubBasin_HUC8.geojson"
OUT_DIR = Path(__file__).resolve().parent / "plots"
os.makedirs(OUT_DIR, exist_ok=True)

SNOTEL_DIR = DATA_DIR / "weather_data"  # <start>_<end>_SNOTEL_daily_data.zarr
SNOTEL_DOWNLOADER = BASE_DIR / "downloaders" / "snotel_downloader.py"  # used only if a SNOTEL zarr is missing

# --- Non-AR event windows (5-day window, taking the min): 8 days, T-2 to T+5 around the event day T ---
NON_AR_EVENTS = [
    {"label": "April_1996_AR0", "start": "1996-04-23", "end": "1996-04-30", "row_label": "Apr 23–30, 1996\n(AR0)"},
    {"label": "August_2004_AR0", "start": "2004-08-24", "end": "2004-08-31", "row_label": "Aug 24–31, 2004\n(AR0)"},
    {"label": "December_2017_AR0", "start": "2017-12-28", "end": "2018-01-04", "row_label": "Dec 28, 2017–Jan 4, 2018\n(AR0)"},
    {"label": "September_1997_AR0", "start": "1997-09-15", "end": "1997-09-22", "row_label": "Sep 15–22, 1997\n(AR0)"},
    {"label": "February_2014_AR0", "start": "2014-02-15", "end": "2014-02-22", "row_label": "Feb 15–22, 2014\n(AR0)"},
    {"label": "February_1982_AR0", "start": "1982-01-30", "end": "1982-02-06", "row_label": "Jan 30–Feb 6, 1982\n(AR0)"}
]

PRODUCTS = ['PRISM', 'Daymet', 'PNNL', 'CONUS404', 'UCLA', 'GridMET']
BIAS_PRODUCTS = [p for p in PRODUCTS if p != 'PRISM']

MM_PER_INCH = 25.4  # SNOTEL precipitation is in inches
BIAS_PERCENTILE = 0.995  # colour range = this quantile of |bias| over all maps
BIAS_CMAP = "RdBu_r"

# Figure settings. Publication font sizes (figure is ~15 in wide; expect ~0.45x reduction to journal page width)
FIGSIZE = (15, 17.5)
DPI = 300
FONT_SUPTITLE = 24
FONT_TITLE = 18       # column titles
FONT_ROW_LABEL = 18
FONT_CBAR_LABEL = 18
FONT_CBAR_TICK = 16
FONT_NA = 16

# Day alignment of the products (GitHub issue #61): True = all products on the PRISM
# day (12 UTC on D-1 to 12 UTC on D); False = each product's own date labels
ALIGN = True

# Product names used here -> names in common/prism_day.py
PRODUCT_KEYS = {'PRISM': 'prism', 'Daymet': 'daymet', 'PNNL': 'pnnl',
                'CONUS404': 'conus', 'UCLA': 'ucla', 'GridMET': 'gridmet'}

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

print("Loading reference grid...")
prism_sample_date = pd.Timestamp("2020-01-01")
da_prism_sample = prism_day.load_prism_day(prism_sample_date)
if da_prism_sample is None:
    raise FileNotFoundError(f"Could not find PRISM data for reference date {prism_sample_date.date()}")
prism_lon_2d, prism_lat_2d = np.meshgrid(da_prism_sample['x'].values, da_prism_sample['y'].values)
row_min_p, row_max_p, col_min_p, col_max_p = crop_rows_cols(prism_lon_2d, prism_lat_2d)
ref_lon = prism_lon_2d[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
ref_lat = prism_lat_2d[row_min_p:row_max_p+1, col_min_p:col_max_p+1]
print(f"Reference PRISM grid shape: {ref_lon.shape}")

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
    """Total precipitation of each product over the event window, on its native grid.

    Daily data come from common/prism_day.py (on the PRISM day when ALIGN is True,
    on each product's own date labels otherwise), cropped to BB. A product is left
    out (drawn as N/A) unless every day of the window is available.
    """
    start = event["start"]
    end   = event["end"]
    dates = pd.date_range(start, end, freq='D')
    print(f"\n  Loading data for {event['label']} ({start} → {end})...")

    grids = {p: None for p in PRODUCTS}
    for prod in PRODUCTS:
        da = prism_day.load_daily(PRODUCT_KEYS[prod], dates, align=ALIGN, crop=BB)
        n_days = 0 if da is None else da.sizes['time']
        if n_days < len(dates):
            print(f"    [WARN] {prod}: {n_days}/{len(dates)} days available - left out")
            continue
        grids[prod] = da.sum(dim='time', skipna=False)
        print(f"    {prod} OK ({n_days} days)")
    return grids


def main():
    # Load and regrid all events
    event_regridded_grids = {}
    event_snotel_data = {}

    for event in NON_AR_EVENTS:
        start = event["start"]
        end = event["end"]
        print(f"\nProcessing: {event['label']}")
        grids = load_event_grids(event)

        # Load SNOTEL data
        zarr_path = os.path.join(SNOTEL_DIR, f"{start}_{end}_SNOTEL_daily_data.zarr")
        if not os.path.exists(zarr_path):
            print(f"  Downloading SNOTEL daily data for {start} to {end}...")
            try:
                subprocess.run([
                    sys.executable,
                    str(SNOTEL_DOWNLOADER),
                    "--startDate", start,
                    "--endDate", end,
                    "--geojson", str(BOUNDARY_PATH),
                    "--frequency", "daily",
                    "--outputDir", str(SNOTEL_DIR)
                ], check=True)
            except subprocess.CalledProcessError:
                print(f"  [WARN] SNOTEL download failed (metloom module may not be available)")

        # Load SNOTEL data and filter to event period (start to end)
        try:
            ds_snotel = xr.open_zarr(zarr_path)
            snotel_lats = ds_snotel.lat.values
            snotel_lons = ds_snotel.lon.values
            snotel_cumulative_inches = ds_snotel.PRECIPITATION.sel(time=slice(start, end)).sum(dim='time').values
            snotel_cumulative_mm = snotel_cumulative_inches * MM_PER_INCH

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

    # Compute PRISM values at SNOTEL station locations for bias calculation
    print("\nComputing PRISM values at SNOTEL stations...")
    event_snotel_prism = {}
    for event in NON_AR_EVENTS:
        elabel = event['label']
        snotel = event_snotel_data[elabel]
        if snotel is not None:
            prism_grid = event_regridded_grids[elabel]['PRISM']
            if prism_grid is not None and not np.all(np.isnan(prism_grid)):
                pts = np.column_stack((ref_lon.flatten(), ref_lat.flatten()))
                prism_vals_flat = prism_grid.flatten()
                ok = ~np.isnan(prism_vals_flat)
                if ok.any():
                    prism_at_stations = griddata(
                        pts[ok], prism_vals_flat[ok],
                        np.column_stack((snotel['lons'], snotel['lats'])),
                        method='linear'
                    )
                    event_snotel_prism[elabel] = prism_at_stations
                else:
                    event_snotel_prism[elabel] = None
            else:
                event_snotel_prism[elabel] = None
        else:
            event_snotel_prism[elabel] = None

    # Compute SNOTEL bias (SNOTEL - PRISM)
    print("Computing SNOTEL station bias...")
    event_snotel_bias = {}
    for event in NON_AR_EVENTS:
        elabel = event['label']
        snotel = event_snotel_data[elabel]
        prism_at_stn = event_snotel_prism[elabel]
        if snotel is not None and prism_at_stn is not None:
            bias = snotel['vals'] - prism_at_stn
            event_snotel_bias[elabel] = bias
        else:
            event_snotel_bias[elabel] = None

    # Compute bias grids (product - PRISM, only where PRISM source data exists)
    print("\nComputing bias grids (relative to PRISM)...")
    event_bias_grids = {}
    for elabel in event_regridded_grids:
        event_bias_grids[elabel] = {}
        prism_grid = event_regridded_grids[elabel]['PRISM']
        prism_regridded_valid = ~np.isnan(prism_grid)

        for prod in PRODUCTS:
            if prod != 'PRISM':
                prod_grid = event_regridded_grids[elabel][prod]
                bias = np.where(
                    prism_regridded_valid & ~np.isnan(prod_grid),
                    prod_grid - prism_grid,
                    np.nan
                )
                bias[~mask_2d] = np.nan
                event_bias_grids[elabel][prod] = bias

    # Dynamic bias limits: BIAS_PERCENTILE (99.5th percentile) of absolute bias
    print("\nCalculating dynamic bias colorbar limits...")
    all_bias_arrays = []
    for elabel in event_bias_grids:
        for prod in PRODUCTS:
            if prod != 'PRISM':
                g = event_bias_grids[elabel][prod]
                if g is not None:
                    v = g[~np.isnan(g)]
                    if len(v) > 0:
                        all_bias_arrays.append(np.abs(v))
    if all_bias_arrays:
        all_bias_vals = np.concatenate(all_bias_arrays)
        vmax_bias = float(np.ceil(np.quantile(all_bias_vals, BIAS_PERCENTILE)))
    else:
        vmax_bias = 50.0
    vmin_bias = -vmax_bias
    print(f"Bias color range: {vmin_bias}–{vmax_bias} mm")

    # Plot grid: 6 rows × 5 columns (bias relative to PRISM)
    plt.rcParams.update({
        'font.size': FONT_CBAR_LABEL,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
    })

    fig, axes = plt.subplots(
        len(NON_AR_EVENTS), len(BIAS_PRODUCTS),
        figsize=FIGSIZE,
        subplot_kw={"projection": ccrs.PlateCarree()},
        facecolor='#ffffff'
    )

    im = None
    for row_idx, event in enumerate(NON_AR_EVENTS):
        elabel = event['label']
        for col_idx, prod in enumerate(BIAS_PRODUCTS):
            ax = axes[row_idx, col_idx]
            grid = event_bias_grids[elabel][prod]

            if grid is not None and not np.all(np.isnan(grid)):
                im = ax.pcolormesh(
                    ref_lon, ref_lat, grid,
                    transform=ccrs.PlateCarree(),
                    cmap=BIAS_CMAP, vmin=vmin_bias, vmax=vmax_bias,
                    shading='auto'
                )

                # Overlay SNOTEL bias at station locations
                snotel = event_snotel_data[elabel]
                snotel_bias = event_snotel_bias[elabel]
                if snotel is not None and snotel_bias is not None:
                    ax.scatter(
                        snotel['lons'], snotel['lats'],
                        c=snotel_bias, cmap=BIAS_CMAP,
                        vmin=vmin_bias, vmax=vmax_bias,
                        edgecolors="black", linewidths=1.2,
                        s=60, transform=ccrs.PlateCarree(),
                        zorder=10
                    )
            else:
                # Grey placeholder if product missing
                ax.set_facecolor('#dddddd')
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=FONT_NA, color='#555555')

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
                        fontsize=FONT_ROW_LABEL, fontweight='bold', va='center', ha='right')

        # Column titles on the first row
        if row_idx == 0:
            for col_idx, prod in enumerate(BIAS_PRODUCTS):
                label = f"{prod} - PRISM"
                axes[0, col_idx].set_title(label, fontsize=FONT_TITLE, fontweight='bold', pad=12)

    # Colorbar at the bottom of the grid
    if im is not None:
        cbar_ax = fig.add_axes([0.30, 0.05, 0.40, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                            label="Bias relative to PRISM (mm)")
        cbar.ax.tick_params(labelsize=FONT_CBAR_TICK)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10, wspace=0.05, hspace=0.08)

    plt.suptitle(
        "Bias in Cumulative Precipitation during Non-Atmospheric River Events\n"
        "Multi-Product Comparison (Bias relative to PRISM)\n"
        "(Non-AR Event Day: 2 days after period start)",
        fontsize=FONT_SUPTITLE, fontweight='bold', y=0.99
    )

    out_png = os.path.join(OUT_DIR, "non_ar_events_cumulative_spatial_bias_grid_8_days.png")
    plt.savefig(out_png, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\nSaved combined comparison figure to: {out_png}")


if __name__ == "__main__":
    main()
