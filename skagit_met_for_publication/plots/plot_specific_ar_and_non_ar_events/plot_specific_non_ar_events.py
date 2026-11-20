"""Cumulative precipitation of six products during selected non-AR events.

For each event in ``SPECIFIC_DATES``, daily basin-mean precipitation over the
Skagit Basin (Upper Skagit, Sauk and Lower Skagit HUC8 sub-basins) is
extracted from six gridded products (PRISM, PNNL, Daymet, CONUS404, UCLA,
GridMET) for a 9-day window (T-3 to T+5), and the cumulative totals are
plotted in a 3x2 panel figure. The cumulative mean daily precipitation of the
SNOTEL stations inside the basin is overlaid as a dashed black line (station
mean, not a gridded basin mean; SNOTEL zarrs are downloaded on demand with
downloaders/snotel_downloader.py). Each panel title gives the event's AR scale
and the peak USGS 12200500 discharge in the window. All panels share the same
y-axis range.

Input
    /data0/skagit_met/data_transfer/data/...      (gridded precipitation products)
    data/GIS/SkagitSubBasin_HUC8.geojson
    <VAULT_DIR>/conus404_daily/conus404_skagit_precip_daily_full.zarr
    data/usgs_12200500_discharge.rdb
    analysis/outputs/5day_taking_min/3_multi_product_bulk_bias_data.csv
Output
    plots/plot_specific_ar_and_non_ar_events/specific_non_ar_events_cumulative_precipitation.png

Usage
    python plot_specific_non_ar_events.py
"""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Shared daily loaders (common/prism_day.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import prism_day  # noqa: E402

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
# Event table from the "5-day window, taking the min" run (used for the AR scale in panel titles)
EVENTS_CSV = BASE_DIR / "analysis" / "outputs" / "5day_taking_min" / "3_multi_product_bulk_bias_data.csv"
HYDRO_Q_PATH = BASE_DIR / "data" / "usgs_12200500_discharge.rdb"
OUTPUT_DIR = Path(__file__).resolve().parent

# SNOTEL station data: <start>_<end>_SNOTEL_daily_data.zarr, downloaded on demand
SNOTEL_DIR = BASE_DIR / "data" / "weather_data"
SNOTEL_DOWNLOADER = BASE_DIR / "downloaders" / "snotel_downloader.py"
SNOTEL_GEOJSON = BASE_DIR / "data" / "GIS" / "SkagitBoundary.json"
INCH_TO_MM = 25.4

# Products, in plotting order, with their line colors and markers
PRODUCTS = ['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet', 'snotel']
PRODUCT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#000000']
PRODUCT_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*']
PRODUCT_LINESTYLES = ['-', '-', '-', '-', '-', '-', '--']

# Day alignment of the products (GitHub issue #61): True = all products on the PRISM
# day (12 UTC on D-1 to 12 UTC on D); False = each product's own date labels (as before)
ALIGN = True

# Event window: T-3 to T+5 days
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
FONT_LEGEND = 16

# Non-AR event dates to plot (top-discharge non-AR events, 5-day window, taking the min)
SPECIFIC_DATES = [
    '1996-04-25',
    '2004-08-26',
    '2017-12-30',
    '1997-09-17',
    '2014-02-17',
    '1982-02-01'
]


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


def load_snotel_daily_mean(window_dates):
    """Daily precipitation (mm) averaged over the SNOTEL stations in the basin.

    Reads <start>_<end>_SNOTEL_daily_data.zarr from SNOTEL_DIR, downloading it
    with the SNOTEL downloader if missing. Returns (series indexed by date,
    number of stations), or (None, 0) if the data are unavailable.
    """
    start = window_dates[0].strftime('%Y-%m-%d')
    end = window_dates[-1].strftime('%Y-%m-%d')
    zarr_path = SNOTEL_DIR / f"{start}_{end}_SNOTEL_daily_data.zarr"

    if not zarr_path.exists():
        print(f"  Downloading SNOTEL daily data for {start} to {end}...")
        try:
            subprocess.run([
                sys.executable, str(SNOTEL_DOWNLOADER),
                "--startDate", start,
                "--endDate", end,
                "--geojson", str(SNOTEL_GEOJSON),
                "--frequency", "daily",
                "--outputDir", str(SNOTEL_DIR),
            ], check=True)
        except subprocess.CalledProcessError:
            print("  [WARN] SNOTEL download failed")
            return None, 0

    try:
        ds = xr.open_zarr(zarr_path)
        precip_mm = ds['PRECIPITATION'].sel(time=slice(start, end)).load() * INCH_TO_MM
        ds.close()
    except Exception as e:
        print(f"  [WARN] SNOTEL loading failed: {e}")
        return None, 0

    precip_mm = precip_mm.where(precip_mm >= 0)
    n_stations = int(precip_mm.notnull().any(dim='time').sum())
    daily_mean = precip_mm.mean(dim='site', skipna=True).to_series()
    daily_mean.index = pd.to_datetime(daily_mean.index).normalize()
    print(f"  ✓ SNOTEL: {n_stations} stations")
    return daily_mean, n_stations


def extract_event_window(event_date, products_to_extract=PRODUCTS):
    """Daily basin-mean precipitation for a 9-day window (T-3 to T+5) around an event.

    Gridded products come from common/prism_day.py and are averaged over the HUC8
    sub-basin mask (Upper Skagit, Sauk, Lower Skagit). Returns a DataFrame indexed
    by date string with one column per product.
    """
    window_dates = [event_date + pd.Timedelta(days=i)
                    for i in range(-WINDOW_DAYS_BEFORE, WINDOW_DAYS_AFTER + 1)]

    results = {f'{date.strftime("%Y-%m-%d")}': {} for date in window_dates}

    # Gridded products: daily basin means from the shared loader (common/prism_day.py),
    # on the PRISM day when ALIGN is True, on each product's own date labels otherwise.
    for product in products_to_extract:
        if product == 'snotel':
            continue
        da = prism_day.load_daily(product, window_dates, align=ALIGN)
        if da is None:
            print(f"  [WARN] {product}: no data for this window")
            continue
        daily = prism_day.basin_mean(da, prism_day.basin_mask(da))
        for date, val in daily.items():
            key = date.strftime("%Y-%m-%d")
            if key in results:
                results[key][product] = float(val)
        print(f"  {product}: {int(daily.notna().sum())}/{len(window_dates)} days")

    # Extract SNOTEL (mean of the stations inside the basin, not a gridded basin mean)
    n_snotel = 0
    if 'snotel' in products_to_extract:
        snotel_daily, n_snotel = load_snotel_daily_mean(window_dates)
        if snotel_daily is not None:
            for date in window_dates:
                normalized_date = date.normalize()
                if normalized_date in snotel_daily.index and not np.isnan(snotel_daily[normalized_date]):
                    results[f'{date.strftime("%Y-%m-%d")}']['snotel'] = float(snotel_daily[normalized_date])

    df = pd.DataFrame(results).T
    df.attrs['n_snotel'] = n_snotel
    return df


def plot_specific_non_ar_events():
    """Extract and plot specific non-AR events"""
    print("Loading events data...")
    events_df = pd.read_csv(EVENTS_CSV)
    events_df['date'] = pd.to_datetime(events_df['date'])

    print("Loading discharge data...")
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    # Convert specific dates to datetime
    event_dates = [pd.to_datetime(d) for d in SPECIFIC_DATES]

    print(f"Extracting and plotting {len(event_dates)} specific non-AR events...")

    # Extract data for all events and calculate max cumulative precipitation
    products = PRODUCTS
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
                       linestyle=PRODUCT_LINESTYLES[prod_idx],
                       marker=PRODUCT_MARKERS[prod_idx], linewidth=2, markersize=5,
                       color=PRODUCT_COLORS[prod_idx], alpha=0.8)

        ar_label = f'(AR = {ar_scale} | Discharge = {discharge_cms:.0f} cms)' if ar_scale is not None else ''
        ax.set_title(f'{event_date.strftime("%Y-%m-%d")} {ar_label}', fontsize=FONT_TITLE, fontweight='bold')
        if idx >= len(axes) - 2:
            ax.set_xlabel('Date', fontsize=FONT_LABEL)
        if idx % 2 == 0:
            ax.set_ylabel('Cumulative\nPrecipitation (mm)', fontsize=FONT_LABEL)
        ax.set_ylim(0, ylim_max)
        # Number of SNOTEL stations varies by event, so it is noted per panel
        if 'snotel' in event_data.columns:
            ax.text(0.02, 0.96, f"SNOTEL n = {event_data.attrs.get('n_snotel', 0)}",
                    transform=ax.transAxes, ha='left', va='top', fontsize=FONT_LEGEND)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Cumulative Precipitation during Top Discharge non-AR Events', fontsize=FONT_SUPTITLE, fontweight='bold')
    plt.tight_layout(h_pad=2.0)

    # Single shared legend below the panels (collected from all panels, since
    # some products are missing for some events, e.g. PNNL after 2020)
    legend_entries = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend_entries.setdefault(label, handle)
    fig.legend(legend_entries.values(), legend_entries.keys(), loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=len(legend_entries), fontsize=FONT_LEGEND,
               frameon=False)

    output_path = os.path.join(OUTPUT_DIR, 'specific_non_ar_events_cumulative_precipitation.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_specific_non_ar_events()
