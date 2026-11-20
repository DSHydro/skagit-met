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
    <VAULT_DIR>/conus404_daily/conus404_skagit_precip_daily_full.zarr
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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared daily loaders (common/prism_day.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import prism_day  # noqa: E402

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
EVENTS_CSV = BASE_DIR / "analysis" / "outputs" / "5day_taking_min_and_filling" / "4_clean_bias_table.csv"
HYDRO_Q_PATH = BASE_DIR / "data" / "usgs_12200500_discharge.rdb"
OUTPUT_DIR = Path(__file__).resolve().parent

# Products, in plotting order, with their line colors and markers
PRODUCTS = ['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']
PRODUCT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
PRODUCT_MARKERS = ['o', 's', '^', 'D', 'v', 'p']

# Day alignment of the products (GitHub issue #61): True = all products on the PRISM
# day (12 UTC on D-1 to 12 UTC on D); False = each product's own date labels (as before)
ALIGN = True

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


def extract_event_window(event_date, products_to_extract=PRODUCTS, start_date=None, end_date=None):
    """Daily basin-mean precipitation around an event for each product.

    The default window is 9 days (T-3 to T+5); ``start_date``/``end_date`` give a
    custom range instead. Gridded products come from common/prism_day.py and are
    averaged over the HUC8 sub-basin mask (Upper Skagit, Sauk, Lower Skagit).
    Returns a DataFrame indexed by date string with one column per product.
    """
    if start_date and end_date:
        # Custom date range
        window_dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq='D').tolist()
    else:
        # Default window (T-3 to T+5)
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
