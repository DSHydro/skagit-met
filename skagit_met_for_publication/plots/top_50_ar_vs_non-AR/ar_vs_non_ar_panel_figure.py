"""AR vs non-AR events: cumulative precipitation, streamflow and AR-scale distribution.

Publication panel figure with one row per event sample and three columns:

Rows (each uses its own events CSV)
    (a) No filtering: all non-AR events (0day_taking_min_and_filling)
    (b) Non-AR events at least 5 days away from AR events (5day_taking_min_and_filling)

Columns
    1. Cumulative PRISM basin-mean precipitation, T-3 to T+5 days around the
       event date: median and min-max band of the top 50 AR events vs the top
       50 non-AR events.
    2. Cumulative USGS 12200500 discharge (as mm over the drainage area),
       T-3 to T+8 days, for the same events.
    3. Percentage of each AR scale (0-5) within each streamflow bucket, using
       ALL events of the row's events CSV in water years 1982-2024
       (1 Oct 1981 - 30 Sep 2024).

The top 50 events of each type are those with the largest PRISM 3-day total
(``prism_3d_tot``: event day and the previous 2 days) over the whole record
of the row's events CSV (Jan 1981 - Dec 2024); this selection was previously
done in comparison.ipynb. The daily PRISM values of the selected
events are cached in panel_data_row_<row>_<ar|non_ar>_prism.csv; a cache is
reused only while it holds exactly the selected events, otherwise it is
rebuilt from the PRISM files.

Input
    /data0/skagit_met/data_transfer/data/prism_new_hq/...   (daily PRISM, outside this repository)
    data/GIS/SkagitSubBasin_HUC8.geojson
    data/usgs_12200500_discharge.rdb
    analysis/outputs/0day_taking_min_and_filling/3_multi_product_bulk_bias_data.csv
    analysis/outputs/5day_taking_min_and_filling/3_multi_product_bulk_bias_data.csv
Output
    plots/top_50_ar_vs_non-AR/ar_vs_non_ar_panel_prism.png
    plots/top_50_ar_vs_non-AR/panel_data_row_<a|b>_<ar|non_ar>_prism.csv   (cache)

Usage
    python ar_vs_non_ar_panel_figure.py
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
from scipy.integrate import cumulative_trapezoid

# Shared raw-data archive (outside this repository)
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PRISM_ROOT = os.path.join(VAULT_DIR, "prism_new_hq")

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
HUC8_GEO = BASE_DIR / "data" / "GIS" / "SkagitSubBasin_HUC8.geojson"
HYDRO_Q_PATH = BASE_DIR / "data" / "usgs_12200500_discharge.rdb"
OUTPUTS_CSV_DIR = BASE_DIR / "analysis" / "outputs"
OUTPUT_DIR = Path(__file__).resolve().parent

DRAINAGE_AREA_MI2 = 3093  # USGS 12200500
DRAINAGE_AREA_FT2 = DRAINAGE_AREA_MI2 * 27_878_400
SECONDS_PER_DAY = 60 * 60 * 24
MM_PER_FT = 304.8

PRODUCT = "prism"

# Event windows, in days relative to the event date (inclusive)
PRECIP_WINDOW_DAYS = (-3, 5)
DISCHARGE_WINDOW_DAYS = (-3, 8)

# Study period in water years (WY N = 1 Oct of year N-1 to 30 Sep of year N),
# used for column 3 only. The top-N selection (columns 1-2) uses the whole record.
WY_START, WY_END = 1982, 2024

# Event selection (formerly comparison.ipynb): for each row, the TOP_N AR events
# (ar_scale > 0) and the TOP_N non-AR events (ar_scale == 0) of that row's events
# CSV, ranked by RANK_COLUMN (PRISM 3-day total: event day and the previous 2 days).
TOP_N = 50
RANK_COLUMN = "prism_3d_tot"

# ---- Edit these rows: each row is one line of panels in the figure ----
ROWS = [
    {
        "label": "(a)",
        "description": "No filtering",
        "events_csv": os.path.join(OUTPUTS_CSV_DIR, "0day_taking_min_and_filling/3_multi_product_bulk_bias_data.csv"),
    },
    {
        "label": "(b)",
        "description": "Non-ARs ≥5 days from ARs",
        "events_csv": os.path.join(OUTPUTS_CSV_DIR, "5day_taking_min_and_filling/3_multi_product_bulk_bias_data.csv"),
    },
]

COLOR_AR = '#1f77b4'
COLOR_NON_AR = '#ff7f0e'
MARKER_AR = 's'
MARKER_NON_AR = 'o'

# Streamflow buckets (cms), left-closed: [0, 600), [600, 1200), ...
SF_LABELS = ["< 600", "600 - 1200", "1200 - 1700", "> 1700"]
SF_BINS = [0, 600, 1200, 1700, float('inf')]
CFS_PER_CMS = 35.3147

# Figure settings
PANEL_ROW_SIZE = (18, 5)  # width, height per row (inches); wide enough for the column-3 tick labels
COLUMN_SPACING = 0.28  # horizontal space between columns (fraction of the mean axes width)
COLUMN_WIDTHS = [1, 1, 1.5]  # relative widths; column 3 is wider so its bucket labels fit
ROW_SPACING = 0.55  # vertical space between rows (fraction of the mean axes height), room for the row labels
DPI = 300

# Publication font sizes (figure is 18 in wide; expect ~0.4x reduction to journal page width).
# Scaled 1.2x from the 15-in version so text prints at the same size after reduction.
FONT_PANEL_LABEL = 24
FONT_TITLE = 18
FONT_LABEL = 17
FONT_TICK = 16
FONT_LEGEND = 14


# ---------------- Event selection and data loading ----------------

def load_events(events_csv):
    """Read an events CSV and keep only events in water years WY_START-WY_END."""
    events = pd.read_csv(events_csv)
    dates = pd.to_datetime(events['date'])
    water_year = dates.dt.year + (dates.dt.month >= 10)  # Oct-Dec belong to the next WY
    return events[water_year.between(WY_START, WY_END)].copy()


def select_top_events(events_csv, n=TOP_N, rank_column=RANK_COLUMN):
    """Return (ar_dates, non_ar_dates): the n largest events by rank_column for each type (whole record)."""
    events = pd.read_csv(events_csv)
    ar_dates = events[events['ar_scale'] > 0].nlargest(n, rank_column)['date'].tolist()
    non_ar_dates = events[events['ar_scale'] == 0].nlargest(n, rank_column)['date'].tolist()
    return ar_dates, non_ar_dates


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
    zip_path = os.path.join(prism_root, get_decade_folder(date.year), f"prism_ppt_us_25m_{date_str}.zip")
    if not os.path.exists(zip_path):
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith('.tif'):
                        da = rioxarray.open_rasterio(os.path.join(root, f))
                        return da.squeeze().copy(deep=True)
    except Exception:
        pass
    return None


def load_usgs_rdb(filepath, param_name):
    """Load USGS RDB file"""
    try:
        df = pd.read_csv(filepath, sep='\t', comment='#')
        df = df.iloc[1:].reset_index(drop=True)
        val_cols = [c for c in df.columns if '_00' in c and not c.endswith('_cd')]
        if not val_cols:
            return pd.DataFrame(columns=['date', param_name])
        df = df[['datetime', val_cols[0]]]
        df.columns = ['date', param_name]
        df['date'] = pd.to_datetime(df['date'])
        df[param_name] = pd.to_numeric(df[param_name], errors='coerce')
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame(columns=['date', param_name])


def load_regions():
    return gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")


def get_mask(gdf, lon, lat):
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = [gdf[gdf["Name"].str.contains(r, case=False)].index[0]
                      for r in ["Upper Skagit", "Sauk", "Lower Skagit"]]
    return mask.sel(region=region_indices)


def calculate_basin_mean(da, mask_2d):
    """Basin mean of a 2D grid; negative values are treated as nodata."""
    data = da.values.astype(float)
    data[data < 0] = np.nan
    mask_idx = mask_2d.values.flatten() > 0
    if not mask_idx.any():
        return np.nan
    return np.nanmean(data.flatten()[mask_idx])


def extract_event_window(event_date, gdf):
    """Daily basin-mean PRISM precipitation over PRECIP_WINDOW_DAYS (T-3 to T+5)."""
    first, last = PRECIP_WINDOW_DAYS
    window_dates = [event_date + pd.Timedelta(days=i) for i in range(first, last + 1)]
    values = {}
    for date in window_dates:
        try:
            da = load_prism_from_new_format(date, PRISM_ROOT)
            if da is None:
                continue
            m_prism = get_mask(gdf, da.x.values, da.y.values).any(dim='region')
            values[date.normalize()] = float(calculate_basin_mean(da, m_prism))
        except Exception:
            pass
    return pd.DataFrame({PRODUCT: pd.Series(values, dtype=float)})


def extract_all_events(event_dates):
    """Extract the PRISM window of every event; returns {event_date_str: DataFrame}."""
    gdf = load_regions()
    all_data = {}
    for event_date_str in event_dates:
        try:
            window_df = extract_event_window(pd.to_datetime(event_date_str), gdf)
            if len(window_df) > 0:
                all_data[event_date_str] = window_df
        except Exception as e:
            print(f"  Error extracting {event_date_str}: {e}")
    return all_data


def compute_cumulative_stats(all_event_data, product=PRODUCT):
    """Median and min-max cumulative precipitation, indexed by days relative to event date."""
    cumulative_series = []
    for event_date_str, window_df in all_event_data.items():
        if product in window_df.columns:
            window_df = window_df.sort_index()
            cumsum = window_df[product].cumsum()
            cumsum.index = (window_df.index - pd.to_datetime(event_date_str)).days
            cumulative_series.append(cumsum)
    if not cumulative_series:
        return None, None, None
    combined = pd.concat(cumulative_series, axis=1).sort_index()
    return combined.median(axis=1), combined.min(axis=1), combined.max(axis=1)


def calculate_cumulative_discharge_mm(discharge_df):
    """Convert discharge in cfs to cumulative mm using trapezoidal integration."""
    if discharge_df.empty:
        return pd.Series(dtype=float)
    # ft3/s spread over the drainage area -> ft/s -> mm/day
    discharge_mmd = discharge_df['discharge_cfs'] / DRAINAGE_AREA_FT2 * SECONDS_PER_DAY * MM_PER_FT
    dates = pd.to_datetime(discharge_df['date'].values)
    days = (dates - dates[0]).total_seconds() / SECONDS_PER_DAY
    cum = cumulative_trapezoid(discharge_mmd.values, x=days.values, initial=0)
    return pd.Series(cum, index=discharge_df.index)


def compute_cumulative_discharge_stats(event_dates, q_df):
    """Median and min-max cumulative discharge over DISCHARGE_WINDOW_DAYS (T-3 to T+8), indexed by relative days."""
    first, last = DISCHARGE_WINDOW_DAYS
    cumulative_series = []
    for event_date_str in event_dates:
        event_date = pd.to_datetime(event_date_str)
        window = q_df[(q_df['date'] >= event_date + pd.Timedelta(days=first)) &
                      (q_df['date'] <= event_date + pd.Timedelta(days=last))]
        if not window.empty:
            cum = calculate_cumulative_discharge_mm(window)
            delta = window['date'] - event_date
            cum.index = delta.dt.days + delta.dt.seconds / SECONDS_PER_DAY
            cumulative_series.append(cum)
    if not cumulative_series:
        return None, None, None
    combined = pd.concat(cumulative_series, axis=1).sort_index()
    return combined.median(axis=1), combined.min(axis=1), combined.max(axis=1)


# ---------------- Plotting ----------------


def plot_band(ax, median, lo, hi, color, marker, name):
    """Plot a median line with a shaded min-max band (skipped if there is no data)."""
    if median is None:
        return
    ax.fill_between(median.index, lo, hi, alpha=0.25, color=color, label=f'{name} (min-max)')
    ax.plot(median.index, median, linewidth=2.5, color=color, marker=marker,
            markersize=6, label=f'{name} (median)', alpha=0.9)


def ar_scale_by_streamflow_bucket(events_csv):
    """Percentage of each AR scale within each streamflow bucket, over all events in events_csv (WY_START-WY_END)."""
    events = load_events(events_csv)
    events['discharge_cms'] = events['discharge_cfs'] / CFS_PER_CMS
    events['streamflow_bucket'] = pd.cut(events['discharge_cms'], bins=SF_BINS, labels=SF_LABELS, right=False)
    counts = events.groupby(['streamflow_bucket', 'ar_scale'], observed=False).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def plot_bucket_bars(ax, pct):
    """Stacked bars of AR-scale percentages per streamflow bucket (reversed inferno colors)."""
    cmap = plt.cm.inferno
    scales = sorted(pct.columns)
    n = len(scales)
    x = np.arange(len(SF_LABELS))
    bottom = np.zeros(len(SF_LABELS))
    for i, scale in enumerate(scales):
        vals = pct[scale].reindex(SF_LABELS).fillna(0).values
        ax.bar(x, vals, 0.6, bottom=bottom, label=f'Scale {int(scale)}',
               color=cmap(0.95 - (i / (n - 1)) * 0.85), edgecolor='white', linewidth=1.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(' - ', '–') for label in SF_LABELS])  # "600–1200": narrower
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_xlabel('Streamflow Bucket (cms)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title='AR Scale', fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
              loc='upper left', bbox_to_anchor=(1, 1))


def load_or_extract(dates, csv_path):
    """Return event precipitation for `dates`, using csv_path as a cache.

    The cache is used only if it holds exactly the requested event dates;
    otherwise (missing file or different events) the data are extracted from
    the PRISM files again and the cache is overwritten.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'])
        if set(df['event_date']) == set(dates):
            print(f"  Loading saved data: {csv_path}")
            return {ev: g.set_index('date')[[PRODUCT]] for ev, g in df.groupby('event_date')}
        print(f"  Saved data does not match the selected events, extracting again: {csv_path}")
    data = extract_all_events(dates)
    pd.concat(data, names=['event_date', 'date']).reset_index().to_csv(csv_path, index=False)
    print(f"  Saved data: {csv_path}")
    return data


def main():
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    fig, axes = plt.subplots(len(ROWS), 3, figsize=(PANEL_ROW_SIZE[0], PANEL_ROW_SIZE[1] * len(ROWS)),
                             gridspec_kw={'width_ratios': COLUMN_WIDTHS, 'wspace': COLUMN_SPACING, 'hspace': ROW_SPACING})
    axes = np.atleast_2d(axes)

    for r, row in enumerate(ROWS):
        ax_p, ax_q, ax_b = axes[r]
        ar_dates, non_ar_dates = select_top_events(row['events_csv'])
        print(f"Row {row['label']}: extracting {len(ar_dates)} AR / {len(non_ar_dates)} non-AR events...")
        tag = row['label'].strip('()')
        ar_data = load_or_extract(ar_dates, os.path.join(OUTPUT_DIR, f"panel_data_row_{tag}_ar_{PRODUCT}.csv"))
        non_ar_data = load_or_extract(non_ar_dates, os.path.join(OUTPUT_DIR, f"panel_data_row_{tag}_non_ar_{PRODUCT}.csv"))

        plot_band(ax_p, *compute_cumulative_stats(non_ar_data, PRODUCT), COLOR_NON_AR, MARKER_NON_AR, 'Non-AR')
        plot_band(ax_p, *compute_cumulative_stats(ar_data, PRODUCT), COLOR_AR, MARKER_AR, 'AR')
        plot_band(ax_q, *compute_cumulative_discharge_stats(non_ar_dates, q_df), COLOR_NON_AR, MARKER_NON_AR, 'Non-AR')
        plot_band(ax_q, *compute_cumulative_discharge_stats(ar_dates, q_df), COLOR_AR, MARKER_AR, 'AR')

        ax_q.sharey(ax_p)
        ax_q.tick_params(labelleft=False)
        for ax, ylabel in [(ax_p, 'Cumulative\nPrecipitation (mm)'), (ax_q, 'Cumulative\nDischarge (mm)')]:
            ax.set_xlabel('Days Relative to Event Date', fontsize=FONT_LABEL, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight='bold')
            ax.legend(loc='upper left', fontsize=FONT_LEGEND, framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', labelsize=FONT_TICK)

        plot_bucket_bars(ax_b, ar_scale_by_streamflow_bucket(row['events_csv']))
        # Top row sits above the column titles; other rows sit just above their axes
        label_y = 1.27 if r == 0 else 1.10
        ax_p.text(-0.18, label_y, f"{row['label']} {row['description']}", transform=ax_p.transAxes,
                  fontsize=FONT_PANEL_LABEL, fontweight='bold', va='bottom')

    axes[0, 0].set_title(f'Cumulative Precipitation\n({PRODUCT.upper()}, top 50)', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    axes[0, 1].set_title('Cumulative Streamflow\n(top 50)', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    axes[0, 2].set_title(f'AR Scale by Streamflow Bucket\n(all events, WY {WY_START}–{WY_END})',
                         fontsize=FONT_TITLE, fontweight='bold', pad=10)

    output_path = os.path.join(OUTPUT_DIR, f'ar_vs_non_ar_panel_{PRODUCT}.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
