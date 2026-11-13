"""
Multi-panel figure: for each row (one AR / non-AR date pair + one events CSV),
plot cumulative PRISM precipitation, cumulative streamflow, and the distribution
of AR scales across streamflow buckets.
"""

import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask
import rioxarray
from scipy.integrate import cumulative_trapezoid

VAULT_DIR = "/data0/skagit_met/data_transfer/data"
BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/skagit_met_for_publication"
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
HYDRO_Q_PATH = os.path.join(BASE_DIR, "data", "usgs_12200500_discharge.rdb")
PRISM_ROOT = os.path.join(VAULT_DIR, "prism_new_hq")

DRAINAGE_AREA_MI2 = 3093  # USGS 12200500
DRAINAGE_AREA_FT2 = DRAINAGE_AREA_MI2 * 27_878_400

OUTPUT_DIR = os.path.join(BASE_DIR, "plots/top_50_ar_vs_non-AR")
OUTPUTS_CSV_DIR = os.path.join(BASE_DIR, "analysis/outputs")
PRODUCT = "prism"

# ---- Edit these rows: each row is one line of panels in the figure ----
ROWS = [
    {
        "label": "(a)",
        "description": "No filtering",
        "ar_dates": ['1990-11-24', '2006-11-07', '2009-01-08', '1995-11-30', '2003-11-19', '2003-10-22', '2003-10-21', '2003-10-18', '1997-03-20', '1989-11-10', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '1986-02-25', '2007-12-04', '2014-11-28', '1995-02-20', '2005-01-20', '2002-02-22', '1982-01-24', '1998-11-16', '2011-01-17', '1982-02-14', '1998-12-30', '2004-12-11', '2017-11-23', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1986-11-20', '2022-12-27', '1985-10-27', '2001-11-15', '1981-02-17', '1984-01-05', '1999-11-14', '1994-12-20', '2003-03-14', '1997-01-20', '1983-11-16', '1984-11-03', '2006-01-12', '1998-11-26', '1998-12-28', '1984-01-25', '2018-11-04'],
        "non_ar_dates": ['2009-01-09', '1995-11-09', '1995-12-01', '2006-11-09', '2021-11-16', '1997-03-21', '2017-11-24', '2003-10-23', '1995-02-21', '2015-11-14', '1986-02-26', '2002-02-24', '1986-11-25', '2015-02-08', '1982-12-04', '1986-11-22', '2017-10-20', '2020-01-08', '1981-02-18', '1988-04-08', '2014-11-29', '1991-01-15', '1989-12-06', '2022-11-06', '2015-11-18', '2020-02-02', '1983-01-11', '1990-02-12', '2011-01-18', '2018-11-28', '2008-11-09', '2015-12-10', '1991-12-08', '2014-01-14', '1982-02-17', '2004-12-12', '1985-10-25', '2008-11-14', '1990-01-10', '2007-03-25', '2003-11-21', '1998-11-22', '1998-12-14', '1997-01-03', '2010-12-15', '2015-01-07', '2021-10-30', '2006-01-11', '1998-11-27', '2015-11-03'],
        "events_csv": os.path.join(OUTPUTS_CSV_DIR, "0day_taking_min_and_filling/3_multi_product_bulk_bias_data.csv"),
    },
    {
        "label": "(b)",
        "description": "Non-ARs ≥5 days from ARs",
        "ar_dates": ['1990-11-24', '2006-11-07', '2009-01-08', '1995-11-30', '2003-11-19', '2003-10-22', '2003-10-21', '2003-10-18', '1997-03-20', '1989-11-10', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '1986-02-25', '2007-12-04', '2014-11-28', '1995-02-20', '2005-01-20', '2002-02-22', '1982-01-24', '1998-11-16', '2011-01-17', '1982-02-14', '1998-12-30', '2004-12-11', '2017-11-23', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1986-11-20', '2022-12-27', '1985-10-27', '2001-11-15', '1981-02-17', '1984-01-05', '1999-11-14', '1994-12-20', '2003-03-14', '1997-01-20', '1983-11-16', '1984-11-03', '2006-01-12', '1998-11-26', '1998-12-28', '1984-01-25', '2018-11-04'],
        "non_ar_dates": ['2019-12-23', '1997-09-18', '2014-02-19', '2024-03-02', '1990-01-29', '2012-03-16', '2011-03-16', '2007-02-21', '1982-02-01', '2012-12-07', '1997-02-20', '1998-01-17', '1985-06-08', '1991-01-01', '2020-01-13', '2024-06-05', '2007-12-25', '1998-12-02', '1999-01-20', '1991-11-26', '2017-03-19', '2023-12-11', '2011-10-12', '2011-11-14', '2011-04-06', '1982-01-16', '2004-09-17', '2013-11-19', '1997-04-22', '2021-12-25', '2000-01-10', '1984-11-30', '1988-03-09', '2006-04-16', '2001-11-01', '2003-02-22', '1999-03-02', '1981-04-11', '1985-11-08', '1996-12-22', '2019-01-24', '1994-02-18', '1990-08-31', '1995-01-19', '2018-04-18', '2001-12-04', '2009-12-22', '2001-02-05', '1993-11-22', '2024-12-30'],
        "events_csv": os.path.join(OUTPUTS_CSV_DIR, "5day_taking_min_and_filling/3_multi_product_bulk_bias_data.csv"),
    },
]

COLOR_AR = '#1f77b4'
COLOR_NON_AR = '#ff7f0e'
SF_LABELS = ["< 600", "600 - 1200", "1200 - 1700", "> 1700"]
SF_BINS = [0, 600, 1200, 1700, float('inf')]
CFS_PER_CMS = 35.3147

# Publication font sizes (figure is 15 in wide; expect ~0.5x reduction to journal page width)
FONT_PANEL_LABEL = 20
FONT_TITLE = 15
FONT_LABEL = 14
FONT_TICK = 13
FONT_LEGEND = 12


# ---------------- Data loading (copied from ar_vs_non_ar_comparison.py) ----------------

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
    """Daily basin-mean PRISM precipitation for T-3 to T+5."""
    window_dates = [event_date + pd.Timedelta(days=i) for i in range(-3, 6)]
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
    discharge_mmd = discharge_df['discharge_cfs'] / DRAINAGE_AREA_FT2 * 60 * 60 * 24 * 304.8
    dates = pd.to_datetime(discharge_df['date'].values)
    days = (dates - dates[0]).total_seconds() / (60 * 60 * 24)
    cum = cumulative_trapezoid(discharge_mmd.values, x=days.values, initial=0)
    return pd.Series(cum, index=discharge_df.index)


def compute_cumulative_discharge_stats(event_dates, q_df):
    """Median and min-max cumulative discharge for T-3 to T+8, indexed by relative days."""
    cumulative_series = []
    for event_date_str in event_dates:
        event_date = pd.to_datetime(event_date_str)
        window = q_df[(q_df['date'] >= event_date - pd.Timedelta(days=3)) &
                      (q_df['date'] <= event_date + pd.Timedelta(days=8))]
        if not window.empty:
            cum = calculate_cumulative_discharge_mm(window)
            delta = window['date'] - event_date
            cum.index = delta.dt.days + delta.dt.seconds / (24 * 3600)
            cumulative_series.append(cum)
    if not cumulative_series:
        return None, None, None
    combined = pd.concat(cumulative_series, axis=1).sort_index()
    return combined.median(axis=1), combined.min(axis=1), combined.max(axis=1)


# ---------------- Plotting ----------------


def plot_band(ax, median, lo, hi, color, marker, name):
    if median is None:
        return
    ax.fill_between(median.index, lo, hi, alpha=0.25, color=color, label=f'{name} (min-max)')
    ax.plot(median.index, median, linewidth=2.5, color=color, marker=marker,
            markersize=6, label=f'{name} (median)', alpha=0.9)


def ar_scale_by_streamflow_bucket(events_csv):
    events = pd.read_csv(events_csv)
    events['discharge_cms'] = events['discharge_cfs'] / CFS_PER_CMS
    events['streamflow_bucket'] = pd.cut(events['discharge_cms'], bins=SF_BINS, labels=SF_LABELS, right=False)
    counts = events.groupby(['streamflow_bucket', 'ar_scale'], observed=False).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def plot_bucket_bars(ax, pct):
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
    ax.set_xticklabels(SF_LABELS)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_xlabel('Streamflow Bucket (cms)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(title='AR Scale', fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND,
              loc='upper left', bbox_to_anchor=(1, 1))


def load_or_extract(dates, csv_path):
    """Read event precipitation from csv_path if it exists; otherwise extract it and save it there.
    Delete the CSV if you change the dates."""
    if os.path.exists(csv_path):
        print(f"  Loading saved data: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['date'])
        return {ev: g.set_index('date')[[PRODUCT]] for ev, g in df.groupby('event_date')}
    data = extract_all_events(dates)
    pd.concat(data, names=['event_date', 'date']).reset_index().to_csv(csv_path, index=False)
    print(f"  Saved data: {csv_path}")
    return data


def main():
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    fig, axes = plt.subplots(len(ROWS), 3, figsize=(15, 5 * len(ROWS)),
                             gridspec_kw={'width_ratios': [1, 1, 1.1], 'wspace': 0.18, 'hspace': 0.4})
    axes = np.atleast_2d(axes)

    for r, row in enumerate(ROWS):
        ax_p, ax_q, ax_b = axes[r]
        print(f"Row {row['label']}: extracting {len(row['ar_dates'])} AR / {len(row['non_ar_dates'])} non-AR events...")
        tag = row['label'].strip('()')
        ar_data = load_or_extract(row['ar_dates'], os.path.join(OUTPUT_DIR, f"panel_data_row_{tag}_ar_{PRODUCT}.csv"))
        non_ar_data = load_or_extract(row['non_ar_dates'], os.path.join(OUTPUT_DIR, f"panel_data_row_{tag}_non_ar_{PRODUCT}.csv"))

        plot_band(ax_p, *compute_cumulative_stats(non_ar_data, PRODUCT), COLOR_NON_AR, 'o', 'Non-AR')
        plot_band(ax_p, *compute_cumulative_stats(ar_data, PRODUCT), COLOR_AR, 's', 'AR')
        plot_band(ax_q, *compute_cumulative_discharge_stats(row['non_ar_dates'], q_df), COLOR_NON_AR, 'o', 'Non-AR')
        plot_band(ax_q, *compute_cumulative_discharge_stats(row['ar_dates'], q_df), COLOR_AR, 's', 'AR')

        ax_q.sharey(ax_p)
        ax_q.tick_params(labelleft=False)
        for ax, ylabel in [(ax_p, 'Cumulative Precipitation (mm)'), (ax_q, 'Cumulative Discharge (mm)')]:
            ax.set_xlabel('Days Relative to Event Date', fontsize=FONT_LABEL, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=FONT_LABEL, fontweight='bold')
            ax.legend(loc='upper left', fontsize=FONT_LEGEND, framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', labelsize=FONT_TICK)

        plot_bucket_bars(ax_b, ar_scale_by_streamflow_bucket(row['events_csv']))
        # Top row sits above the column titles; other rows sit just above their axes
        label_y = 1.22 if r == 0 else 1.04
        ax_p.text(-0.18, label_y, f"{row['label']} {row['description']}", transform=ax_p.transAxes,
                  fontsize=FONT_PANEL_LABEL, fontweight='bold', va='bottom')

    axes[0, 0].set_title(f'Cumulative Precipitation\n({PRODUCT.upper()}, top 50)', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    axes[0, 1].set_title('Cumulative Streamflow\n(top 50)', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    axes[0, 2].set_title('AR Scale by Streamflow Bucket\n(all events, WY 1982–2024)',
                         fontsize=FONT_TITLE, fontweight='bold', pad=10)

    output_path = os.path.join(OUTPUT_DIR, f'ar_vs_non_ar_panel_{PRODUCT}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
