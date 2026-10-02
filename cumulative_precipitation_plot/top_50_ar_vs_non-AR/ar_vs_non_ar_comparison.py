"""
Compare cumulative precipitation and streamflow between AR and non-AR events.

Creates side-by-side plots showing median cumulative precipitation and cumulative
discharge with 25%-75% quantile ranges shaded for both AR and non-AR event groups.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Add parent directory to path to import plot_specific_ar_events
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_specific_ar_and_non_ar_events.plot_specific_ar_events import extract_event_window, load_usgs_rdb

# Hydrology data
HYDRO_BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
HYDRO_EXP_DATA_DIR = os.path.join(HYDRO_BASE_DIR, "experiments_2/data")
HYDRO_Q_PATH = os.path.join(HYDRO_EXP_DATA_DIR, "usgs_12200500_discharge.rdb")

# Drainage area (square miles)
DRAINAGE_AREA_MI2 = 3093 #mi^2 extracted from the USGS website
DRAINAGE_AREA_FT2 = DRAINAGE_AREA_MI2 * 27_878_400

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit-met"
OUTPUT_DIR = os.path.join(BASE_DIR, "cumulative_precipitation_plot/top_50_ar_vs_non-AR/cumulative_precipitation_plot")

# Event dates from the comparison notebook
# NON_AR_DATES = [
#     '2021-11-16', '2010-12-14', '1995-02-21', '2015-11-14', '2020-01-08',
#     '1988-04-08', '1990-10-06', '2011-01-18', '2020-01-09', '2011-11-25',
#     '2014-01-14', '2004-12-12', '2017-12-20', '1995-11-26', '2020-02-03',
#     '2014-03-06', '1994-10-29', '1996-11-29', '1991-12-07', '2014-02-19',
#     '2014-03-17', '1995-12-02', '1999-12-13', '1997-10-06', '2007-03-13',
#     '2014-03-04', '1996-11-28', '2016-10-10', '2021-02-23', '2020-02-08',
#     '2024-03-02', '2001-12-19', '2005-01-21', '1997-02-01', '2018-01-30',
#     '2012-10-21', '2014-10-23', '1981-02-20', '2011-11-28', '1990-11-14',
#     '1997-01-31', '2020-09-27', '2006-01-08', '2016-10-11', '2008-11-15',
#     '1982-12-18', '1993-03-24', '2022-11-01', '2007-03-14', '2024-01-11'
# ]

# NON_AR_DATES = ['2008-07-02', '2008-07-03', '2008-07-04', '2012-07-10', '2012-07-15', '2020-06-01', '2022-07-04', '2022-07-02', '2012-07-18', '1997-05-13', '1986-06-01', '2018-05-16', '1999-08-05', '1999-07-11', '1986-06-02', '1981-01-02', '2008-07-07', '1999-07-22', '1997-06-24', '2011-07-17', '2014-05-16', '1987-05-01', '2011-07-19', '2011-07-09', '1997-06-06', '1999-07-21', '1985-05-24', '2011-07-20', '2011-07-23', '2018-05-24', '2023-05-05', '2018-05-09', '2011-07-18', '2023-05-15', '2017-05-23', '1982-07-04', '1999-07-08', '2006-05-17', '2014-05-04', '1982-05-26', '1986-06-03', '2011-06-14', '2002-07-09', '1982-06-12', '1991-07-25', '2017-12-02', '1985-06-09', '2008-06-01', '1999-07-30', '2022-06-30'] #5d window and 2000 non-AR sample
# NON_AR_DATES = ['2008-07-04', '2012-07-14', '2021-06-30', '1982-06-18', '2002-06-15', '2012-07-16', '2021-07-01', '2002-06-16', '2012-07-17', '1999-06-17', '2002-06-14', '2006-05-18', '2011-07-03', '2012-06-22', '1986-06-01', '2002-07-13', '2002-07-12', '2011-06-07', '2012-07-05', '1999-07-12', '1999-06-18', '2012-05-15', '2018-05-10', '2022-07-05', '2017-06-01', '1991-01-16', '2012-07-07', '1997-06-06', '1982-06-15', '1999-07-25', '2011-07-23', '2002-06-18', '1985-05-25', '2014-05-25', '1982-07-04', '2021-06-05', '1999-07-26', '1986-06-03', '2011-06-14', '1997-06-25', '2012-05-01', '1990-07-12', '2011-07-16', '1999-07-17', '1990-06-11', '2008-07-08', '2017-12-03', '1997-06-26', '2011-06-06', '1999-07-18'] #3d window and 200 non-AR sample
# NON_AR_DATES = ['2021-11-19', '2008-07-02', '2020-05-31', '1995-12-04', '1982-06-21', '1990-11-17', '2012-07-09', '2021-06-30', '1985-06-08', '1997-05-16', '2012-06-24', '2002-06-15', '2017-05-30', '2006-05-19', '2021-06-03', '2011-07-07', '2022-07-03', '1999-06-17', '1997-06-05', '2012-05-16', '2021-12-06', '1986-06-01', '1991-07-04', '1999-05-25', '2002-07-13', '2023-05-17', '2018-05-16', '1999-08-05', '2011-06-07', '2007-07-07', '2014-06-29', '1997-06-24', '1991-01-16', '2012-05-25', '1991-02-12', '1987-05-01', '2025-12-23', '1997-04-21', '1982-07-15', '1984-01-09', '2013-06-21', '2009-06-03', '1990-07-13', '2012-06-06', '2022-06-14', '2013-11-19', '2000-06-29', '1991-07-25', '2017-12-02', '1989-06-06'] #new filtering (keep contigous dates as one event and assign the date with peak discharge) - peak discharge
# NON_AR_DATES = ['1991-03-04', '2012-12-18', '1992-12-22', '2007-02-21', '1983-03-31', '1997-02-20', '2020-03-31', '2018-04-16', '1985-06-08', '1982-03-13'] #, top 10: new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
# NON_AR_DATES = ['1991-03-04', '2012-12-18', '1992-12-22', '2007-02-21', '1983-03-31', '1997-02-20', '2020-03-31', '2018-04-16', '1985-06-08', '1982-03-13', '1990-01-27', '1994-03-19', '1986-02-17', '2021-12-24', '1991-01-01', '1991-11-27', '2010-01-02', '2018-02-18', '1988-03-07', '2011-11-14', '1998-01-07', '2004-09-18', '1982-11-18', '2013-11-19', '1997-04-22', '1983-11-27', '2008-02-01', '2006-04-15', '1988-11-12', '1990-03-10', '1981-04-10', '2000-01-10', '2002-03-21', '2001-05-01', '2015-12-21', '2009-04-14', '2003-02-22', '2000-03-19', '1992-06-14', '2008-12-29', '1984-03-14', '1984-12-23', '2013-03-21', '1990-09-01', '2016-12-30', '2014-02-25', '1996-12-22', '2023-02-08', '1985-04-01', '2016-12-12'] #top 50: new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
NON_AR_DATES = ['1991-03-04', '2014-03-04', '1990-01-29', '2012-12-18', '1991-01-16', '1992-12-22', '2007-02-21', '1983-03-31', '1997-02-20', '2020-03-31', '2018-04-16', '2010-01-03', '1985-06-08', '1982-03-13', '1994-03-19', '1986-02-17', '2021-12-24', '1991-01-01', '2020-01-13', '1991-11-27', '2018-02-18', '2017-03-04', '1988-03-07', '2011-11-14', '1998-01-07', '1997-04-21', '2004-09-16', '2006-11-24', '1982-11-18', '2013-11-19', '1983-11-27', '2012-02-27', '1991-12-13', '2008-02-01', '2006-04-15', '1988-11-12', '1990-03-10', '1981-04-10', '2000-01-10', '2002-03-21', '2001-05-01', '2015-12-21', '2009-01-02', '2009-04-14', '1984-05-12', '2003-02-22', '2000-03-19', '1992-06-14', '2013-02-24', '1984-03-14'] #top 50: new filtering (first, 3 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip


# AR_DATES = [
#     '1990-11-24', '2006-11-07', '2009-01-08', '1990-11-25', '1990-11-11',
#     '1995-11-30', '1990-11-10', '2003-11-19', '1995-11-10', '2003-10-22',
#     '1989-11-11', '2003-10-21', '2003-10-18', '1982-02-15', '1997-03-20',
#     '1989-11-10', '1995-11-29', '1986-01-19', '1983-01-10', '1995-11-08',
#     '1982-02-16', '2006-11-06', '2021-11-15', '1986-11-24', '1988-10-16',
#     '2005-01-19', '2003-10-17', '1986-02-25', '2003-10-23', '2007-12-04',
#     '1988-10-17', '1990-11-12', '2014-11-28', '2003-10-19', '1985-11-03',
#     '1998-11-15', '1989-11-09', '1995-02-20', '1995-02-19', '2009-01-10',
#     '1994-12-01', '2005-01-20', '2002-02-22', '1982-01-25', '2006-11-05',
#     '1982-01-24', '1998-11-16', '2009-01-07', '2011-01-17', '2010-12-13'
# ]

# AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-25', '1990-11-11', '1995-11-30', '1990-11-10', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21', '2003-10-18', '1982-02-15', '1997-03-20', '1989-11-10', '1995-11-29', '1986-01-19', '1983-01-10', '1995-11-08', '1982-02-16', '2006-11-06', '2021-11-15', '1986-11-24', '1988-10-16', '2005-01-19', '2003-10-17', '1986-02-25', '2003-10-23', '2007-12-04', '1988-10-17', '1990-11-12', '2014-11-28', '2003-10-19', '1985-11-03', '1998-11-15', '1989-11-09', '1995-02-20', '1995-02-19', '2009-01-10', '1994-12-01', '2005-01-20', '2002-02-22', '1982-01-25', '2006-11-05', '1982-01-24', '1998-11-16', '2009-01-07', '2011-01-17', '2010-12-13']
# AR_DATES = ['1990-11-25', '1995-11-30', '1990-11-11', '2006-11-07', '2003-10-22', '2025-12-12', '2003-10-21', '2021-11-15', '1989-11-11', '2010-12-13', '1984-01-05', '2003-10-23', '2025-12-17', '2017-11-23', '1999-11-13', '2011-01-17', '1999-11-14', '2007-03-25', '1997-03-20', '2002-01-08', '2015-11-18', '2009-01-08', '2014-11-28', '2007-12-04', '2021-11-29', '2004-12-11', '2009-11-17', '1986-11-24', '2003-11-19', '2021-10-29', '2015-12-09', '2021-12-01', '2001-11-15', '1986-01-19', '2005-01-19', '1997-07-09', '1989-11-12', '1986-02-25', '2008-05-18', '2007-03-12', '2008-11-13', '2015-02-07', '2016-02-16', '2009-11-18', '1983-11-16', '1995-11-12', '2020-02-01', '1985-11-03', '1999-11-15', '2017-11-25'] #new filtering (keep contigous dates as one event and assign the date with peak discharge) - preak discharge
# AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-11', '1995-11-30', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21'] #top 10: new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
# AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-11', '1995-11-30', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21', '1982-02-15', '1997-03-20', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '2005-01-19', '1986-02-25', '2003-10-23', '2007-12-04', '2014-11-28', '1985-11-03', '1998-11-15', '1995-02-20', '2009-01-10', '1994-12-01', '2002-02-22', '1982-01-25', '1982-01-24', '2011-01-17', '2010-12-13', '1997-10-31', '1998-12-30', '2004-12-11', '1986-11-25', '2017-11-23', '2014-01-13', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1990-10-05', '1996-02-08', '1986-11-20', '1999-11-13', '2022-12-27', '1985-10-27', '1981-02-18'] #top 50: new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip
AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-11', '1995-11-30', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21', '1982-02-15', '1997-03-20', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '2005-01-19', '1986-02-25', '2003-10-23', '2007-12-04', '2014-11-28', '1985-11-03', '1998-11-15', '1995-02-20', '2009-01-10', '1994-12-01', '2002-02-22', '1982-01-25', '1982-01-24', '2011-01-17', '2010-12-13', '1997-10-31', '1998-12-30', '2004-12-11', '1986-11-25', '2017-11-23', '2014-01-13', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1990-10-05', '1996-02-08', '1986-11-20', '1999-11-13', '2022-12-27', '1985-10-27', '1981-02-18'] #top 50: new filtering (first, 3 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip


def extract_all_events(event_dates, products=['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']):
    """Extract cumulative precipitation for all events in a list."""
    all_data = {}

    for event_date_str in event_dates:
        event_date = pd.to_datetime(event_date_str)
        try:
            window_df = extract_event_window(event_date, products_to_extract=products)
            if window_df is not None and len(window_df) > 0:
                all_data[event_date_str] = window_df
        except Exception as e:
            print(f"  Error extracting {event_date_str}: {e}")

    return all_data


def compute_cumulative_stats(all_event_data, product='prism'):
    """Compute median and min-max cumulative precipitation across events.

    Returns statistics indexed by relative days from event date (day 0 = event date).
    """
    cumulative_series = []

    for event_date_str, window_df in all_event_data.items():
        if product in window_df.columns:
            event_date = pd.to_datetime(event_date_str)
            window_df.index = pd.to_datetime(window_df.index)
            window_df = window_df.sort_index()

            # Convert dates to relative days from event date
            relative_days = (window_df.index - event_date).days

            cumsum = window_df[product].cumsum()
            cumsum.index = relative_days
            cumulative_series.append(cumsum)

    if not cumulative_series:
        return None, None, None

    # Align all series to common relative day index
    combined = pd.concat(cumulative_series, axis=1)

    # Calculate statistics
    median = combined.median(axis=1)
    min_val = combined.min(axis=1)
    max_val = combined.max(axis=1)

    # Sort by relative days
    median = median.sort_index()
    min_val = min_val.sort_index()
    max_val = max_val.sort_index()

    return median, min_val, max_val


def calculate_cumulative_discharge_mm(discharge_df):
    """Convert discharge in cfs to cumulative mm using trapezoidal integration."""
    if discharge_df.empty or 'discharge_cfs' not in discharge_df.columns:
        return pd.Series(dtype=float)

    discharge_fts = discharge_df['discharge_cfs'] / DRAINAGE_AREA_FT2
    discharge_ftd = discharge_fts * 60 * 60 * 24
    discharge_mmd = discharge_ftd * 304.8

    dates = discharge_df['date'].values
    date_numeric = (pd.to_datetime(dates) - pd.to_datetime(dates[0])).total_seconds() / (60 * 60 * 24)

    cum_discharge = cumulative_trapezoid(
        discharge_mmd.values,
        x=date_numeric.values,
        initial=0
    )

    return pd.Series(cum_discharge, index=discharge_df.index)


def compute_cumulative_discharge_stats(event_dates, q_df):
    """Compute median and min-max cumulative discharge across events.

    Returns statistics indexed by relative days from event date.
    """
    cumulative_series = []

    for event_date_str in event_dates:
        event_date = pd.to_datetime(event_date_str)

        # Extract 12-day window (T-3 to T+8)
        window_start = event_date - pd.Timedelta(days=3)
        window_end = event_date + pd.Timedelta(days=8)

        # Get discharge data for this window
        discharge_window = q_df[(q_df['date'] >= window_start) & (q_df['date'] <= window_end)]

        if not discharge_window.empty:
            # Calculate cumulative discharge
            cum_discharge = calculate_cumulative_discharge_mm(discharge_window)

            # Convert dates to relative days
            relative_days = (discharge_window['date'] - event_date).dt.days + (discharge_window['date'] - event_date).dt.seconds / (24 * 3600)
            cum_discharge.index = relative_days
            cumulative_series.append(cum_discharge)

    if not cumulative_series:
        return None, None, None

    # Align all series to common relative day index
    combined = pd.concat(cumulative_series, axis=1)

    # Calculate statistics
    median = combined.median(axis=1)
    min_val = combined.min(axis=1)
    max_val = combined.max(axis=1)

    # Sort by relative days
    median = median.sort_index()
    min_val = min_val.sort_index()
    max_val = max_val.sort_index()

    return median, min_val, max_val


def plot_ar_comparison(product='prism'):
    """Create comparison plot of AR vs non-AR events (precipitation + streamflow)."""
    print(f"Extracting data for non-AR events...")
    non_ar_data = extract_all_events(NON_AR_DATES)
    print(f"  Extracted {len(non_ar_data)} non-AR events")

    print(f"Extracting data for AR events...")
    ar_data = extract_all_events(AR_DATES)
    print(f"  Extracted {len(ar_data)} AR events")

    # Load discharge data
    print(f"Loading discharge data...")
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    # Compute precipitation statistics
    print(f"Computing cumulative precipitation statistics for {product.upper()}...")
    non_ar_precip_median, non_ar_precip_min, non_ar_precip_max = compute_cumulative_stats(non_ar_data, product)
    ar_precip_median, ar_precip_min, ar_precip_max = compute_cumulative_stats(ar_data, product)

    # Compute discharge statistics
    print(f"Computing cumulative discharge statistics...")
    non_ar_discharge_median, non_ar_discharge_min, non_ar_discharge_max = compute_cumulative_discharge_stats(NON_AR_DATES, q_df)
    ar_discharge_median, ar_discharge_min, ar_discharge_max = compute_cumulative_discharge_stats(AR_DATES, q_df)

    # Create side-by-side plots with shared y-axis
    fig, (ax_precip, ax_discharge) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Color palette: categorical colors (AR in blue, non-AR in coral/orange)
    color_ar = '#1f77b4'      # Blue for AR
    color_non_ar = '#ff7f0e'  # Orange for non-AR

    # ===== PRECIPITATION PLOT =====
    # Plot non-AR events
    if non_ar_precip_median is not None:
        ax_precip.fill_between(
            non_ar_precip_median.index,
            non_ar_precip_min,
            non_ar_precip_max,
            alpha=0.25,
            color=color_non_ar,
            label='Non-AR (min-max)'
        )
        ax_precip.plot(
            non_ar_precip_median.index,
            non_ar_precip_median,
            linewidth=2.5,
            color=color_non_ar,
            marker='o',
            markersize=6,
            label='Non-AR (median)',
            alpha=0.9
        )

    # Plot AR events
    if ar_precip_median is not None:
        ax_precip.fill_between(
            ar_precip_median.index,
            ar_precip_min,
            ar_precip_max,
            alpha=0.25,
            color=color_ar,
            label='AR (min-max)'
        )
        ax_precip.plot(
            ar_precip_median.index,
            ar_precip_median,
            linewidth=2.5,
            color=color_ar,
            marker='s',
            markersize=6,
            label='AR (median)',
            alpha=0.9
        )

    ax_precip.set_xlabel('Days Relative to Event Date', fontsize=11, fontweight='bold')
    ax_precip.set_ylabel('Cumulative Precipitation (mm)', fontsize=11, fontweight='bold')
    ax_precip.set_title(f'Cumulative {product.upper()} Precipitation', fontsize=12, fontweight='bold', pad=15)
    ax_precip.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax_precip.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # ===== DISCHARGE PLOT =====
    # Plot non-AR events
    if non_ar_discharge_median is not None:
        ax_discharge.fill_between(
            non_ar_discharge_median.index,
            non_ar_discharge_min,
            non_ar_discharge_max,
            alpha=0.25,
            color=color_non_ar,
            label='Non-AR (min-max)'
        )
        ax_discharge.plot(
            non_ar_discharge_median.index,
            non_ar_discharge_median,
            linewidth=2.5,
            color=color_non_ar,
            marker='o',
            markersize=6,
            label='Non-AR (median)',
            alpha=0.9
        )

    # Plot AR events
    if ar_discharge_median is not None:
        ax_discharge.fill_between(
            ar_discharge_median.index,
            ar_discharge_min,
            ar_discharge_max,
            alpha=0.25,
            color=color_ar,
            label='AR (min-max)'
        )
        ax_discharge.plot(
            ar_discharge_median.index,
            ar_discharge_median,
            linewidth=2.5,
            color=color_ar,
            marker='s',
            markersize=6,
            label='AR (median)',
            alpha=0.9
        )

    ax_discharge.set_xlabel('Days Relative to Event Date', fontsize=11, fontweight='bold')
    ax_discharge.set_ylabel('Cumulative Discharge (mm)', fontsize=11, fontweight='bold')
    ax_discharge.set_title('Cumulative Streamflow (USGS 12200500)', fontsize=12, fontweight='bold', pad=15)
    ax_discharge.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax_discharge.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    fig.suptitle(
        f'Top 50 Events: AR vs Non-AR Comparison\n'
        f'Precipitation ({product.upper()}) and Streamflow (Median with Min-Max Range)',
        fontsize=14,
        fontweight='bold',
        y=1.00
    )

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'ar_vs_non_ar_cumulative_{product}_with_discharge.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_ar_comparison(product='prism')
