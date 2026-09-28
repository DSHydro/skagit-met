"""
Compare peak discharge between AR and non-AR events using boxplot.

Extracts maximum discharge in the event window (T-2 to T+5) for each event
and creates a boxplot comparing AR vs non-AR groups.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import plot_specific_ar_events
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_specific_ar_and_non_ar_events.plot_specific_ar_events import load_usgs_rdb

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
OUTPUT_DIR = os.path.join(BASE_DIR, "cumulative_precipitation_plot/top_50_ar_vs_non-AR/cumulative_precipitation_plot")

# Hydrology data
HYDRO_BASE_DIR = "/data0/nksp2/skagit/skagit_2/skagit-met"
HYDRO_EXP_DATA_DIR = os.path.join(HYDRO_BASE_DIR, "experiments_2/data")
HYDRO_Q_PATH = os.path.join(HYDRO_EXP_DATA_DIR, "usgs_12200500_discharge.rdb")

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
# NON_AR_DATES = ['2008-07-02', '2008-07-01', '2008-07-03', '2008-06-30', '2008-07-04', '2020-05-31', '2008-07-05', '1982-06-19', '2008-06-29', '2012-07-09', '2012-07-14', '2012-07-10', '1985-06-08', '2012-07-11', '2012-07-13', '2012-07-15', '1997-05-16', '2008-05-29', '2012-07-19', '1997-05-15', '1982-06-18', '2008-05-28', '2008-07-06', '1997-05-14', '1997-05-17', '2002-06-15', '2012-07-16', '2017-05-30', '2017-05-31', '2021-06-03', '2020-06-01', '2012-07-12', '2008-05-27', '2011-07-07', '2021-06-04', '2002-06-16', '2011-07-08', '2022-07-03', '2012-07-17', '2002-06-14', '2012-07-08', '2022-07-04', '2022-07-02', '2012-07-18', '2011-07-04', '1997-05-13', '2008-05-26', '1986-06-01', '1982-06-17', '1991-07-04'] #5d window
# NON_AR_DATES = ['2008-07-02', '2008-07-01', '2008-07-03', '2008-06-30', '2008-07-04', '2020-05-31', '2008-07-05', '1982-06-19', '2008-06-29', '2012-07-09', '2012-07-14', '2012-07-10', '1985-06-08', '2012-07-11', '2012-07-13', '2012-07-15', '1997-05-16', '2008-05-29', '2012-07-19', '1997-05-15', '1982-06-18', '2008-05-28', '2008-07-06', '1997-05-14', '1997-05-17', '2002-06-15', '2012-07-16', '2017-05-30', '2017-05-31', '2021-06-03', '2020-06-01', '2012-07-12', '2008-05-27', '2011-07-07', '2021-06-04', '2002-06-16', '2011-07-08', '2022-07-03', '2012-07-17', '2002-06-14', '2012-07-08', '2022-07-04', '2022-07-02', '2012-07-18', '2011-07-04', '1997-05-13', '2008-05-26', '1986-06-01', '1982-06-17', '1991-07-04'] #5d window
# NON_AR_DATES = ['2008-07-02', '2008-07-03', '2008-07-04', '2012-07-10', '2012-07-15', '2020-06-01', '2022-07-04', '2022-07-02', '2012-07-18', '1997-05-13', '1986-06-01', '2018-05-16', '1999-08-05', '1999-07-11', '1986-06-02', '1981-01-02', '2008-07-07', '1999-07-22', '1997-06-24', '2011-07-17', '2014-05-16', '1987-05-01', '2011-07-19', '2011-07-09', '1997-06-06', '1999-07-21', '1985-05-24', '2011-07-20', '2011-07-23', '2018-05-24', '2023-05-05', '2018-05-09', '2011-07-18', '2023-05-15', '2017-05-23', '1982-07-04', '1999-07-08', '2006-05-17', '2014-05-04', '1982-05-26', '1986-06-03', '2011-06-14', '2002-07-09', '1982-06-12', '1991-07-25', '2017-12-02', '1985-06-09', '2008-06-01', '1999-07-30', '2022-06-30'] #5d window and 2000 non-AR sample
# NON_AR_DATES = ['2008-07-04', '2012-07-14', '2021-06-30', '1982-06-18', '2002-06-15', '2012-07-16', '2021-07-01', '2002-06-16', '2012-07-17', '1999-06-17', '2002-06-14', '2006-05-18', '2011-07-03', '2012-06-22', '1986-06-01', '2002-07-13', '2002-07-12', '2011-06-07', '2012-07-05', '1999-07-12', '1999-06-18', '2012-05-15', '2018-05-10', '2022-07-05', '2017-06-01', '1991-01-16', '2012-07-07', '1997-06-06', '1982-06-15', '1999-07-25', '2011-07-23', '2002-06-18', '1985-05-25', '2014-05-25', '1982-07-04', '2021-06-05', '1999-07-26', '1986-06-03', '2011-06-14', '1997-06-25', '2012-05-01', '1990-07-12', '2011-07-16', '1999-07-17', '1990-06-11', '2008-07-08', '2017-12-03', '1997-06-26', '2011-06-06', '1999-07-18'] #3d window and 2000 non-AR sample
# NON_AR_DATES = ['2021-11-19', '2008-07-02', '2008-07-01', '2008-07-03', '2008-06-30', '2008-07-04', '2021-11-20', '2020-05-31', '1995-12-04', '2008-07-05', '1982-06-21', '1982-06-20', '1982-06-19', '1990-11-17', '2008-06-29', '2012-07-09', '2012-07-14', '1981-01-01', '2012-07-10', '2021-06-30', '1985-06-08', '2012-07-11', '2012-07-13', '2012-07-15', '1997-05-16', '2008-05-29', '2012-06-24', '2012-07-04', '2012-07-19', '1997-05-15', '1982-06-18', '2008-05-28', '2008-07-06', '1995-12-05', '1997-05-14', '1997-05-17', '2012-06-23', '2002-06-15', '2012-07-16', '2017-05-30', '2006-05-19', '2017-05-31', '2021-07-01', '2021-06-03', '2020-06-01', '2012-07-12', '2008-05-27', '2011-07-07', '2021-06-04', '2002-06-16'] #3d window
NON_AR_DATES = ['1991-03-04', '2012-12-18', '1992-12-22', '2007-02-21', '1983-03-31', '1997-02-20', '2020-03-31', '2018-04-16', '1985-06-08', '1982-03-13'] #, '1990-01-27', '1994-03-19', '1986-02-17', '2021-12-24', '1991-01-01', '1991-11-27', '2010-01-02', '2018-02-18', '1988-03-07', '2011-11-14', '1998-01-07', '2004-09-18', '1982-11-18', '2013-11-19', '1997-04-22', '1983-11-27', '2008-02-01', '2006-04-15', '1988-11-12', '1990-03-10', '1981-04-10', '2000-01-10', '2002-03-21', '2001-05-01', '2015-12-21', '2009-04-14', '2003-02-22', '2000-03-19', '1992-06-14', '2008-12-29', '1984-03-14', '1984-12-23', '2013-03-21', '1990-09-01', '2016-12-30', '2014-02-25', '1996-12-22', '2023-02-08', '1985-04-01', '2016-12-12'] #new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip

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

# AR_DATES = ['1990-11-25', '1995-11-30', '1990-11-11', '2006-11-07', '2003-10-22', '2025-12-12', '2025-12-11', '2003-10-21', '1990-11-24', '2021-11-15', '1995-11-29', '1990-11-10', '1990-11-12', '1989-11-11', '2010-12-13', '2025-12-13', '1984-01-05', '2003-10-23', '2025-12-17', '2017-11-23', '1999-11-13', '1990-11-13', '2011-01-17', '1999-11-14', '2007-03-25', '1997-03-20', '2002-01-08', '2025-12-16', '2015-11-18', '2009-01-08', '1989-11-10', '2014-11-28', '2007-12-04', '2021-11-29', '2025-12-14', '2025-12-18', '2003-10-18', '2004-12-11', '2009-11-17', '2021-11-13', '1986-11-24', '2003-11-19', '2021-10-29', '2015-12-09', '1990-11-23', '2021-12-01', '2001-11-15', '1986-01-19', '2005-01-19', '2021-12-02']
AR_DATES = ['1990-11-24', '2006-11-07', '2009-01-08', '1990-11-11', '1995-11-30', '2003-11-19', '1995-11-10', '2003-10-22', '1989-11-11', '2003-10-21'] #, '1982-02-15', '1997-03-20', '1986-01-19', '1983-01-10', '1995-11-08', '2021-11-15', '1986-11-24', '1988-10-16', '2005-01-19', '1986-02-25', '2003-10-23', '2007-12-04', '2014-11-28', '1985-11-03', '1998-11-15', '1995-02-20', '2009-01-10', '1994-12-01', '2002-02-22', '1982-01-25', '1982-01-24', '2011-01-17', '2010-12-13', '1997-10-31', '1998-12-30', '2004-12-11', '1986-11-25', '2017-11-23', '2014-01-13', '2002-01-08', '1988-04-06', '2021-10-29', '1997-01-01', '1990-10-05', '1996-02-08', '1986-11-20', '1999-11-13', '2022-12-27', '1985-10-27', '1981-02-18'] #new filtering (first, 5 day window arroun ar, then keep contigous dates as one event and assign the date with peak prism_3d_tot) - peak precip



def extract_peak_discharge(event_dates, q_df):
    """Extract maximum discharge in the event window (T-2 to T+5) for each event."""
    peak_discharges = []

    for event_date_str in event_dates:
        event_date = pd.to_datetime(event_date_str)

        # Extract window (T-2 to T+5)
        window_start = event_date - pd.Timedelta(days=2)
        window_end = event_date + pd.Timedelta(days=5)

        # Get discharge data for this window
        discharge_window = q_df[(q_df['date'] >= window_start) & (q_df['date'] <= window_end)]

        if not discharge_window.empty:
            max_cfs = discharge_window['discharge_cfs'].max()
            max_cms = max_cfs * 0.0283168  # Convert cfs to cms
            peak_discharges.append(max_cms)
        else:
            peak_discharges.append(np.nan)

    return peak_discharges


def plot_peak_discharge_comparison():
    """Create boxplot comparing peak discharge between AR and non-AR events."""
    print("Loading discharge data...")
    q_df = load_usgs_rdb(HYDRO_Q_PATH, 'discharge_cfs')

    print("Extracting peak discharge for non-AR events...")
    non_ar_peaks = extract_peak_discharge(NON_AR_DATES, q_df)
    non_ar_peaks = [x for x in non_ar_peaks if not np.isnan(x)]
    print(f"  Extracted {len(non_ar_peaks)} non-AR peak discharges")

    print("Extracting peak discharge for AR events...")
    ar_peaks = extract_peak_discharge(AR_DATES, q_df)
    ar_peaks = [x for x in ar_peaks if not np.isnan(x)]
    print(f"  Extracted {len(ar_peaks)} AR peak discharges")

    # Create boxplot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Prepare data for boxplot
    data_to_plot = [non_ar_peaks, ar_peaks]
    labels = ['Non-AR', 'AR']

    # Color palette
    colors = ['#ff7f0e', '#1f77b4']  # Orange for non-AR, blue for AR

    # Create boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6)
    ax.set_xticklabels(labels)

    # Customize boxplot colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize whiskers, caps, medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5, color='black', alpha=0.7)
    for cap in bp['caps']:
        cap.set(linewidth=1.5, color='black', alpha=0.7)
    for median in bp['medians']:
        median.set(linewidth=2.5, color='black')

    # Labels and formatting
    ax.set_xlabel('Event Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Peak Discharge (m³/s)', fontsize=11, fontweight='bold')
    ax.set_title('Peak Discharge Comparison: AR vs Non-AR Events\n(Event Window: T-2 to T+5)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)

    # Print statistics
    print("\n=== Peak Discharge Statistics ===")
    print(f"Non-AR Events (n={len(non_ar_peaks)}):")
    print(f"  Median: {np.median(non_ar_peaks):.1f} m³/s")
    print(f"  Mean:   {np.mean(non_ar_peaks):.1f} m³/s")
    print(f"  Std:    {np.std(non_ar_peaks):.1f} m³/s")
    print(f"  Min:    {np.min(non_ar_peaks):.1f} m³/s")
    print(f"  Max:    {np.max(non_ar_peaks):.1f} m³/s")

    print(f"\nAR Events (n={len(ar_peaks)}):")
    print(f"  Median: {np.median(ar_peaks):.1f} m³/s")
    print(f"  Mean:   {np.mean(ar_peaks):.1f} m³/s")
    print(f"  Std:    {np.std(ar_peaks):.1f} m³/s")
    print(f"  Min:    {np.min(ar_peaks):.1f} m³/s")
    print(f"  Max:    {np.max(ar_peaks):.1f} m³/s")

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'ar_vs_non_ar_peak_discharge_boxplot.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_peak_discharge_comparison()
