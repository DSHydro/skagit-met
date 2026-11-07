"""Scatter plot: antecedent precipitation vs quickflow, colored by days to prior event.

Creates a scatter plot showing the relationship for events with prior event within 7 days:
- X-axis: 7-day antecedent precipitation (prism_7d_tot)
- Y-axis: quickflow (direct_runoff_cms)
- Color: days to previous event

Also loads quickflow (direct_runoff_cms) from baseflow-separated discharge data.

Only includes events with prior event within 5-day window.

Input: outputs/3.75_events_with_prior_counts.csv
       data/baseflow_separation/usgs_12200500_discharge_with_baseflow.csv
Output: plots/9.75_flow_vs_days_to_prior_event.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/skagit_met_for_publication"
INPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs/3.75_events_with_prior_counts.csv")
QUICKFLOW_CSV = Path("/data0/hernanqd/plots_code/skagit-met/data/baseflow_separation/usgs_12200500_discharge_with_baseflow.csv")
OUT_DIR = os.path.join(BASE_DIR, "plots/bias_boxplots/plots")

def load_quickflow_data():
    """Load quickflow (direct_runoff_cms) from baseflow-separated discharge data."""
    print(f"Loading quickflow data from {QUICKFLOW_CSV}...")
    qf_df = pd.read_csv(QUICKFLOW_CSV, parse_dates=['date'])
    print(f"  Loaded {len(qf_df)} days of discharge data")
    return qf_df[['date', 'direct_runoff_cms']].rename(columns={'direct_runoff_cms': 'quickflow_cms'})


def plot_flow_vs_days_to_prior_event():
    """Create scatter plot of flow vs days to prior event."""

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

    # Convert discharge from cfs to cms (1 cfs = 0.0283168 cms)
    df['discharge_cms'] = df['discharge_cfs'] * 0.0283168

    # Load and merge quickflow data
    qf_df = load_quickflow_data()
    df = pd.merge(df, qf_df, on='date', how='left')
    if 'quickflow_cms' in df.columns:
        print(f"  Merged quickflow data for {df['quickflow_cms'].notna().sum()} events")

    # Remove rows where days_to_prior_event is NaN (first event)
    df_clean = df.dropna(subset=['days_to_prior_event'])

    # Split data into two groups: within 5 days and beyond 5 days
    df_within_5 = df_clean[df_clean['days_to_prior_event'] <= 5].copy()
    df_beyond_5 = df_clean[df_clean['days_to_prior_event'] > 5].copy()

    print(f"Total events: {len(df)}")
    print(f"Events with prior event within 5-day window: {len(df_within_5)}")
    print(f"Events with prior event > 5 days: {len(df_beyond_5)}")
    print(f"\nWithin 7-day window:")
    print(f"  Days to prior event - min: {df_within_5['days_to_prior_event'].min():.0f}, "
          f"max: {df_within_5['days_to_prior_event'].max():.0f}, "
          f"mean: {df_within_5['days_to_prior_event'].mean():.1f}")
    print(f"  Antecedent precip (prism_7d_tot) - min: {df_within_5['prism_7d_tot'].min():.1f}, "
          f"max: {df_within_5['prism_7d_tot'].max():.1f}, "
          f"mean: {df_within_5['prism_7d_tot'].mean():.1f}")
    if 'quickflow_cms' in df_within_5.columns and df_within_5['quickflow_cms'].notna().sum() > 0:
        print(f"  Quickflow (cms) - min: {df_within_5['quickflow_cms'].min():.2f}, "
              f"max: {df_within_5['quickflow_cms'].max():.2f}, "
              f"mean: {df_within_5['quickflow_cms'].mean():.2f}")

    print(f"\nBeyond 7-day window:")
    print(f"  Days to prior event - min: {df_beyond_5['days_to_prior_event'].min():.0f}, "
          f"max: {df_beyond_5['days_to_prior_event'].max():.0f}, "
          f"mean: {df_beyond_5['days_to_prior_event'].mean():.1f}")
    print(f"  Antecedent precip (prism_7d_tot) - min: {df_beyond_5['prism_7d_tot'].min():.1f}, "
          f"max: {df_beyond_5['prism_7d_tot'].max():.1f}, "
          f"mean: {df_beyond_5['prism_7d_tot'].mean():.1f}")
    if 'quickflow_cms' in df_beyond_5.columns and df_beyond_5['quickflow_cms'].notna().sum() > 0:
        print(f"  Quickflow (cms) - min: {df_beyond_5['quickflow_cms'].min():.2f}, "
              f"max: {df_beyond_5['quickflow_cms'].max():.2f}, "
              f"mean: {df_beyond_5['quickflow_cms'].mean():.2f}")

    # Create 1x2 subplot grid with shared y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True, facecolor='white')

    # Left plot: events within 7 days
    scatter1 = ax1.scatter(
        df_within_5['prism_7d_tot'],
        df_within_5['discharge_cms'],
        c=df_within_5['days_to_prior_event'],
        s=60,
        alpha=0.6,
        cmap='viridis',
        edgecolors='none',
        vmin=0,
        vmax=5
    )

    ax1.set_xlabel('7-Day Antecedent Precipitation (mm)', fontsize=14, labelpad=10)
    ax1.set_ylabel('Discharge (cms)', fontsize=14, labelpad=10)
    ax1.set_title(f'Days to Prior Event ≤ 5 (n={len(df_within_5)})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Right plot: events beyond 3 days
    scatter2 = ax2.scatter(
        df_beyond_5['prism_7d_tot'],
        df_beyond_5['discharge_cms'],
        c=df_beyond_5['days_to_prior_event'],
        s=60,
        alpha=0.6,
        cmap='viridis',
        edgecolors='none',
        vmin=0,
        vmax=df_clean['days_to_prior_event'].max()
    )

    ax2.set_xlabel('7-Day Antecedent Precipitation (mm)', fontsize=14, labelpad=10)
    ax2.set_title(f'Days to Prior Event > 5 (n={len(df_beyond_5)})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Streamflow vs Antecedent Precipitation by Days to Previous Event',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    # Save figure
    out_img = os.path.join(OUT_DIR, "9.75_flow_vs_days_to_prior_event.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {out_img}")
    plt.close()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total points plotted (within 7 days): {len(df_within_5)}")
    print(f"Total points plotted (beyond 7 days): {len(df_beyond_5)}")

    # Breakdown by antecedent precipitation bins for within 7-day group
    print("\nQuickflow by antecedent precipitation quintiles (Days to Prior ≤ 5):")
    df_within_5['precip_quintile'] = pd.qcut(df_within_5['prism_7d_tot'], q=5, labels=['Q1 (driest)', 'Q2', 'Q3', 'Q4', 'Q5 (wettest)'], duplicates='drop')
    for quintile in df_within_5['precip_quintile'].unique():
        subset = df_within_5[df_within_5['precip_quintile'] == quintile]
        print(f"\n{quintile}:")
        print(f"  Antecedent precip range: {subset['prism_7d_tot'].min():.1f} - {subset['prism_7d_tot'].max():.1f} mm")
        if 'quickflow_cms' in subset.columns and subset['quickflow_cms'].notna().sum() > 0:
            print(f"  Quickflow - mean: {subset['quickflow_cms'].mean():.2f}, median: {subset['quickflow_cms'].median():.2f}, max: {subset['quickflow_cms'].max():.2f}")
        print(f"  Days to prior - mean: {subset['days_to_prior_event'].mean():.1f}, median: {subset['days_to_prior_event'].median():.1f}")

    # Breakdown by antecedent precipitation bins for beyond 7-day group
    print("\n\nQuickflow by antecedent precipitation quintiles (Days to Prior > 5):")
    df_beyond_5['precip_quintile'] = pd.qcut(df_beyond_5['prism_7d_tot'], q=5, labels=['Q1 (driest)', 'Q2', 'Q3', 'Q4', 'Q5 (wettest)'], duplicates='drop')
    for quintile in df_beyond_5['precip_quintile'].unique():
        subset = df_beyond_5[df_beyond_5['precip_quintile'] == quintile]
        print(f"\n{quintile}:")
        print(f"  Antecedent precip range: {subset['prism_7d_tot'].min():.1f} - {subset['prism_7d_tot'].max():.1f} mm")
        if 'quickflow_cms' in subset.columns and subset['quickflow_cms'].notna().sum() > 0:
            print(f"  Quickflow - mean: {subset['quickflow_cms'].mean():.2f}, median: {subset['quickflow_cms'].median():.2f}, max: {subset['quickflow_cms'].max():.2f}")
        print(f"  Days to prior - mean: {subset['days_to_prior_event'].mean():.1f}, median: {subset['days_to_prior_event'].median():.1f}")

if __name__ == "__main__":
    plot_flow_vs_days_to_prior_event()
    print("\nDone!")
