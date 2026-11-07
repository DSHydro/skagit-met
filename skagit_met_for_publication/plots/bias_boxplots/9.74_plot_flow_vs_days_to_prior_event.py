"""Scatter plot: 7-day antecedent precipitation vs quickflow, colored by prior 7-day event count.

Creates a scatter plot showing the relationship:
- X-axis: 7-day antecedent precipitation (prism_7d_tot)
- Y-axis: quickflow (direct_runoff_cms)
- Color: number of events in prior 7 days (events_in_prior_7days)

Loads quickflow (direct_runoff_cms) from baseflow-separated discharge data.

Input: outputs/3.75_events_with_prior_counts.csv
       data/baseflow_separation/usgs_12200500_discharge_with_baseflow.csv
Output: plots/9.74_flow_vs_days_to_prior_event.png
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
    """Create scatter plot of quickflow vs antecedent precip colored by prior 7-day events."""

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

    # Load and merge quickflow data
    qf_df = load_quickflow_data()
    df = pd.merge(df, qf_df, on='date', how='left')
    if 'quickflow_cms' in df.columns:
        print(f"  Merged quickflow data for {df['quickflow_cms'].notna().sum()} events")

    # Remove rows with missing data
    df_clean = df.dropna(subset=['prism_7d_tot', 'quickflow_cms', 'events_in_prior_7days'])

    print(f"Total events with complete data: {len(df_clean)}")
    print(f"\nData ranges:")
    print(f"  7-day antecedent precip (prism_7d_tot) - min: {df_clean['prism_7d_tot'].min():.1f}, "
          f"max: {df_clean['prism_7d_tot'].max():.1f}, "
          f"mean: {df_clean['prism_7d_tot'].mean():.1f}")
    print(f"  Quickflow (cms) - min: {df_clean['quickflow_cms'].min():.2f}, "
          f"max: {df_clean['quickflow_cms'].max():.2f}, "
          f"mean: {df_clean['quickflow_cms'].mean():.2f}")
    print(f"  Events in prior 7 days - min: {df_clean['events_in_prior_7days'].min():.0f}, "
          f"max: {df_clean['events_in_prior_7days'].max():.0f}, "
          f"mean: {df_clean['events_in_prior_7days'].mean():.1f}")

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    scatter = ax.scatter(
        df_clean['prism_7d_tot'],
        df_clean['quickflow_cms'],
        c=df_clean['events_in_prior_7days'],
        s=60,
        alpha=0.6,
        cmap='viridis',
        edgecolors='none'
    )

    ax.set_xlabel('7-Day Antecedent Precipitation (mm)', fontsize=14, labelpad=10)
    ax.set_ylabel('Quickflow (cms)', fontsize=14, labelpad=10)
    ax.set_title('Quickflow vs Antecedent Precipitation by Prior 7-Day Event Count',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    cbar = plt.colorbar(scatter, ax=ax, label='Events in Prior 7 Days')
    cbar.set_label('Events in Prior 7 Days', fontsize=12)

    plt.tight_layout()

    # Save figure
    out_img = os.path.join(OUT_DIR, "9.74_flow_vs_days_to_prior_event.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {out_img}")
    plt.close()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total points plotted: {len(df_clean)}")

    # Breakdown by antecedent precipitation bins
    print("\nQuickflow by antecedent precipitation quintiles:")
    df_clean['precip_quintile'] = pd.qcut(df_clean['prism_7d_tot'], q=5, labels=['Q1 (driest)', 'Q2', 'Q3', 'Q4', 'Q5 (wettest)'], duplicates='drop')
    for quintile in df_clean['precip_quintile'].unique():
        subset = df_clean[df_clean['precip_quintile'] == quintile]
        print(f"\n{quintile}:")
        print(f"  Antecedent precip range: {subset['prism_7d_tot'].min():.1f} - {subset['prism_7d_tot'].max():.1f} mm")
        print(f"  Quickflow - mean: {subset['quickflow_cms'].mean():.2f}, median: {subset['quickflow_cms'].median():.2f}, max: {subset['quickflow_cms'].max():.2f}")
        print(f"  Prior 7-day events - mean: {subset['events_in_prior_7days'].mean():.1f}, median: {subset['events_in_prior_7days'].median():.1f}")

if __name__ == "__main__":
    plot_flow_vs_days_to_prior_event()
    print("\nDone!")
