"""Scatter plot: antecedent precipitation vs days to prior event, colored by discharge.

Creates a scatter plot showing the relationship for events with prior event within 7 days:
- X-axis: days to the previous precipitation event
- Y-axis: 7-day antecedent precipitation (prism_7d_tot)
- Color: discharge (cms)

Only includes events with prior event within 7-day window.

Input: outputs/3.75_events_with_prior_counts.csv
Output: plots/9.75_flow_vs_days_to_prior_event.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/updating_multi_product_bulk_bias"
INPUT_CSV = os.path.join(BASE_DIR, "outputs/3.75_events_with_prior_counts.csv")
OUT_DIR = os.path.join(BASE_DIR, "plots")

def plot_flow_vs_days_to_prior_event():
    """Create scatter plot of flow vs days to prior event."""

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

    # Convert discharge from cfs to cms (1 cfs = 0.0283168 cms)
    df['discharge_cms'] = df['discharge_cfs'] * 0.0283168

    # Remove rows where days_to_prior_event is NaN (first event)
    df_clean = df.dropna(subset=['days_to_prior_event'])

    # Filter to only events with prior event within 7-day window
    df_clean = df_clean[df_clean['days_to_prior_event'] <= 7]

    print(f"Total events: {len(df)}")
    print(f"Events with prior event within 7-day window: {len(df_clean)}")
    print(f"Days to prior event - min: {df_clean['days_to_prior_event'].min():.0f}, "
          f"max: {df_clean['days_to_prior_event'].max():.0f}, "
          f"mean: {df_clean['days_to_prior_event'].mean():.1f}")
    print(f"Discharge (cms) - min: {df_clean['discharge_cms'].min():.1f}, "
          f"max: {df_clean['discharge_cms'].max():.1f}, "
          f"mean: {df_clean['discharge_cms'].mean():.1f}")
    print(f"Antecedent precip (prism_7d_tot) - min: {df_clean['prism_7d_tot'].min():.1f}, "
          f"max: {df_clean['prism_7d_tot'].max():.1f}, "
          f"mean: {df_clean['prism_7d_tot'].mean():.1f}")

    plt.figure(figsize=(12, 8), facecolor='white')

    # Create scatter plot
    scatter = plt.scatter(
        df_clean['prism_7d_tot'],
        df_clean['discharge_cms'],
        c=df_clean['days_to_prior_event'],
        s=60,
        alpha=0.6,
        cmap='viridis',
        edgecolors='none',
        vmin=0,
        vmax=7
    )

    plt.xlabel('7-Day Antecedent Precipitation (mm)', fontsize=14, labelpad=10)
    plt.ylabel('Discharge (cms)', fontsize=14, labelpad=10)
    plt.title('Streamflow vs Antecedent Precipitation\nColored by Days to Previous Event',
              fontsize=16, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(scatter, label='Days to Previous Event', pad=0.02)
    cbar.ax.tick_params(labelsize=11)

    # Grid and styling
#     plt.xlim(0,50)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save figure
    out_img = os.path.join(OUT_DIR, "9.75_flow_vs_days_to_prior_event.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {out_img}")
    plt.close()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total points plotted: {len(df_clean)}")

    # Breakdown by antecedent precipitation bins
    print("\nDischarge statistics by antecedent precipitation quintiles:")
    df_clean['precip_quintile'] = pd.qcut(df_clean['prism_7d_tot'], q=5, labels=['Q1 (driest)', 'Q2', 'Q3', 'Q4', 'Q5 (wettest)'])
    for quintile in ['Q1 (driest)', 'Q2', 'Q3', 'Q4', 'Q5 (wettest)']:
        subset = df_clean[df_clean['precip_quintile'] == quintile]
        print(f"\n{quintile}:")
        print(f"  Antecedent precip range: {subset['prism_7d_tot'].min():.1f} - {subset['prism_7d_tot'].max():.1f} mm")
        print(f"  Discharge - mean: {subset['discharge_cms'].mean():.1f}, median: {subset['discharge_cms'].median():.1f}, max: {subset['discharge_cms'].max():.1f}")
        print(f"  Days to prior - mean: {subset['days_to_prior_event'].mean():.1f}, median: {subset['days_to_prior_event'].median():.1f}")

if __name__ == "__main__":
    plot_flow_vs_days_to_prior_event()
    print("\nDone!")
