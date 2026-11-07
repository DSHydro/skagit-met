"""2x2 grid scatter plots: streamflow vs antecedent precip by AR status.

Creates four subplots showing for events with prior event within 7 days:
- Top left: AR events with previous AR event
- Top right: AR events with previous non-AR event
- Bottom left: non-AR events with previous AR event
- Bottom right: non-AR events with previous non-AR event

All plots show antecedent precipitation (x-axis) vs discharge (y-axis),
colored by days to previous event.

Input: outputs/3.75_events_with_prior_counts.csv
Output: plots/9.76_flow_vs_precip_ar_categories.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/skagit_met_for_publication"
INPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs/3.75_events_with_prior_counts.csv")
OUT_DIR = os.path.join(BASE_DIR, "plots/bias_boxplots/plots")

def plot_flow_vs_precip_by_ar_categories():
    """Create 2x2 grid of scatter plots by AR status."""

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

    # Convert discharge from cfs to cms (1 cfs = 0.0283168 cms)
    df['discharge_cms'] = df['discharge_cfs'] * 0.0283168

    # Sort by date to ensure proper ordering
    df = df.sort_values('date').reset_index(drop=True)

    # Add previous event's AR status
    df['prev_ar_scale'] = df['ar_scale'].shift(1)

    # Remove rows where days_to_prior_event is NaN (first event)
    df_clean = df.dropna(subset=['days_to_prior_event', 'prev_ar_scale'])

    # Filter to only events with prior event within 7-day window
    df_clean = df_clean[df_clean['days_to_prior_event'] <= 7]

    # Create categories
    df_clean['current_is_ar'] = df_clean['ar_scale'] > 0
    df_clean['prev_is_ar'] = df_clean['prev_ar_scale'] > 0

    # Define the four categories
    categories = [
        (True, True, 'AR with previous AR'),
        (True, False, 'AR with previous non-AR'),
        (False, True, 'non-AR with previous AR'),
        (False, False, 'non-AR with previous non-AR'),
    ]

    print(f"Total events: {len(df)}")
    print(f"Events with prior event within 7-day window: {len(df_clean)}")
    print("\nEvent counts by category:")
    for current_ar, prev_ar, label in categories:
        count = len(df_clean[(df_clean['current_is_ar'] == current_ar) & (df_clean['prev_is_ar'] == prev_ar)])
        print(f"  {label}: {count}")

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='white', sharex=True, sharey=True)
    fig.suptitle('Streamflow vs Antecedent Precipitation by AR Status\nColored by Days to Previous Event',
                 fontsize=16, fontweight='bold', y=0.995)

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    last_scatter = None

    for (current_ar, prev_ar, label), (row, col) in zip(categories, positions):
        ax = axes[row, col]

        # Filter data for this category
        mask = (df_clean['current_is_ar'] == current_ar) & (df_clean['prev_is_ar'] == prev_ar)
        subset = df_clean[mask]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(label, fontsize=13, fontweight='bold')
            continue

        # Create scatter plot
        last_scatter = ax.scatter(
            subset['prism_7d_tot'],
            subset['discharge_cms'],
            c=subset['days_to_prior_event'],
            s=60,
            alpha=0.6,
            cmap='viridis',
            edgecolors='none',
            vmin=0,
            vmax=7
        )

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'{label} (n={len(subset)})', fontsize=13, fontweight='bold')

    # Set axis labels on outer edges
    for row in range(2):
        axes[row, 0].set_ylabel('Discharge (cms)', fontsize=12, labelpad=10)
    for col in range(2):
        axes[1, col].set_xlabel('7-Day Antecedent Precipitation (mm)', fontsize=12, labelpad=10)

    # Add single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_scatter, cax=cbar_ax)
    cbar.set_label('Days to Previous Event', fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 0.91, 0.99])

    # Save figure
    out_img = os.path.join(OUT_DIR, "9.76_flow_vs_precip_ar_categories.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {out_img}")
    plt.close()

    # Print summary statistics by category
    print("\n=== Summary Statistics by Category ===")
    for current_ar, prev_ar, label in categories:
        mask = (df_clean['current_is_ar'] == current_ar) & (df_clean['prev_is_ar'] == prev_ar)
        subset = df_clean[mask]

        if len(subset) == 0:
            print(f"\n{label}: No data")
            continue

        print(f"\n{label} (n={len(subset)}):")
        print(f"  Discharge - mean: {subset['discharge_cms'].mean():.1f}, "
              f"median: {subset['discharge_cms'].median():.1f}, "
              f"max: {subset['discharge_cms'].max():.1f}")
        print(f"  Antecedent precip - mean: {subset['prism_7d_tot'].mean():.1f}, "
              f"median: {subset['prism_7d_tot'].median():.1f}, "
              f"max: {subset['prism_7d_tot'].max():.1f}")
        print(f"  Days to prior - mean: {subset['days_to_prior_event'].mean():.1f}, "
              f"median: {subset['days_to_prior_event'].median():.1f}")

if __name__ == "__main__":
    plot_flow_vs_precip_by_ar_categories()
    print("\nDone!")
