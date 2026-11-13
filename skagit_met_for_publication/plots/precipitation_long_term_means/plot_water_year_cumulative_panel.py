"""
Publication figure: 3x1 panel of water-year cumulative precipitation.

Panels (top to bottom):
  (a) Lower Skagit
  (b) Upper Skagit
  (c) Sauk

Reads the same CSV as plot_water_year_cumulative_data.py
(run create_water_year_cumulative_csv.py first if it is missing).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = "/data0/hernanqd/plots_code/skagit-met/skagit_met_for_publication/plots/precipitation_long_term_means"
CSV_PATH = os.path.join(OUTPUT_DIR, 'water_year_cumulative_precipitation.csv')
OUTPUT_NAME = 'water_year_cumulative_precipitation_panel'

GEOMETRIES = ['Lower Skagit', 'Upper Skagit', 'Sauk']
PRODUCTS = ['PRISM', 'PNNL', 'DAYMET', 'CONUS', 'UCLA', 'GRIDMET']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Figure size in inches: ~7 in = full-page width for most journals
FIG_WIDTH = 7.0
FIG_HEIGHT = 8.0

# Font sizes suited to a printed figure at FIG_WIDTH
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'savefig.dpi': 300,
})


def plot_water_year_cumulative_panel():
    """Create a 3x1 panel figure of water-year cumulative precipitation."""

    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please run create_water_year_cumulative_csv.py first to generate the data.")
        return

    df = pd.read_csv(CSV_PATH, index_col='WaterYear').sort_index()
    print(f"Loaded data from {CSV_PATH}")

    water_years = df.index.tolist()

    fig, axes = plt.subplots(len(GEOMETRIES), 1, figsize=(FIG_WIDTH, FIG_HEIGHT),
                             sharex=True, sharey=True)

    for i, (ax, geometry) in enumerate(zip(axes, GEOMETRIES)):
        for product, color in zip(PRODUCTS, COLORS):
            col_name = f"{geometry}_{product}"
            if col_name not in df.columns:
                continue
            totals = df[col_name].dropna()
            if len(totals) > 0:
                ax.plot(totals.index, totals.values, label=product,
                        linewidth=1.4, color=color, alpha=0.9)

        # Panel label + basin name inside the axes
        ax.text(0.01, 0.96, f"({chr(97 + i)}) {geometry}", transform=ax.transAxes,
                ha='left', va='top', fontweight='bold')

        ax.set_ylabel('Precipitation (mm)')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        ax.set_xlim(water_years[0], water_years[-1])

    # Shared x-axis
    step = 5
    ticks = [y for y in water_years if y % step == 0]
    axes[-1].set_xticks(ticks)
    axes[-1].set_xlabel('Water Year')

    # Single legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.5, columnspacing=1.0)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.subplots_adjust(hspace=0.08)

    output_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_NAME}.png')
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_water_year_cumulative_panel()
