"""Water-year precipitation totals of six products, one figure per basin area.

Reads the CSV written by water_year_cumulative.py and, for each area
(Upper Skagit, Sauk, Lower Skagit, and "All" = the three combined), plots the
water-year precipitation total of each product (PRISM, PNNL, Daymet,
CONUS404, UCLA, GridMET) against water year.

Main functions:
  plot_water_year_cumulative: Generate and save cumulative precipitation plots

Input
    plots/precipitation_long_term_means/water_year_cumulative_precipitation.csv
Output
    plots/precipitation_long_term_means/water_year_cumulative_precipitation_<area>.png
        (<area> = upper_skagit, sauk, lower_skagit, all)

Usage
    python plot_water_year_cumulative_data.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths: input CSV and figures live next to this script
OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / 'water_year_cumulative_precipitation.csv'

# Areas (CSV column prefixes) and products (CSV column suffixes), with line colors
GEOMETRIES = ['Upper Skagit', 'Sauk', 'Lower Skagit', 'All']
PRODUCTS = ['PRISM', 'PNNL', 'DAYMET', 'CONUS', 'UCLA', 'GRIDMET']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Figure settings
FIGSIZE = (18, 8)
DPI = 300


def plot_water_year_cumulative():
    """Read CSV and create separate cumulative precipitation plots for each geometry."""

    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please run water_year_cumulative.py first to generate the data.")
        return

    # Read CSV
    df = pd.read_csv(CSV_PATH, index_col='WaterYear')
    print(f"Loaded data from {CSV_PATH}")

    water_years = sorted(df.index.tolist())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create separate plot for each geometry
    for geometry in GEOMETRIES:
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Plot each product for this geometry
        for product, color in zip(PRODUCTS, COLORS):
            col_name = f"{geometry}_{product}"
            if col_name in df.columns:
                totals = df[col_name].dropna()
                if len(totals) > 0:
                    ax.plot(totals.index, totals.values, label=product,
                            linewidth=2.5, color=color, alpha=0.85)

        ax.set_xlabel('Water Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=11, fontweight='bold')
        ax.set_title(f'{geometry} - Water Year Cumulative Precipitation by Dataset', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        if len(water_years) > 1:
            ax.set_xticks(water_years[::max(1, len(water_years)//10)])
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save individual plot
        safe_name = geometry.lower().replace(' ', '_')
        output_path = os.path.join(OUTPUT_DIR, f'water_year_cumulative_precipitation_{safe_name}.png')
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    plot_water_year_cumulative()
