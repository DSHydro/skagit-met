"""
Visualize cumulative precipitation by water year from CSV data.

This module reads cumulative precipitation data from CSV and creates
plots showing the trends across different weather products and geometries.

Main functions:
  plot_water_year_cumulative: Generate and save cumulative precipitation plots
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
OUTPUT_DIR = os.path.join("/data0/hernanqd/plots_code/skagit-met/long-term_means/cumulative_precipitation_plot")
CSV_PATH = os.path.join(OUTPUT_DIR, 'water_year_cumulative_precipitation.csv')


def plot_water_year_cumulative():
    """Read CSV and create separate cumulative precipitation plots for each geometry."""

    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please run create_water_year_cumulative_csv.py first to generate the data.")
        return

    # Read CSV
    df = pd.read_csv(CSV_PATH, index_col='WaterYear')
    print(f"Loaded data from {CSV_PATH}")

    # Get unique geometries and products from column names
    geometries = ['Upper Skagit', 'Sauk', 'Lower Skagit', 'All']
    products = ['PRISM', 'PNNL', 'DAYMET', 'CONUS', 'UCLA', 'GRIDMET']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    water_years = sorted(df.index.tolist())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create separate plot for each geometry
    for geometry in geometries:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot each product for this geometry
        for prod_idx, product in enumerate(products):
            col_name = f"{geometry}_{product}"
            if col_name in df.columns:
                totals = df[col_name].dropna()
                if len(totals) > 0:
                    ax.plot(totals.index, totals.values, label=product,
                           marker=markers[prod_idx], linewidth=2.5, markersize=6,
                           color=colors[prod_idx], alpha=0.85)

        ax.set_xlabel('Water Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=11, fontweight='bold')
        ax.set_title(f'{geometry} - Water Year Cumulative Precipitation by Dataset', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        if len(water_years) > 1:
            ax.set_xticks(water_years[::max(1, len(water_years)//5)])
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save individual plot
        safe_name = geometry.lower().replace(' ', '_')
        output_path = os.path.join(OUTPUT_DIR, f'water_year_cumulative_precipitation_{safe_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    plot_water_year_cumulative()
