"""
Compare cumulative precipitation derived from hourly vs daily data.

This script compares the cumulative_precip CSVs (from daily data) with
cumulative_from_hourly_precip CSVs (from hourly data) for each event,
computing the mean percent difference per product.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
OUTPUT_DIR = os.path.join(BASE_DIR, "cumulative_precipitation_plot")
TIMESERIES_DIR = os.path.join(OUTPUT_DIR, "timeseries_data")

# Specific event dates to compare
SPECIFIC_DATES = [
    {'event_date': '1990-11-24', 'start_date': datetime(1990, 11, 21, 0, 0, 0), 'end_date': datetime(1990, 11, 29, 23, 0, 0)},
    {'event_date': '1995-11-29', 'start_date': datetime(1995, 11, 26, 0, 0, 0), 'end_date': datetime(1995, 12, 4, 23, 0, 0)},
    {'event_date': '1990-11-10', 'start_date': datetime(1990, 11, 7, 0, 0, 0), 'end_date': datetime(1990, 11, 15, 23, 0, 0)},
    {'event_date': '2006-11-07', 'start_date': datetime(2006, 11, 4, 0, 0, 0), 'end_date': datetime(2006, 11, 12, 23, 0, 0)},
    {'event_date': '2003-10-21', 'start_date': datetime(2003, 10, 14, 0, 0, 0), 'end_date': datetime(2003, 10, 28, 23, 0, 0)},
    {'event_date': '2021-11-15', 'start_date': datetime(2021, 11, 12, 0, 0, 0), 'end_date': datetime(2021, 11, 20, 23, 0, 0)},
]

PRODUCTS = ['CONUS', 'UCLA', 'PNNL']


def compute_mean_percent_diff(event_date, event_str):
    """
    Compute mean percent difference between hourly-derived and daily-derived
    cumulative precipitation for a single event.
    """
    # File paths
    daily_file = os.path.join(TIMESERIES_DIR, f'cumulative_precip_{event_str}.csv')
    hourly_file = os.path.join(TIMESERIES_DIR, f'cumulative_from_hourly_precip_{event_str}.csv')

    # Check if files exist
    if not os.path.exists(daily_file):
        print(f"  ✗ Daily file not found: {daily_file}")
        return None
    if not os.path.exists(hourly_file):
        print(f"  ✗ Hourly file not found: {hourly_file}")
        return None

    # Load data
    try:
        df_daily = pd.read_csv(daily_file)
        df_hourly = pd.read_csv(hourly_file)
        print(f"  ✓ Daily file loaded: {len(df_daily)} rows")
        print(f"    Columns: {list(df_daily.columns)}")
        print(f"  ✓ Hourly file loaded: {len(df_hourly)} rows")
        print(f"    Columns: {list(df_hourly.columns)}")
    except Exception as e:
        print(f"  ✗ Error loading files: {e}")
        return None

    # Convert datetime columns
    df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
    df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])

    # Results for this event
    results = {'event_date': event_date}

    # Compare each product
    for product in PRODUCTS:
        # Daily file uses lowercase product names (e.g., 'conus', 'pnnl', 'ucla')
        daily_col = product.lower()
        if daily_col not in df_daily.columns:
            results[product] = None
            continue

        # Hourly file uses uppercase with suffix (e.g., 'CONUS_daily_cumsum_mm')
        hourly_col = f'{product.upper()}_daily_cumsum_mm'
        if hourly_col not in df_hourly.columns:
            results[product] = None
            continue

        # Get values (drop NaN)
        daily_vals = df_daily[daily_col].dropna().values
        hourly_vals = df_hourly[hourly_col].dropna().values

        if len(daily_vals) == 0 or len(hourly_vals) == 0:
            results[product] = None
            continue

        # Compute percent difference: ((hourly - daily) / daily) * 100
        # Use the minimum length to align
        min_len = min(len(daily_vals), len(hourly_vals))
        daily_vals = daily_vals[-min_len:]  # Take last N values to ensure end alignment
        hourly_vals = hourly_vals[-min_len:]

        # Avoid division by zero
        mask = daily_vals != 0
        if mask.sum() == 0:
            results[product] = None
            continue

        percent_diffs = ((hourly_vals[mask] - daily_vals[mask]) / daily_vals[mask]) * 100
        mean_percent_diff = np.mean(percent_diffs)

        results[product] = mean_percent_diff

    return results


def main():
    print("Comparing cumulative precipitation: Hourly-derived vs Daily-derived\n")
    print("=" * 80)

    all_results = []

    for event_info in SPECIFIC_DATES:
        event_date = event_info['event_date']
        event_str = pd.to_datetime(event_date).strftime('%Y%m%d')

        print(f"\nEvent: {event_date}")
        print("-" * 80)

        results = compute_mean_percent_diff(event_date, event_str)

        if results:
            all_results.append(results)
            for product in PRODUCTS:
                diff = results.get(product)
                if diff is not None:
                    print(f"  {product:6s}: {diff:8.2f}% (hourly vs daily)")
                else:
                    print(f"  {product:6s}: N/A")

    # Summary statistics
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY - Mean Percent Difference by Product (All Events)")
        print("=" * 80)

        for product in PRODUCTS:
            diffs = [r[product] for r in all_results if r.get(product) is not None]
            if diffs:
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                print(f"  {product:6s}: {mean_diff:8.2f}% ± {std_diff:6.2f}% (mean ± std)")
            else:
                print(f"  {product:6s}: No valid data")

        # Save results to CSV
        results_df = pd.DataFrame(all_results)
        results_file = os.path.join(OUTPUT_DIR, 'comparison_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
