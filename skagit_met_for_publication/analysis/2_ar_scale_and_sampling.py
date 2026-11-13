"""Combine AR scales across grid points and select AR and isolated non-AR days.

Steps:

1. Combined AR scale: for each day, ``ar_scale`` is the minimum of the AR
   scales at the three grid points (47.5N, 48.0N and 48.5N at 124.5W).
   A day is therefore an AR day (``ar_scale`` >= 1) only if all three grid
   points have an AR; if at least one grid point has scale 0, the day is a
   non-AR day (``ar_scale`` = 0).
2. Isolated non-AR days: non-AR days within ±WINDOW_DAYS days (inclusive)
   of any AR day are removed. All AR days are kept.
3. Period: only days from START_YEAR to END_YEAR are kept. This is applied
   after step 2, so AR days in 1980 still exclude nearby days in early 1981.

Input (in ``<BASE_DIR>/analysis/outputs``):
    1_skagit_daily_integrated_dataset_1980_2025.csv   (from script 1)

Output (in ``<BASE_DIR>/analysis/outputs``):
    2_ar_scale_and_sampling.csv
        Same columns as the input plus ``ar_scale`` (combined AR scale).

Usage:
    python 2_ar_scale_and_sampling.py
"""

import os

import numpy as np
import pandas as pd

# --- Paths ---
# Project root = parent of the folder containing this script (i.e. skagit_met_for_publication/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AR_DATASET_PATH = os.path.join(BASE_DIR, "analysis/outputs/1_skagit_daily_integrated_dataset_1980_2025.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs/2_ar_scale_and_sampling.csv")

# --- Settings ---
WINDOW_DAYS = 5      # non-AR days within ±WINDOW_DAYS of an AR day are removed
START_YEAR = 1981    # first year kept in the output
END_YEAR = 2025      # last year kept in the output

def main():
    """Build the AR / isolated non-AR daily subset and save it to OUTPUT_CSV."""
    print("Selecting AR and isolated non-AR days...")

    # 1. Load AR Master Catalogue
    ar_df = pd.read_csv(AR_DATASET_PATH)
    ar_df['date'] = pd.to_datetime(ar_df['date'])

    # Aggregated AR Scale logic
    scale_cols = [c for c in ar_df.columns if 'ar_scale_' in c]
    if scale_cols:
        ar_df['ar_scale'] = ar_df[scale_cols].min(axis=1)
    elif 'ar_scale' not in ar_df.columns:
        ar_df['ar_scale'] = 0.0

    # 1.25 Remove non-AR dates within ±WINDOW_DAYS of any AR event
    ar_dates = ar_df[ar_df['ar_scale'] > 0]['date'].values

    def is_within_ar_window(date, ar_dates, window_days=WINDOW_DAYS):
        """Return True if `date` is within ±window_days (inclusive) of any date in `ar_dates`."""
        if len(ar_dates) == 0:
            return False
        date_np = np.datetime64(date)
        min_distance = np.min(np.abs((ar_dates - date_np) / np.timedelta64(1, 'D')).astype(int))
        return min_distance <= window_days

    # Filter out non-AR dates within AR window
    non_ar_mask = ar_df['ar_scale'] == 0
    non_ar_in_window = pd.Series(False, index=ar_df.index, dtype=bool)
    non_ar_in_window.loc[non_ar_mask] = ar_df.loc[non_ar_mask, 'date'].apply(lambda d: is_within_ar_window(d, ar_dates)).values
    ar_df = ar_df[~(non_ar_mask & non_ar_in_window)]

    removed_count = non_ar_in_window.sum()
    print(f"Removed {removed_count} non-AR dates within ±{WINDOW_DAYS} days of AR events")

    # Combine all remaining events (AR and isolated non-AR)
    analysis_subset = ar_df.sort_values('date').reset_index(drop=True)
    analysis_subset = analysis_subset[(analysis_subset['date'].dt.year >= START_YEAR) & (analysis_subset['date'].dt.year <= END_YEAR)]
    ar_count = len(analysis_subset[analysis_subset['ar_scale'] > 0])
    non_ar_count = len(analysis_subset[analysis_subset['ar_scale'] == 0])
    print(f"Total analysis targets: {len(analysis_subset)} (AR: {ar_count}, Non-AR Scale 0: {non_ar_count})")

    # Save analysis subset
    analysis_subset.to_csv(OUTPUT_CSV, index=False)

    print(f"Dataset saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
