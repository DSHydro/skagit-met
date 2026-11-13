"""Count how many events precede each event within a short look-back window.

For each event date T in the output of script 3 (AR and non-AR events, after
deduplication), counts the other events whose date d satisfies

    T - 7 days < d <= T - 1 day      (i.e. d in T-6 ... T-1)

and stores the count in a new column ``events_in_prior_7days``. The event
date T itself is not counted.

Input (in ``<BASE_DIR>/analysis/outputs``):
    3_multi_product_bulk_bias_data.csv   (from script 3)

Output (in ``<BASE_DIR>/analysis/outputs``):
    3.5_events_with_prior_counts.csv
        Input columns (sorted by date) plus ``events_in_prior_7days``.

Usage:
    python 3.5_count_prior_events.py
"""

import os

import pandas as pd

# ============================================================
# PATHS
# ============================================================
# Project root = parent of the folder containing this script (i.e. skagit_met_for_publication/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs/3_multi_product_bulk_bias_data.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs/3.5_events_with_prior_counts.csv")

# ============================================================
# SETTINGS
# ============================================================
LOOKBACK_DAYS = 7   # events with T - LOOKBACK_DAYS < date <= T - 1 are counted


def count_prior_events(df):
    """Add ``events_in_prior_7days``: number of events with T-7 < date <= T-1.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with a datetime column 'date'.

    Returns
    -------
    pandas.DataFrame
        Copy sorted by date with the new column added.
    """
    df = df.sort_values('date').reset_index(drop=True)
    df['events_in_prior_7days'] = 0

    for idx, row in df.iterrows():
        event_date = row['date']
        start_date = event_date - pd.Timedelta(days=LOOKBACK_DAYS)
        end_date = event_date - pd.Timedelta(days=1)

        prior_events = df[(df['date'] > start_date) & (df['date'] <= end_date)]
        df.loc[idx, 'events_in_prior_7days'] = len(prior_events)

    return df


def main():
    """Load script 3 output, count prior events, save and print a summary."""
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

    print(f"Processing {len(df)} events...")
    df = count_prior_events(df)

    print(f"Saving output to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSummary:")
    print(f"Total events: {len(df)}")
    print(f"Events with prior events: {(df['events_in_prior_7days'] > 0).sum()}")
    print(f"Events with no prior events: {(df['events_in_prior_7days'] == 0).sum()}")
    print("\nDistribution of prior event counts:")
    print(df['events_in_prior_7days'].value_counts().sort_index())
    print(f"\nOutput saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
