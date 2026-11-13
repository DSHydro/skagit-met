"""Count preceding events (all and non-AR) in 7- and 30-day look-back windows.

For each event date T in the output of script 3 (AR and non-AR events, after
deduplication), adds:

    events_in_prior_7days          events with  T - 7 days  < d <= T - 1 day  (d in T-6 ... T-1)
    non_ar_events_in_prior_7days   same window, only events with ar_scale == 0
    events_in_prior_30days         events with  T - 30 days < d <= T - 1 day  (d in T-29 ... T-1)
    non_ar_events_in_prior_30days  same window, only events with ar_scale == 0
    days_to_prior_event            days since the previous event in the table
                                   (empty for the first event)

The event date T itself is never counted.

Input (in ``<BASE_DIR>/analysis/outputs``):
    3_multi_product_bulk_bias_data.csv   (from script 3)

Output (in ``<BASE_DIR>/analysis/outputs``):
    3.75_events_with_prior_counts.csv
        Input columns (sorted by date) plus the five columns above.

Usage:
    python 3.75_count_prior_events.py
"""

import os

import pandas as pd

# ============================================================
# PATHS
# ============================================================
# Project root = parent of the folder containing this script (i.e. skagit_met_for_publication/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs/3_multi_product_bulk_bias_data.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs/3.75_events_with_prior_counts.csv")

# ============================================================
# SETTINGS
# ============================================================
SHORT_LOOKBACK_DAYS = 7    # events with T - 7  < date <= T - 1 are counted
LONG_LOOKBACK_DAYS = 30    # events with T - 30 < date <= T - 1 are counted


def count_prior_events(df):
    """Add prior-event counts (7 and 30 days, all and non-AR) and days to the previous event.

    Parameters
    ----------
    df : pandas.DataFrame
        Events with a datetime column 'date' and a column 'ar_scale'.

    Returns
    -------
    pandas.DataFrame
        Copy sorted by date with the five new columns added.
    """
    df = df.sort_values('date').reset_index(drop=True)
    df['events_in_prior_7days'] = 0
    df['events_in_prior_30days'] = 0
    df['non_ar_events_in_prior_7days'] = 0
    df['non_ar_events_in_prior_30days'] = 0
    df['days_to_prior_event'] = None

    for idx, row in df.iterrows():
        event_date = row['date']
        end_date = event_date - pd.Timedelta(days=1)

        # Short look-back window
        start_date_7 = event_date - pd.Timedelta(days=SHORT_LOOKBACK_DAYS)
        prior_events_7 = df[(df['date'] > start_date_7) & (df['date'] <= end_date)]
        df.loc[idx, 'events_in_prior_7days'] = len(prior_events_7)
        non_ar_events_7 = prior_events_7[prior_events_7['ar_scale'] == 0]
        df.loc[idx, 'non_ar_events_in_prior_7days'] = len(non_ar_events_7)

        # Long look-back window
        start_date_30 = event_date - pd.Timedelta(days=LONG_LOOKBACK_DAYS)
        prior_events_30 = df[(df['date'] > start_date_30) & (df['date'] <= end_date)]
        df.loc[idx, 'events_in_prior_30days'] = len(prior_events_30)
        non_ar_events_30 = prior_events_30[prior_events_30['ar_scale'] == 0]
        df.loc[idx, 'non_ar_events_in_prior_30days'] = len(non_ar_events_30)

        # Days since the previous event (table is sorted by date)
        if idx > 0:
            prior_event_date = df.loc[idx - 1, 'date']
            days_diff = (event_date - prior_event_date).days
            df.loc[idx, 'days_to_prior_event'] = days_diff

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
    print("\nAll events in prior 7 days:")
    print(f"  Events with prior events: {(df['events_in_prior_7days'] > 0).sum()}")
    print(f"  Events with no prior events: {(df['events_in_prior_7days'] == 0).sum()}")
    print("  Distribution:")
    print(df['events_in_prior_7days'].value_counts().sort_index())
    print("\nNon-AR events (ar_scale=0) in prior 7 days:")
    print(f"  Events with prior non-AR events: {(df['non_ar_events_in_prior_7days'] > 0).sum()}")
    print(f"  Events with no prior non-AR events: {(df['non_ar_events_in_prior_7days'] == 0).sum()}")
    print("  Distribution:")
    print(df['non_ar_events_in_prior_7days'].value_counts().sort_index())
    print("\nAll events in prior 30 days:")
    print(f"  Events with prior events: {(df['events_in_prior_30days'] > 0).sum()}")
    print(f"  Events with no prior events: {(df['events_in_prior_30days'] == 0).sum()}")
    print("  Distribution:")
    print(df['events_in_prior_30days'].value_counts().sort_index())
    print("\nNon-AR events (ar_scale=0) in prior 30 days:")
    print(f"  Events with prior non-AR events: {(df['non_ar_events_in_prior_30days'] > 0).sum()}")
    print(f"  Events with no prior non-AR events: {(df['non_ar_events_in_prior_30days'] == 0).sum()}")
    print("  Distribution:")
    print(df['non_ar_events_in_prior_30days'].value_counts().sort_index())
    print("\nDays to prior event:")
    print(f"  Mean: {df['days_to_prior_event'].mean():.2f} days")
    print(f"  Median: {df['days_to_prior_event'].median():.2f} days")
    print(f"  Min: {df['days_to_prior_event'].min():.0f} days")
    print(f"  Max: {df['days_to_prior_event'].max():.0f} days")
    print(f"\nOutput saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
