"""Count prior events within 7-day window for each precipitation event.

For each event in the bulk bias dataset, counts how many other events occurred
in the 7 days prior (T-7 to T-1, not including the event date T itself).
Creates a new column 'events_in_prior_7days' with this count.

Input: outputs/3_multi_product_bulk_bias_data.csv
Output: outputs/3.5_events_with_prior_counts.csv
"""

import os
import pandas as pd

BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/updating_multi_product_bulk_bias"
INPUT_CSV = os.path.join(BASE_DIR, "outputs/3_multi_product_bulk_bias_data.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs/3.75_events_with_prior_counts.csv")

def count_prior_events(df):
    """For each event, count events in the 30 days prior, including non-AR events (ar_scale=0)."""
    df = df.sort_values('date').reset_index(drop=True)
    df['events_in_prior_30days'] = 0
    df['non_ar_events_in_prior_30days'] = 0
    df['days_to_prior_event'] = None

    for idx, row in df.iterrows():
        event_date = row['date']
        # 30-day lookback window: T-30 to T-1 (not including T)
        start_date = event_date - pd.Timedelta(days=30)
        end_date = event_date - pd.Timedelta(days=1)

        # Count all events in the prior 30-day window
        prior_events = df[(df['date'] > start_date) & (df['date'] <= end_date)]
        df.loc[idx, 'events_in_prior_30days'] = len(prior_events)

        # Count non-AR events (ar_scale = 0) in the prior 30-day window
        non_ar_events = prior_events[prior_events['ar_scale'] == 0]
        df.loc[idx, 'non_ar_events_in_prior_30days'] = len(non_ar_events)

        # Find days to the most recent prior event
        if idx > 0:
            prior_event_date = df.loc[idx - 1, 'date']
            days_diff = (event_date - prior_event_date).days
            df.loc[idx, 'days_to_prior_event'] = days_diff

    return df

if __name__ == "__main__":
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

    print(f"Processing {len(df)} events...")
    df = count_prior_events(df)

    print(f"Saving output to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSummary:")
    print(f"Total events: {len(df)}")
    print(f"\nAll events in prior 30 days:")
    print(f"  Events with prior events: {(df['events_in_prior_30days'] > 0).sum()}")
    print(f"  Events with no prior events: {(df['events_in_prior_30days'] == 0).sum()}")
    print(f"  Distribution:")
    print(df['events_in_prior_30days'].value_counts().sort_index())
    print(f"\nNon-AR events (ar_scale=0) in prior 30 days:")
    print(f"  Events with prior non-AR events: {(df['non_ar_events_in_prior_30days'] > 0).sum()}")
    print(f"  Events with no prior non-AR events: {(df['non_ar_events_in_prior_30days'] == 0).sum()}")
    print(f"  Distribution:")
    print(df['non_ar_events_in_prior_30days'].value_counts().sort_index())
    print(f"\nDays to prior event:")
    print(f"  Mean: {df['days_to_prior_event'].mean():.2f} days")
    print(f"  Median: {df['days_to_prior_event'].median():.2f} days")
    print(f"  Min: {df['days_to_prior_event'].min():.0f} days")
    print(f"  Max: {df['days_to_prior_event'].max():.0f} days")
    print(f"\nOutput saved to {OUTPUT_CSV}")
