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
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs/3.5_events_with_prior_counts.csv")

def count_prior_events(df):
    """For each event, count how many events occurred in the 7 days prior."""
    df = df.sort_values('date').reset_index(drop=True)
    df['events_in_prior_7days'] = 0

    for idx, row in df.iterrows():
        event_date = row['date']
        # 7-day lookback window: T-7 to T-1 (not including T)
        start_date = event_date - pd.Timedelta(days=7)
        end_date = event_date - pd.Timedelta(days=1)

        # Count events in the prior 7-day window
        prior_events = df[(df['date'] > start_date) & (df['date'] <= end_date)]
        df.loc[idx, 'events_in_prior_7days'] = len(prior_events)

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
    print(f"Events with prior events: {(df['events_in_prior_7days'] > 0).sum()}")
    print(f"Events with no prior events: {(df['events_in_prior_7days'] == 0).sum()}")
    print(f"\nDistribution of prior event counts:")
    print(df['events_in_prior_7days'].value_counts().sort_index())
    print(f"\nOutput saved to {OUTPUT_CSV}")
