"""Extract basin-mean precipitation from multiple products for each AR / non-AR day.

For every date T in the output of script 2 (AR days and isolated non-AR days),
basin-mean daily precipitation is computed for each product and summarised
over windows ending on T:

* 3-day window (T-2 to T): total and daily maximum - all products
  (``<product>_3d_tot``, ``<product>_3d_max``).
* 7-day window (T-6 to T): total and daily maximum - all products
  (``<product>_7d_tot``, ``<product>_7d_max``).

A window metric is NaN unless all days of the window are available.

Daily data come from common/prism_day.py (shared with the plot scripts), which
puts every product on the PRISM day: 12 UTC on D-1 to 12 UTC on D (04 PST to
04 PST), labelled D (GitHub issue #61). See that module for how each product is
aligned. ``--no-align`` uses the products' own date labels instead, to reproduce
the earlier, unaligned results.

Products (column prefix):
    prism        PRISM daily 4 km (zipped GeoTIFF, one file per day)
    pnnl         PNNL WRF historical run, hourly (up to 2020)
    daymet       Daymet daily
    conus        CONUS404 (hourly when aligned, daily product otherwise)
    ucla         UCLA WRF-ERA5 d02 (12 UTC running totals when aligned, daily files otherwise)
    gridmet      gridMET daily

Basin mean: unweighted mean of grid cells inside the union of the HUC8
sub-basins Upper Skagit, Sauk and Lower Skagit, using regionmask on each
product's native grid (prism_day.basin_mask / basin_mean).

Deduplication: after extraction, runs of consecutive AR days with the same
``ar_scale`` - and runs of consecutive non-AR days - are each reduced to the
single day with the highest ``prism_7d_tot``.

Inputs:
    <BASE_DIR>/analysis/outputs/2_ar_scale_and_sampling.csv   (from script 2)
    Precipitation datasets - see the paths in common/prism_day.py.

Output:
    <BASE_DIR>/analysis/outputs/3_multi_product_bulk_bias_data_prism_day.csv   (aligned)
    <BASE_DIR>/analysis/outputs/3_multi_product_bulk_bias_data.csv             (--no-align)
        Columns of the input plus the 3-day and 7-day metrics listed above.

Usage:
    python 3_bulk_bias_new_version.py              # all products on the PRISM day
    python 3_bulk_bias_new_version.py --no-align   # date labels used as-is
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# ============================================================
# PATHS - project files (relative to this repository)
# ============================================================
# Project root = parent of the folder containing this script (i.e. skagit_met_for_publication/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BULK_CSV = os.path.join(BASE_DIR, "analysis/outputs/2_ar_scale_and_sampling.csv")
OUTPUT_CSV_UNALIGNED = os.path.join(BASE_DIR, "analysis/outputs/3_multi_product_bulk_bias_data.csv")
OUTPUT_CSV_ALIGNED = os.path.join(BASE_DIR, "analysis/outputs/3_multi_product_bulk_bias_data_prism_day.csv")

# Shared loaders (common/prism_day.py); PYTHONPATH also reaches the parallel workers
sys.path.insert(0, BASE_DIR)
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [BASE_DIR, os.environ.get("PYTHONPATH")]))
from common import prism_day  # noqa: E402

# ============================================================
# SETTINGS
# ============================================================
ALIGN_TO_PRISM_DAY = True  # --no-align sets False
N_JOBS = 12                # parallel workers (one year per worker)
# Column prefixes in output order (PRISM first, as in earlier outputs)
OUTPUT_PRODUCTS = ["prism", "pnnl", "daymet", "conus", "ucla", "gridmet"]
WINDOWS = {"3d": 3, "7d": 7}


def process_year(year, event_dates, align_to_prism_day=ALIGN_TO_PRISM_DAY):
    """Compute the 3-day and 7-day precipitation metrics for all event dates in one year.

    Parameters
    ----------
    year : int
        Calendar year being processed (one parallel job per year).
    event_dates : array of datetime64
        Dates T (AR or non-AR days) in this year.
    align_to_prism_day : bool
        True: all products on the PRISM day; False: products' own date labels.

    Returns
    -------
    pandas.DataFrame
        One row per event date with columns 'date' and '<product>_<3d|7d>_<tot|max>'.
    """
    print(f"Processing Year {year} with {len(event_dates)} events...")
    event_dates = pd.DatetimeIndex(event_dates)

    # All days needed by the windows (T-6 through T)
    window_dates = sorted({t - pd.Timedelta(days=i) for t in event_dates for i in range(7)})

    # Basin-mean daily series of each product (days missing in the source are absent)
    daily = {}
    for product in OUTPUT_PRODUCTS:
        da = prism_day.load_daily(product, window_dates, align=align_to_prism_day)
        daily[product] = (prism_day.basin_mean(da, prism_day.basin_mask(da)).dropna()
                          if da is not None else None)

    results = []
    for t in event_dates:
        row = {'date': t}
        for product in OUTPUT_PRODUCTS:
            s = daily[product]
            for tag, n in WINDOWS.items():
                w = [t - pd.Timedelta(days=i) for i in range(n - 1, -1, -1)]
                if s is not None and all(d in s.index for d in w):
                    row[f'{product}_{tag}_tot'] = s.loc[w].sum()
                    row[f'{product}_{tag}_max'] = s.loc[w].max()
                else:
                    row[f'{product}_{tag}_tot'] = np.nan
                    row[f'{product}_{tag}_max'] = np.nan
        results.append(row)
    return pd.DataFrame(results)


def deduplicate_events_by_prism(df):
    """Reduce runs of consecutive days to one representative day each.

    - AR days (ar_scale > 0): consecutive dates with the same ar_scale form one group.
    - Non-AR days (ar_scale == 0): consecutive dates form one group.

    From each group, the day with the highest prism_7d_tot is kept. Groups in
    which prism_7d_tot is NaN for every day are dropped.
    """
    df = df.sort_values('date').reset_index(drop=True)

    # Process AR events (ar_scale > 0)
    ar_mask = df['ar_scale'] > 0
    ar_events = df[ar_mask].copy().reset_index(drop=True)

    if len(ar_events) > 0:
        # Detect scale changes and date gaps
        ar_events['scale_changed'] = ar_events['ar_scale'].ne(ar_events['ar_scale'].shift()).fillna(True)
        ar_events['date_gap'] = ar_events['date'].diff() > pd.Timedelta(days=1)
        ar_events.loc[0, 'date_gap'] = False
        ar_events['group_id'] = (ar_events['scale_changed'] | ar_events['date_gap']).cumsum()

        # Keep row with max prism_7d_tot for each group (skip groups with all NaN)
        idx_list = []
        for _, group_df in ar_events.groupby('group_id'):
            valid_idx = group_df['prism_7d_tot'].idxmax()
            if pd.notna(valid_idx):
                idx_list.append(valid_idx)
        ar_deduplicated = ar_events.loc[idx_list].drop(columns=['scale_changed', 'date_gap', 'group_id']).reset_index(drop=True)
    else:
        ar_deduplicated = pd.DataFrame()

    # Process non-AR events (ar_scale == 0)
    non_ar_events = df[~ar_mask].copy().reset_index(drop=True)

    if len(non_ar_events) > 0:
        # Detect date gaps
        non_ar_events['date_gap'] = non_ar_events['date'].diff() > pd.Timedelta(days=1)
        non_ar_events.loc[0, 'date_gap'] = False
        non_ar_events['group_id'] = non_ar_events['date_gap'].cumsum()

        # Keep row with max prism_7d_tot for each group (skip groups with all NaN)
        idx_list = []
        for _, group_df in non_ar_events.groupby('group_id'):
            valid_idx = group_df['prism_7d_tot'].idxmax()
            if pd.notna(valid_idx):
                idx_list.append(valid_idx)
        non_ar_deduplicated = non_ar_events.loc[idx_list].drop(columns=['date_gap', 'group_id']).reset_index(drop=True)
    else:
        non_ar_deduplicated = pd.DataFrame()

    # Combine and sort
    result = pd.concat([ar_deduplicated, non_ar_deduplicated]).sort_values('date').reset_index(drop=True)
    return result


def main():
    """Extract metrics for all years in parallel, deduplicate and save."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--no-align', action='store_true',
                        help="use product date labels as-is (no PRISM-day alignment)")
    args = parser.parse_args()
    align = ALIGN_TO_PRISM_DAY and not args.no_align
    output_csv = OUTPUT_CSV_ALIGNED if align else OUTPUT_CSV_UNALIGNED
    print(f"Align products to PRISM day: {align}")

    print("Loading template CSV...")
    df_bulk = pd.read_csv(BULK_CSV)
    df_bulk['date'] = pd.to_datetime(df_bulk['date'])

    years = sorted(df_bulk['date'].dt.year.unique())
    print(f"Starting parallel extraction across {len(years)} years with n_jobs={N_JOBS}...")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_year)(
            year,
            df_bulk[df_bulk['date'].dt.year == year]['date'].dt.normalize().unique(),
            align
        ) for year in years
    )

    print("Combining parallel results...")
    corrected_df = pd.concat(results, ignore_index=True)
    corrected_df['date'] = pd.to_datetime(corrected_df['date'])

    # Merge back into original dataframe by date, overwriting the precipitation columns
    cols_to_overwrite = [col for col in corrected_df.columns if col != 'date']

    print("Merging corrected columns into original dataframe...")
    df_bulk_updated = df_bulk.drop(columns=cols_to_overwrite, errors='ignore')
    df_bulk_updated = pd.merge(df_bulk_updated, corrected_df, on='date', how='inner')

    # Ensure all expected data columns exist (create with NaN if missing)
    expected_cols = [f'{p}_{w}_{m}' for p in OUTPUT_PRODUCTS for w in WINDOWS for m in ('tot', 'max')]
    for col in expected_cols:
        if col not in df_bulk_updated.columns:
            df_bulk_updated[col] = np.nan

    # Deduplicate contiguous events by keeping highest prism_7d_tot
    print("Deduplicating contiguous events by prism_7d_tot...")
    df_bulk_updated = deduplicate_events_by_prism(df_bulk_updated)

    # Save the updated CSV
    print(f"Saving updated CSV to {output_csv}...")
    df_bulk_updated.to_csv(output_csv, index=False)
    print("CSV updated successfully!")


if __name__ == "__main__":
    main()
