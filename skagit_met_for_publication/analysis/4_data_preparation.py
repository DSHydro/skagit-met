"""Reduce the event table to the columns used in the bias analysis.

Steps:
1. Load the event table from script 3.5.
2. Keep only the columns in KEEP_COLS (date, AR scale, discharge and the
   precipitation totals of each product).
3. Drop events without a PRISM 7-day total (PRISM is the reference product).

Input (in ``<BASE_DIR>/analysis/outputs``):
    3.5_events_with_prior_counts.csv   (from script 3.5)

Output (in ``<BASE_DIR>/analysis/outputs``):
    4_clean_bias_table.csv   (columns = KEEP_COLS)

Usage:
    python 4_data_preparation.py
"""

import os

import pandas as pd

# ============================================================
# PATHS
# ============================================================
# Project root = parent of the folder containing this script (i.e. skagit_met_for_publication/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs", "3.5_events_with_prior_counts.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs", "4_clean_bias_table.csv")

# ============================================================
# SETTINGS
# ============================================================
KEEP_COLS = [
    "date",
    "ar_scale",
    "discharge_cfs",
    "prism_3d_tot",
    "prism_7d_tot",
    "daymet_7d_tot",
    "pnnl_7d_tot",
    "conus_7d_tot",
    "ucla_7d_tot",
    "gridmet_7d_tot",
    "hrrr_7d_tot",
]


def main():
    """Select the analysis columns, drop events without PRISM data and save."""
    print("Loading bulk bias data...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])

    df = df[KEEP_COLS].copy()

    # PRISM is the reference product: drop events without a PRISM 7-day total.
    df = df.dropna(subset=["prism_7d_tot"])

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned table to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
