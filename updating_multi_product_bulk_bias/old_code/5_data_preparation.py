"""Prepare a clean, easy-to-read analysis table for Skagit precipitation bias work.

This script focuses on the most important steps:
1. Load the bulk bias CSV.
2. Convert the date column to datetime.
3. Keep only the key columns needed for downstream analysis.
4. Save a simplified CSV for reuse.
"""

import os
import pandas as pd

BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de/multi_product_bulk_bias"
INPUT_CSV = os.path.join(BASE_DIR, "outputs", "multi_product_bulk_bias_data.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "clean_bias_table.csv")


def prepare_data():
    print("Loading bulk bias data...")
    df = pd.read_csv(INPUT_CSV)

    # Convert date to datetime for easier filtering.
    df["date"] = pd.to_datetime(df["date"])

    # Keep only the columns needed for the simplified workflow.
    keep_cols = [
        "date",
        "ar_scale",
        "discharge_cfs",
        "prism_3d_tot",
        "daymet_3d_tot",
        "pnnl_3d_tot",
        "conus_3d_tot",
        "ucla_3d_tot",
        "gridmet_3d_tot",
        "hrrr_3d_tot",
        "ornl_mean_3d_tot",
        "ornl_median_3d_tot",
    ]
    df = df[keep_cols].copy()

    # Drop rows with no usable precipitation values.
    df = df.dropna(subset=["prism_3d_tot"])

    # Save a cleaned version for later scripts.
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned table to {OUTPUT_CSV}")


if __name__ == "__main__":
    prepare_data()
