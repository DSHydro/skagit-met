"""Create bias columns and bin labels for plotting.

This script reads the cleaned table, computes product-minus-baseline bias,
creates simple precipitation and streamflow bins, and saves the prepared data.
"""

import os
import pandas as pd

BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de/multi_product_bulk_bias"
INPUT_CSV = os.path.join(BASE_DIR, "data", "clean_bias_table.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "prepared_bias_table.csv")


def preprocess_products():
    print("Loading cleaned bias table...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])

    # Create simple streamflow bins.
    bins_sf = [0, 10000, 20000, 40000, float("inf")]
    labels_sf = ["< 10k", "10k - 20k", "20k - 40k", "> 40k"]
    df["Streamflow Bucket (cfs)"] = pd.cut(df["discharge_cfs"], bins=bins_sf, labels=labels_sf)

    # Create simple precipitation bins based on PRISM totals.
    bins_pr = [-1, 20, 50, 100, float("inf")]
    labels_pr = ["< 20mm", "20 - 50mm", "50 - 100mm", "> 100mm"]
    df["Precipitation Bucket (mm)"] = pd.cut(df["prism_3d_tot"], bins=bins_pr, labels=labels_pr)

    # Compute bias columns.
    df["PNNL"] = df["pnnl_3d_tot"] - df["prism_3d_tot"]
    df["CONUS404"] = df["conus_3d_tot"] - df["prism_3d_tot"]
    df["UCLA"] = df["ucla_3d_tot"] - df["prism_3d_tot"]
    df["GridMET"] = df["gridmet_3d_tot"] - df["prism_3d_tot"]
    df["HRRR"] = df["hrrr_3d_tot"] - df["prism_3d_tot"]
    df["ORNL_mean"] = df["ornl_mean_3d_tot"] - df["prism_3d_tot"]
    df["ORNL_median"] = df["ornl_median_3d_tot"] - df["prism_3d_tot"]

    # Save the prepared table.
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved prepared table to {OUTPUT_CSV}")


if __name__ == "__main__":
    preprocess_products()
