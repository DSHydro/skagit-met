"""Add bias columns and streamflow / precipitation classes for plotting.

Steps:
1. Convert discharge from cfs to m^3/s (``discharge_cms``).
2. Assign each event to a streamflow class (``Streamflow Bucket (cms)``) and
   a precipitation class based on the PRISM 7-day total
   (``Precipitation Bucket (mm)``). Bins are right-closed, e.g. (600, 1200].
3. Compute the bias of each product's 7-day total relative to PRISM:
       absolute bias  <PRODUCT>   = product_7d_tot - prism_7d_tot              [mm]
       percent bias   <PRODUCT>_% = (product_7d_tot - prism_7d_tot)
                                    / prism_7d_tot * 100                       [%]
   for PNNL, Daymet, CONUS404, UCLA and GridMET.
4. The same for the 3-day window (event day and the 2 prior days):
       ``Precipitation Bucket 3d (mm)`` from prism_3d_tot (same bins), and
       <PRODUCT>_3d   = product_3d_tot - prism_3d_tot                         [mm]
       <PRODUCT>_3d_% = (product_3d_tot - prism_3d_tot) / prism_3d_tot * 100  [%]

Input (in ``<BASE_DIR>/analysis/outputs``):
    4_clean_bias_table.csv   (from script 4)

Output (in ``<BASE_DIR>/analysis/outputs``):
    5_prepared_bias_table.csv
        Input columns plus discharge_cms, the class columns and the 7-day and
        3-day absolute and percent bias columns.

Usage:
    python 5_preprocess_products.py
"""

import os

import pandas as pd

# ============================================================
# PATHS
# ============================================================
# Project root = parent of the folder containing this script (i.e. skagit_met_for_publication/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs", "4_clean_bias_table.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "analysis/outputs", "5_prepared_bias_table.csv")

# ============================================================
# SETTINGS
# ============================================================
CFS_TO_CMS = 0.0283168   # 1 ft^3/s = 0.0283168 m^3/s

# Streamflow classes [m^3/s]
STREAMFLOW_BINS = [0, 600, 1200, 1700, float("inf")]
STREAMFLOW_LABELS = ["< 600", "600 - 1200", "1200 - 1700", "> 1700"]

# Precipitation classes, PRISM 7-day (and 3-day) total [mm]
PRECIP_BINS = [-1, 20, 50, 100, float("inf")]
PRECIP_LABELS = ["< 20mm", "20 - 50mm", "50 - 100mm", "> 100mm"]

# Bias column name -> product column prefix
PRODUCT_COLS = {"PNNL": "pnnl", "Daymet": "daymet", "CONUS404": "conus", "UCLA": "ucla", "GridMET": "gridmet"}


def main():
    """Add discharge in m^3/s, class columns and bias columns, then save."""
    print("Loading cleaned bias table...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])

    # Streamflow classes
    df["discharge_cms"] = df["discharge_cfs"] * CFS_TO_CMS
    df["Streamflow Bucket (cms)"] = pd.cut(df["discharge_cms"], bins=STREAMFLOW_BINS, labels=STREAMFLOW_LABELS)

    # Precipitation classes (PRISM 7-day total)
    df["Precipitation Bucket (mm)"] = pd.cut(df["prism_7d_tot"], bins=PRECIP_BINS, labels=PRECIP_LABELS)

    # Absolute bias relative to PRISM [mm]
    df["PNNL"] = df["pnnl_7d_tot"] - df["prism_7d_tot"]
    df["Daymet"] = df["daymet_7d_tot"] - df["prism_7d_tot"]
    df["CONUS404"] = df["conus_7d_tot"] - df["prism_7d_tot"]
    df["UCLA"] = df["ucla_7d_tot"] - df["prism_7d_tot"]
    df["GridMET"] = df["gridmet_7d_tot"] - df["prism_7d_tot"]

    # Percent bias relative to PRISM [%]
    df["PNNL_%"] = (df["pnnl_7d_tot"] - df["prism_7d_tot"]) / df["prism_7d_tot"] * 100
    df["Daymet_%"] = (df["daymet_7d_tot"] - df["prism_7d_tot"]) / df["prism_7d_tot"] * 100
    df["CONUS404_%"] = (df["conus_7d_tot"] - df["prism_7d_tot"]) / df["prism_7d_tot"] * 100
    df["UCLA_%"] = (df["ucla_7d_tot"] - df["prism_7d_tot"]) / df["prism_7d_tot"] * 100
    df["GridMET_%"] = (df["gridmet_7d_tot"] - df["prism_7d_tot"]) / df["prism_7d_tot"] * 100

    # Same for the 3-day window (event day and the 2 prior days)
    df["Precipitation Bucket 3d (mm)"] = pd.cut(df["prism_3d_tot"], bins=PRECIP_BINS, labels=PRECIP_LABELS)
    for name, col in PRODUCT_COLS.items():
        df[f"{name}_3d"] = df[f"{col}_3d_tot"] - df["prism_3d_tot"]
        df[f"{name}_3d_%"] = (df[f"{col}_3d_tot"] - df["prism_3d_tot"]) / df["prism_3d_tot"] * 100

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved prepared table to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
