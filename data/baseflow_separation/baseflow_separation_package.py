"""Baseflow separation for USGS 12200500 daily discharge using the `baseflow` package.

Reads the USGS .rdb file, builds a continuous daily series, separates baseflow and
direct runoff with the `baseflow` package, and saves a CSV plus a 1995 plot.

Install:  pip install baseflow
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import baseflow

# ============================================================
# PATHS AND SETTINGS
# ============================================================

DATA_DIR = Path("/data0/hernanqd/plots_code/skagit-met/data")
HYDRO_Q_PATH = DATA_DIR / "usgs_12200500_discharge.rdb"
OUTPUT_CSV = DATA_DIR / "baseflow_separation/usgs_12200500_discharge_with_baseflow.csv"
OUTPUT_PLOT = DATA_DIR / "baseflow_separation/baseflow_1995.png"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

CFS_TO_CMS = 0.0283168
METHOD = "LH"                        # package method name (Lyne-Hollick)
AREA_KM2 = 3093 * 2.589988           # drainage area, 3,093 mi2 -> ~8,011 km2


# ============================================================
# LOADER
# ============================================================

def load_usgs_rdb(filepath, param_name):
    """Load a USGS .rdb file and return a DataFrame with ['date', param_name]."""
    print(f"Loading {param_name} from {filepath}...")
    df = pd.read_csv(filepath, sep="\t", comment="#")
    df = df.iloc[1:].reset_index(drop=True)  # drop the field-width row (e.g. 5s, 15s)
    val_col = [c for c in df.columns if "_00" in c and not c.endswith("_cd")][0]
    df = df[["datetime", val_col]]
    df.columns = ["date", param_name]
    df["date"] = pd.to_datetime(df["date"])
    df[param_name] = pd.to_numeric(df[param_name], errors="coerce")
    return df


def to_continuous_daily(df, col):
    """One value per day, on a complete daily index (missing days become NaN)."""
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.normalize()
    s = df.groupby("date")[col].mean()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="D")
    n_missing = len(full_index) - len(s)
    if n_missing:
        print(f"  Filled {n_missing} missing days with NaN")
    return s.reindex(full_index).rename_axis("date").reset_index()


# ============================================================
# MAIN
# ============================================================

q_df = to_continuous_daily(load_usgs_rdb(HYDRO_Q_PATH, "discharge_cfs"), "discharge_cfs")
q_df["discharge_cms"] = q_df["discharge_cfs"] * CFS_TO_CMS

q = q_df.set_index("date")["discharge_cms"]          # the package reads the dates from the index
if q.isna().any():
    raise SystemExit(f"{q.isna().sum()} missing days: fill or split the record before separating")

print(f"Applying '{METHOD}' from the baseflow package (area = {AREA_KM2:,.0f} km2)...")
res = baseflow.single(q, area=AREA_KM2, method=[METHOD])    # confirm with help(baseflow.single)
b = res[0] if isinstance(res, tuple) else res
bf = b[METHOD] if isinstance(b, pd.DataFrame) else pd.Series(b)
q_df["baseflow_cms"] = (bf.reindex(q.index).to_numpy() if isinstance(bf.index, pd.DatetimeIndex)
                        else bf.to_numpy())

q_df["direct_runoff_cms"] = q_df["discharge_cms"] - q_df["baseflow_cms"]
q_df["baseflow_cfs"] = q_df["baseflow_cms"] / CFS_TO_CMS
q_df["direct_runoff_cfs"] = q_df["direct_runoff_cms"] / CFS_TO_CMS

# --------------------------------------------------------
# Sanity checks
# --------------------------------------------------------
valid = q_df["baseflow_cms"].notna()
print(f"  Days with negative direct runoff: {(q_df['direct_runoff_cms'] < -1e-9).sum()}")
print(f"  Days with baseflow > discharge:   {(q_df['baseflow_cms'] > q_df['discharge_cms'] + 1e-9).sum()}")
print(f"  Days without a baseflow value:    {(~valid).sum()}")
print(f"  Baseflow index: {q_df.loc[valid, 'baseflow_cms'].sum() / q_df.loc[valid, 'discharge_cms'].sum():.3f}")

# --------------------------------------------------------
# Plot 1995
# --------------------------------------------------------
df_1995 = q_df[q_df["date"].dt.year == 1995]
if len(df_1995) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_1995["date"], df_1995["discharge_cms"],
            label="Total Discharge", color="black", linewidth=1.5, zorder=3)
    ax.fill_between(df_1995["date"], 0, df_1995["baseflow_cms"],
                    label="Baseflow", alpha=0.6, color="steelblue", zorder=2)
    ax.fill_between(df_1995["date"], df_1995["baseflow_cms"], df_1995["discharge_cms"],
                    label="Direct Runoff", alpha=0.6, color="coral", zorder=1)
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Discharge (cms)", fontsize=12, fontweight="bold")
    ax.set_title(f"{METHOD} Baseflow Separation - Skagit River (USGS 12200500, 1995)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_PLOT}")

# --------------------------------------------------------
# Save
# --------------------------------------------------------
export_cols = ["date", "discharge_cfs", "baseflow_cfs", "direct_runoff_cfs",
               "discharge_cms", "baseflow_cms", "direct_runoff_cms"]
q_df[export_cols].to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
