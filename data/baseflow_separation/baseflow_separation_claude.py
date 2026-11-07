"""Baseflow separation for USGS 12200500 daily discharge (Lyne-Hollick filter).

Reads the USGS .rdb daily discharge file, enforces a continuous daily index,
separates baseflow / direct runoff with the Lyne-Hollick recursive digital
filter (forward-backward-forward), and saves a CSV plus a 1995 diagnostic plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("/data0/hernanqd/plots_code/skagit-met/data")
HYDRO_Q_PATH = DATA_DIR / "usgs_12200500_discharge.rdb"
OUTPUT_CSV = DATA_DIR / "baseflow_separation/usgs_12200500_discharge_with_baseflow.csv"
OUTPUT_PLOT = DATA_DIR / "baseflow_separation/baseflow_1995.png"

CFS_TO_CMS = 0.0283168
ALPHA = 0.925   # filter parameter; test 0.90-0.95 for sensitivity
PASSES = 3      # forward, backward, forward


# ============================================================
# LOADER
# ============================================================

def load_usgs_rdb(filepath, param_name):
    """Load a USGS .rdb file and return a DataFrame with ['date', param_name]."""
    print(f"Loading {param_name} from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep="\t", comment="#")
        df = df.iloc[1:].reset_index(drop=True)  # drop the field-width row (e.g. 5s, 15s)
        val_cols = [c for c in df.columns if "_00" in c and not c.endswith("_cd")]
        if not val_cols:
            return pd.DataFrame(columns=["date", param_name])
        val_col = val_cols[0]
        df = df[["datetime", val_col]]
        df.columns = ["date", param_name]
        df["date"] = pd.to_datetime(df["date"])
        df[param_name] = pd.to_numeric(df[param_name], errors="coerce")
        return df
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return pd.DataFrame(columns=["date", param_name])


def to_continuous_daily(df, col):
    """One value per day, on a complete daily index (missing days become NaN)."""
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.normalize()
    df = df.groupby("date", as_index=True)[col].mean()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    n_missing = len(full_index) - len(df)
    if n_missing:
        print(f"  Filled {n_missing} missing days with NaN")
    return df.reindex(full_index).rename_axis("date").reset_index()


# ============================================================
# BASEFLOW SEPARATION - LYNE-HOLLICK
# ============================================================

def _lh_segment(q, alpha, passes):
    """Lyne-Hollick on one gap-free segment. Returns baseflow."""
    bf = q.copy()
    for p in range(passes):
        x = bf if p % 2 == 0 else bf[::-1]          # forward, backward, forward
        qf = np.zeros_like(x)
        for t in range(1, len(x)):
            qf[t] = alpha * qf[t - 1] + 0.5 * (1 + alpha) * (x[t] - x[t - 1])
            qf[t] = min(max(qf[t], 0.0), x[t])       # 0 <= quickflow <= total flow
        b = x - qf
        bf = b if p % 2 == 0 else b[::-1]
    return bf


def lyne_hollick_filter(discharge, alpha=0.925, passes=3):
    """
    Separate baseflow with the Lyne-Hollick (1979) recursive digital filter.

    The filter estimates quickflow from the CHANGE in discharge:
        qf[t] = alpha * qf[t-1] + 0.5 * (1 + alpha) * (q[t] - q[t-1])
        bf[t] = q[t] - qf[t],   with 0 <= qf <= q

    Higher alpha -> smoother quickflow response and higher baseflow is NOT implied;
    always check sensitivity to alpha (typical daily range 0.90-0.95).

    Parameters
    ----------
    discharge : array-like
        Daily discharge on a continuous daily index (NaN allowed; each run of
        valid data is filtered separately so NaNs do not propagate).
    alpha : float
        Filter parameter, default 0.925.
    passes : int
        Total passes, alternating forward/backward/forward (default 3).

    Returns
    -------
    np.ndarray
        Baseflow, same units and length as the input.
    """
    q = np.asarray(discharge, dtype=float)
    out = np.full_like(q, np.nan)
    if len(q) == 0 or np.all(np.isnan(q)):
        return out

    valid = ~np.isnan(q)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(int), [0]))))
    for s, e in zip(edges[::2], edges[1::2]):        # each contiguous run of valid data
        if e - s > 1:
            out[s:e] = _lh_segment(q[s:e], alpha, passes)
        else:
            out[s:e] = q[s:e]
    return out


# ============================================================
# MAIN
# ============================================================

q_df = load_usgs_rdb(HYDRO_Q_PATH, "discharge_cfs")

if len(q_df) > 0 and "discharge_cfs" in q_df.columns:
    q_df = to_continuous_daily(q_df, "discharge_cfs")

    print(f"Applying Lyne-Hollick baseflow separation (alpha={ALPHA}, passes={PASSES})...")
    q_df["baseflow_cfs"] = lyne_hollick_filter(q_df["discharge_cfs"].values, ALPHA, PASSES)
    q_df["direct_runoff_cfs"] = q_df["discharge_cfs"] - q_df["baseflow_cfs"]

    # cms versions (matches the units used in the other figures)
    q_df["discharge_cms"] = q_df["discharge_cfs"] * CFS_TO_CMS
    q_df["baseflow_cms"] = q_df["baseflow_cfs"] * CFS_TO_CMS
    q_df["direct_runoff_cms"] = q_df["direct_runoff_cfs"] * CFS_TO_CMS

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------
    n_neg = (q_df["direct_runoff_cfs"] < -1e-9).sum()
    n_over = (q_df["baseflow_cfs"] > q_df["discharge_cfs"] + 1e-9).sum()
    print(f"  Days with negative direct runoff: {n_neg}")
    print(f"  Days with baseflow > discharge:   {n_over}")

    print("  Baseflow index (sum baseflow / sum discharge) by alpha:")
    for a in (0.90, 0.925, 0.95):
        bf_a = lyne_hollick_filter(q_df["discharge_cfs"].values, a, PASSES)
        bfi = np.nansum(bf_a) / np.nansum(q_df["discharge_cfs"].values)
        print(f"    alpha={a:.3f}: BFI = {bfi:.3f}")

    print("  Sample:")
    print(q_df[["date", "discharge_cfs", "baseflow_cfs", "direct_runoff_cfs"]].head(10))

    # --------------------------------------------------------
    # Plot 1995 baseflow separation
    # --------------------------------------------------------
    q_df["year"] = q_df["date"].dt.year
    df_1995 = q_df[q_df["year"] == 1995].reset_index(drop=True)

    if len(df_1995) > 0:
        print("\n" + "=" * 60)
        print("Creating baseflow separation plot for 1995...")
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(df_1995["date"], df_1995["discharge_cms"],
                label="Total Discharge", color="black", linewidth=1.5, zorder=3)
        ax.fill_between(df_1995["date"], 0, df_1995["baseflow_cms"],
                        label="Baseflow", alpha=0.6, color="steelblue", zorder=2)
        ax.fill_between(df_1995["date"], df_1995["baseflow_cms"], df_1995["discharge_cms"],
                        label="Direct Runoff", alpha=0.6, color="coral", zorder=1)

        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("Discharge (cms)", fontsize=12, fontweight="bold")
        # NOTE: verify the station name; USGS 12200500 is the Skagit near Mount Vernon
        ax.set_title("Lyne-Hollick Baseflow Separation - Skagit River (USGS 12200500, 1995)",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {OUTPUT_PLOT}")

        print("\n1995 Baseflow Separation Statistics:")
        print(f"  Mean Total Discharge: {df_1995['discharge_cms'].mean():.1f} cms")
        print(f"  Mean Baseflow: {df_1995['baseflow_cms'].mean():.1f} cms")
        print(f"  Mean Direct Runoff: {df_1995['direct_runoff_cms'].mean():.1f} cms")
        print(f"  Baseflow fraction: "
              f"{df_1995['baseflow_cms'].mean() / df_1995['discharge_cms'].mean() * 100:.1f}%")
        print(f"  Max Discharge: {df_1995['discharge_cms'].max():.1f} cms")
        print(f"  Min Discharge: {df_1995['discharge_cms'].min():.1f} cms")
        print("=" * 60)

    # --------------------------------------------------------
    # Save full dataset
    # --------------------------------------------------------
    export_cols = [
        "date",
        "discharge_cfs", "baseflow_cfs", "direct_runoff_cfs",
        "discharge_cms", "baseflow_cms", "direct_runoff_cms",
    ]
    q_df[export_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull dataset with baseflow columns saved to:\n  {OUTPUT_CSV}")
else:
    print("No discharge data loaded; nothing to do.")
