"""Discharge vs 7-day antecedent precipitation, split by days to prior event.

Two side-by-side scatter plots (shared y-axis) of events that have a prior
event, split by the time since that prior event:

    left  : days to prior event <= 5
    right : days to prior event > 5

    x-axis : 7-day PRISM antecedent precipitation (``prism_7d_tot``)
    y-axis : USGS 12200500 discharge (cms)
    color  : days to previous event (``days_to_prior_event``)

Quickflow (direct runoff from baseflow separation) is merged in only for the
summary statistics by antecedent-precipitation quintile printed to the
console.

Input
    analysis/outputs/5day_taking_min_and_filling/3.75_events_with_prior_counts.csv
    ../data/baseflow_separation/usgs_12200500_discharge_with_baseflow.csv
Output
    plots/bias_boxplots/plots/9.75_flow_vs_days_to_prior_event.png

Usage
    python 9.75_plot_flow_vs_days_to_prior_event.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "analysis" / "outputs" / "5day_taking_min_and_filling" / "3.75_events_with_prior_counts.csv"
QUICKFLOW_CSV = BASE_DIR.parent / "data" / "baseflow_separation" / "usgs_12200500_discharge_with_baseflow.csv"
OUT_DIR = BASE_DIR / "plots" / "bias_boxplots" / "plots"

CFS_TO_CMS = 0.0283168
DAYS_THRESHOLD = 5  # split between "recent" and "older" prior events
QUINTILE_LABELS = ["Q1 (driest)", "Q2", "Q3", "Q4", "Q5 (wettest)"]


def load_quickflow_data():
    """Load quickflow (direct_runoff_cms) from baseflow-separated discharge data."""
    print(f"Loading quickflow data from {QUICKFLOW_CSV}...")
    qf_df = pd.read_csv(QUICKFLOW_CSV, parse_dates=["date"])
    print(f"  Loaded {len(qf_df)} days of discharge data")
    return qf_df[["date", "direct_runoff_cms"]].rename(columns={"direct_runoff_cms": "quickflow_cms"})


def plot_flow_vs_days_to_prior_event():
    """Create scatter plots of discharge vs antecedent precip by days to prior event."""

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df["discharge_cms"] = df["discharge_cfs"] * CFS_TO_CMS

    # Load and merge quickflow data
    df = pd.merge(df, load_quickflow_data(), on="date", how="left")
    print(f"  Merged quickflow data for {df['quickflow_cms'].notna().sum()} events")

    # Remove rows where days_to_prior_event is NaN (first event)
    df_clean = df.dropna(subset=["days_to_prior_event"])

    # Split into events with a recent and an older prior event
    df_within = df_clean[df_clean["days_to_prior_event"] <= DAYS_THRESHOLD].copy()
    df_beyond = df_clean[df_clean["days_to_prior_event"] > DAYS_THRESHOLD].copy()

    print(f"Total events: {len(df)}")
    print(f"Events with prior event within 5-day window: {len(df_within)}")
    print(f"Events with prior event > 5 days: {len(df_beyond)}")
    print("\nWithin 7-day window:")
    print(f"  Days to prior event - min: {df_within['days_to_prior_event'].min():.0f}, "
          f"max: {df_within['days_to_prior_event'].max():.0f}, "
          f"mean: {df_within['days_to_prior_event'].mean():.1f}")
    print(f"  Antecedent precip (prism_7d_tot) - min: {df_within['prism_7d_tot'].min():.1f}, "
          f"max: {df_within['prism_7d_tot'].max():.1f}, "
          f"mean: {df_within['prism_7d_tot'].mean():.1f}")
    if df_within["quickflow_cms"].notna().any():
        print(f"  Quickflow (cms) - min: {df_within['quickflow_cms'].min():.2f}, "
              f"max: {df_within['quickflow_cms'].max():.2f}, "
              f"mean: {df_within['quickflow_cms'].mean():.2f}")

    print("\nBeyond 7-day window:")
    print(f"  Days to prior event - min: {df_beyond['days_to_prior_event'].min():.0f}, "
          f"max: {df_beyond['days_to_prior_event'].max():.0f}, "
          f"mean: {df_beyond['days_to_prior_event'].mean():.1f}")
    print(f"  Antecedent precip (prism_7d_tot) - min: {df_beyond['prism_7d_tot'].min():.1f}, "
          f"max: {df_beyond['prism_7d_tot'].max():.1f}, "
          f"mean: {df_beyond['prism_7d_tot'].mean():.1f}")
    if df_beyond["quickflow_cms"].notna().any():
        print(f"  Quickflow (cms) - min: {df_beyond['quickflow_cms'].min():.2f}, "
              f"max: {df_beyond['quickflow_cms'].max():.2f}, "
              f"mean: {df_beyond['quickflow_cms'].mean():.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True, facecolor="white")

    # Left: prior event within DAYS_THRESHOLD days
    ax1.scatter(
        df_within["prism_7d_tot"],
        df_within["discharge_cms"],
        c=df_within["days_to_prior_event"],
        s=60,
        alpha=0.6,
        cmap="viridis",
        edgecolors="none",
        vmin=0,
        vmax=DAYS_THRESHOLD,
    )
    ax1.set_xlabel("7-Day Antecedent Precipitation (mm)", fontsize=14, labelpad=10)
    ax1.set_ylabel("Discharge (cms)", fontsize=14, labelpad=10)
    ax1.set_title(f"Days to Prior Event ≤ {DAYS_THRESHOLD} (n={len(df_within)})", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Right: prior event more than DAYS_THRESHOLD days earlier
    ax2.scatter(
        df_beyond["prism_7d_tot"],
        df_beyond["discharge_cms"],
        c=df_beyond["days_to_prior_event"],
        s=60,
        alpha=0.6,
        cmap="viridis",
        edgecolors="none",
        vmin=0,
        vmax=df_clean["days_to_prior_event"].max(),
    )
    ax2.set_xlabel("7-Day Antecedent Precipitation (mm)", fontsize=14, labelpad=10)
    ax2.set_title(f"Days to Prior Event > {DAYS_THRESHOLD} (n={len(df_beyond)})", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Streamflow vs Antecedent Precipitation by Days to Previous Event",
                 fontsize=16, fontweight="bold")

    plt.tight_layout()

    out_img = OUT_DIR / "9.75_flow_vs_days_to_prior_event.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {out_img}")
    plt.close()

    # Summary statistics by antecedent precipitation quintile
    print("\n=== Summary Statistics ===")
    print(f"Total points plotted (within 7 days): {len(df_within)}")
    print(f"Total points plotted (beyond 7 days): {len(df_beyond)}")

    print("\nQuickflow by antecedent precipitation quintiles (Days to Prior ≤ 5):")
    df_within["precip_quintile"] = pd.qcut(df_within["prism_7d_tot"], q=5, labels=QUINTILE_LABELS, duplicates="drop")
    for quintile in df_within["precip_quintile"].unique():
        subset = df_within[df_within["precip_quintile"] == quintile]
        print(f"\n{quintile}:")
        print(f"  Antecedent precip range: {subset['prism_7d_tot'].min():.1f} - {subset['prism_7d_tot'].max():.1f} mm")
        if subset["quickflow_cms"].notna().any():
            print(f"  Quickflow - mean: {subset['quickflow_cms'].mean():.2f}, median: {subset['quickflow_cms'].median():.2f}, max: {subset['quickflow_cms'].max():.2f}")
        print(f"  Days to prior - mean: {subset['days_to_prior_event'].mean():.1f}, median: {subset['days_to_prior_event'].median():.1f}")

    print("\n\nQuickflow by antecedent precipitation quintiles (Days to Prior > 5):")
    df_beyond["precip_quintile"] = pd.qcut(df_beyond["prism_7d_tot"], q=5, labels=QUINTILE_LABELS, duplicates="drop")
    for quintile in df_beyond["precip_quintile"].unique():
        subset = df_beyond[df_beyond["precip_quintile"] == quintile]
        print(f"\n{quintile}:")
        print(f"  Antecedent precip range: {subset['prism_7d_tot'].min():.1f} - {subset['prism_7d_tot'].max():.1f} mm")
        if subset["quickflow_cms"].notna().any():
            print(f"  Quickflow - mean: {subset['quickflow_cms'].mean():.2f}, median: {subset['quickflow_cms'].median():.2f}, max: {subset['quickflow_cms'].max():.2f}")
        print(f"  Days to prior - mean: {subset['days_to_prior_event'].mean():.1f}, median: {subset['days_to_prior_event'].median():.1f}")


if __name__ == "__main__":
    plot_flow_vs_days_to_prior_event()
    print("\nDone!")
