"""Discharge vs 7-day antecedent precipitation by AR status of event pairs.

2x2 grid of scatter plots (shared axes) for events whose previous event
occurred within 7 days, split by the atmospheric river (AR) status of the
current and the previous event:

    top left     : AR event after an AR event
    top right    : AR event after a non-AR event
    bottom left  : non-AR event after an AR event
    bottom right : non-AR event after a non-AR event

    x-axis : 7-day PRISM antecedent precipitation (``prism_7d_tot``)
    y-axis : USGS 12200500 discharge (cms)
    color  : days to previous event (``days_to_prior_event``)

Summary statistics per category are printed to the console.

Input
    analysis/outputs/5day_taking_min_and_filling/3.75_events_with_prior_counts.csv
Output
    plots/bias_boxplots/plots/9.76_flow_vs_precip_ar_categories.png

Usage
    python 9.76_plot_flow_vs_precip_ar_categories.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "analysis" / "outputs" / "5day_taking_min_and_filling" / "3.75_events_with_prior_counts.csv"
OUT_DIR = BASE_DIR / "plots" / "bias_boxplots" / "plots"

CFS_TO_CMS = 0.0283168
MAX_DAYS_TO_PRIOR = 7  # keep events whose previous event is at most this many days earlier

# (current event is AR, previous event is AR, panel label), in panel order
CATEGORIES = [
    (True, True, "AR with previous AR"),
    (True, False, "AR with previous non-AR"),
    (False, True, "non-AR with previous AR"),
    (False, False, "non-AR with previous non-AR"),
]


def plot_flow_vs_precip_by_ar_categories():
    """Create 2x2 grid of scatter plots by AR status."""

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df["discharge_cms"] = df["discharge_cfs"] * CFS_TO_CMS

    # Add the previous event's AR scale (events sorted by date)
    df = df.sort_values("date").reset_index(drop=True)
    df["prev_ar_scale"] = df["ar_scale"].shift(1)

    # Keep events that have a previous event within MAX_DAYS_TO_PRIOR days
    df_clean = df.dropna(subset=["days_to_prior_event", "prev_ar_scale"])
    df_clean = df_clean[df_clean["days_to_prior_event"] <= MAX_DAYS_TO_PRIOR].copy()

    df_clean["current_is_ar"] = df_clean["ar_scale"] > 0
    df_clean["prev_is_ar"] = df_clean["prev_ar_scale"] > 0

    print(f"Total events: {len(df)}")
    print(f"Events with prior event within 7-day window: {len(df_clean)}")
    print("\nEvent counts by category:")
    for current_ar, prev_ar, label in CATEGORIES:
        count = len(df_clean[(df_clean["current_is_ar"] == current_ar) & (df_clean["prev_is_ar"] == prev_ar)])
        print(f"  {label}: {count}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor="white", sharex=True, sharey=True)
    fig.suptitle("Streamflow vs Antecedent Precipitation by AR Status\nColored by Days to Previous Event",
                 fontsize=16, fontweight="bold", y=0.995)

    last_scatter = None
    for (current_ar, prev_ar, label), ax in zip(CATEGORIES, axes.flat):
        mask = (df_clean["current_is_ar"] == current_ar) & (df_clean["prev_is_ar"] == prev_ar)
        subset = df_clean[mask]

        if len(subset) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(label, fontsize=13, fontweight="bold")
            continue

        last_scatter = ax.scatter(
            subset["prism_7d_tot"],
            subset["discharge_cms"],
            c=subset["days_to_prior_event"],
            s=60,
            alpha=0.6,
            cmap="viridis",
            edgecolors="none",
            vmin=0,
            vmax=MAX_DAYS_TO_PRIOR,
        )

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_title(f"{label} (n={len(subset)})", fontsize=13, fontweight="bold")

    # Axis labels on outer edges only
    for row in range(2):
        axes[row, 0].set_ylabel("Discharge (cms)", fontsize=12, labelpad=10)
    for col in range(2):
        axes[1, col].set_xlabel("7-Day Antecedent Precipitation (mm)", fontsize=12, labelpad=10)

    # Single colorbar for all subplots (all panels share vmin/vmax)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_scatter, cax=cbar_ax)
    cbar.set_label("Days to Previous Event", fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 0.91, 0.99])

    out_img = OUT_DIR / "9.76_flow_vs_precip_ar_categories.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {out_img}")
    plt.close()

    # Summary statistics by category
    print("\n=== Summary Statistics by Category ===")
    for current_ar, prev_ar, label in CATEGORIES:
        mask = (df_clean["current_is_ar"] == current_ar) & (df_clean["prev_is_ar"] == prev_ar)
        subset = df_clean[mask]

        if len(subset) == 0:
            print(f"\n{label}: No data")
            continue

        print(f"\n{label} (n={len(subset)}):")
        print(f"  Discharge - mean: {subset['discharge_cms'].mean():.1f}, "
              f"median: {subset['discharge_cms'].median():.1f}, "
              f"max: {subset['discharge_cms'].max():.1f}")
        print(f"  Antecedent precip - mean: {subset['prism_7d_tot'].mean():.1f}, "
              f"median: {subset['prism_7d_tot'].median():.1f}, "
              f"max: {subset['prism_7d_tot'].max():.1f}")
        print(f"  Days to prior - mean: {subset['days_to_prior_event'].mean():.1f}, "
              f"median: {subset['days_to_prior_event'].median():.1f}")


if __name__ == "__main__":
    plot_flow_vs_precip_by_ar_categories()
    print("\nDone!")
