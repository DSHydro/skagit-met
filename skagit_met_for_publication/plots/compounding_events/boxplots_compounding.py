"""Streamflow of compounding events, by days since previous event.

Events are classified by the atmospheric river (AR) status of the event and
of the event that preceded it, giving three compounding scenarios:

    AR -> AR         : AR event preceded by an AR event
    non-AR -> AR     : AR event preceded by a non-AR event
    AR -> non-AR     : non-AR event preceded by an AR event

A non-AR event sandwiched between two AR events is assigned to only one pair,
the one with the closer AR event, so it is not counted twice.

For each scenario, streamflow (USGS 12200500 daily mean discharge on the
event day, converted from cfs to m^3/s) is shown as boxplots binned by the
number of days since the previous event (1, 2, 3-5, >=6; AR -> AR uses
1, 2, >=3 because it has almost no events at >=6 days). The number of events
in each bin is printed at the bottom of each box. The figure is 1x3, one
panel per scenario.

Input
    analysis/outputs/0day_taking_min_and_filling/3.75_events_with_prior_counts.csv
Output
    plots/compounding_events/plots/boxplot_compounding_scenarios_streamflow.png

Usage
    python boxplots_compounding.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "analysis" / "outputs" / "0day_taking_min_and_filling" / "3.75_events_with_prior_counts.csv"
OUT_DIR = BASE_DIR / "plots" / "compounding_events" / "plots"

CFS_TO_CMS = 0.0283168   # 1 ft^3/s = 0.0283168 m^3/s (same factor as analysis/5_preprocess_products.py)

# Common gap-day bins (1, 2, 3–5, ≥6) chosen so each box holds a similar number of events.
# Day 1 is never split, so it stays its own bin. AR → AR has almost no events at ≥6 days,
# so its last two bins are merged into ≥3.
GAP_BINS = [0, 1, 2, 5, np.inf]
GAP_LABELS = ["1", "2", "3–5", "≥6"]
GAP_BINS_AR_AR = [0, 1, 2, np.inf]
GAP_LABELS_AR_AR = ["1", "2", "≥3"]

# Publication font sizes (figure is 15 in wide; expect ~0.45x reduction to journal page width)
FONT_TITLE = 18
FONT_LABEL = 16
FONT_TICK = 15
FONT_COUNT = 13  # event counts at the bottom of each box

print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

# Streamflow on the event day (USGS 12200500 discharge, already in the event table)
df['streamflow_cms'] = df['discharge_cfs'] * CFS_TO_CMS

# Remove rows with missing data
df_clean = df.dropna(subset=['prism_7d_tot', 'streamflow_cms', 'events_in_prior_7days'])

print(f"Total events with complete data: {len(df_clean)}")
print("\nData ranges:")
print(f"  7-day antecedent precip (prism_7d_tot) - min: {df_clean['prism_7d_tot'].min():.1f}, "
        f"max: {df_clean['prism_7d_tot'].max():.1f}, "
        f"mean: {df_clean['prism_7d_tot'].mean():.1f}")
print(f"  Streamflow (cms) - min: {df_clean['streamflow_cms'].min():.2f}, "
        f"max: {df_clean['streamflow_cms'].max():.2f}, "
        f"mean: {df_clean['streamflow_cms'].mean():.2f}")
print(f"  Events in prior 7 days - min: {df_clean['events_in_prior_7days'].min():.0f}, "
        f"max: {df_clean['events_in_prior_7days'].max():.0f}, "
        f"mean: {df_clean['events_in_prior_7days'].mean():.1f}")

d = df_clean.sort_values("date").reset_index(drop=True)

# Track if each event is an AR event or not
d["is_ar"] = d["ar_scale"] > 0

# For each event, determine what type of event preceded it
d["prev_is_ar"] = d["is_ar"].shift(1)
d["gap_days"] = d["date"].diff().dt.days

# Calculate distance to next event
d["next_gap_days"] = (d["date"].shift(-1) - d["date"]).dt.days

# For non-AR events sandwiched between two ARs, decide which AR pair to keep based on closest distance
d["exclude_ar_non_ar"] = False
d["exclude_non_ar_ar"] = False

for i in range(len(d)):
    if d.loc[i, "is_ar"] == False and i > 0 and i < len(d) - 1:
        prev_is_ar = d.loc[i - 1, "is_ar"]
        next_is_ar = d.loc[i + 1, "is_ar"]

        if prev_is_ar and next_is_ar:
            prev_gap = d.loc[i, "gap_days"]
            next_gap = d.loc[i, "next_gap_days"]

            if next_gap < prev_gap:
                # Next AR is closer - exclude non-AR from "AR followed by non-AR"
                d.loc[i, "exclude_ar_non_ar"] = True
            else:
                # Previous AR is closer - exclude next AR from "non-AR followed by AR"
                d.loc[i + 1, "exclude_non_ar_ar"] = True

# Create three scenarios
ar_ar = d[(d["is_ar"]) & (d["prev_is_ar"] == True)].copy()  # AR followed by AR
non_ar_ar = d[(d["is_ar"]) & (d["prev_is_ar"] == False) & (~d["exclude_non_ar_ar"])].copy()  # non-AR followed by AR
ar_non_ar = d[(d["is_ar"] == False) & (d["prev_is_ar"] == True) & (~d["exclude_ar_non_ar"])].copy()  # AR followed by non-AR

# Clean data
ar_ar = ar_ar.dropna(subset=["gap_days", "streamflow_cms"])
non_ar_ar = non_ar_ar.dropna(subset=["gap_days", "streamflow_cms"])
ar_non_ar = ar_non_ar.dropna(subset=["gap_days", "streamflow_cms"])

def bin_gap_days(gaps, bins, labels):
    """Assign gap_days to right-closed bins, e.g. (2, 5] -> '3–5'."""
    return pd.cut(gaps, bins=bins, labels=labels).astype(str)

ar_ar["gap_bin"] = bin_gap_days(ar_ar["gap_days"], GAP_BINS_AR_AR, GAP_LABELS_AR_AR)
non_ar_ar["gap_bin"] = bin_gap_days(non_ar_ar["gap_days"], GAP_BINS, GAP_LABELS)
ar_non_ar["gap_bin"] = bin_gap_days(ar_non_ar["gap_days"], GAP_BINS, GAP_LABELS)

for name, sub in [("AR → AR", ar_ar), ("non-AR → AR", non_ar_ar), ("AR → non-AR", ar_non_ar)]:
    print(f"  {name} events per bin: {sub['gap_bin'].value_counts().to_dict()}")

print("\nCompounding Event Scenarios:")
print(f"  AR followed by AR: {len(ar_ar)} events")
print(f"  non-AR followed by AR: {len(non_ar_ar)} events")
print(f"  AR followed by non-AR: {len(ar_non_ar)} events")

print("\nAR → AR:")
print(f"  n={len(ar_ar)}, median streamflow={ar_ar['streamflow_cms'].median():.2f} cms")

print("\nnon-AR → AR:")
print(f"  n={len(non_ar_ar)}, median streamflow={non_ar_ar['streamflow_cms'].median():.2f} cms")

print("\nAR → non-AR:")
print(f"  n={len(ar_non_ar)}, median streamflow={ar_non_ar['streamflow_cms'].median():.2f} cms")

def create_boxplot_figure(datasets, y_col, y_label, output_filename, bin_orders):
    """Create boxplot figure for a given variable with custom bin ordering."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    plot_datasets = [
        (datasets[0], "AR → AR", axes[0], bin_orders[0]),
        (datasets[1], "non-AR → AR", axes[1], bin_orders[1]),
        (datasets[2], "AR → non-AR", axes[2], bin_orders[2])
    ]

    for data, title, ax, bin_order in plot_datasets:
        if len(data) > 0:
            sns.boxplot(data=data, x="gap_bin", y=y_col, palette="Blues_r",
                        legend=False, showfliers=False, ax=ax, order=bin_order)

            ax.set_xlabel("Days since previous event", fontsize=FONT_LABEL)
            # y-axis is shared, so label it only on the left panel
            ax.set_ylabel(y_label if ax is axes[0] else "", fontsize=FONT_LABEL)
            ax.set_title(f"{title} (n={len(data)})", fontsize=FONT_TITLE, fontweight='bold')
            ax.tick_params(axis='both', labelsize=FONT_TICK)

            # Add counts above each bin
            counts = data["gap_bin"].value_counts().reindex(bin_order, fill_value=0)
            for i, count in enumerate(counts):
                if count > 0:
                    ax.text(i, 0, f"{int(count)}", ha="center", va="bottom", fontsize=FONT_COUNT)

    # y-axis is shared, so this sets all three panels to start at 0
    axes[0].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / output_filename, dpi=300)
    print(f"\nSaved to {OUT_DIR / output_filename}")

# Define bin orders for each scenario
bin_orders = [
    GAP_LABELS_AR_AR,  # AR → AR
    GAP_LABELS,        # non-AR → AR
    GAP_LABELS,        # AR → non-AR
]

create_boxplot_figure([ar_ar, non_ar_ar, ar_non_ar], "streamflow_cms", "Streamflow (cms)",
                     "boxplot_compounding_scenarios_streamflow.png", bin_orders)