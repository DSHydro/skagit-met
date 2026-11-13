"""Quickflow and streamflow of compounding events, by individual day since previous event.

Daily-resolution companion to ``boxplots_compounding.py``. Events are
classified by the atmospheric river (AR) status of the event and of the event
that preceded it, giving three compounding scenarios:

    AR -> AR         : AR event preceded by an AR event
    non-AR -> AR     : AR event preceded by a non-AR event
    AR -> non-AR     : non-AR event preceded by an AR event

A non-AR event sandwiched between two AR events is assigned to only one pair,
the one with the closer AR event, so it is not counted twice.

For each scenario, quickflow (direct runoff from baseflow separation of the
USGS 12200500 discharge) and streamflow are shown as boxplots with one box
per number of days since the previous event (no binning). The number of
events for each day is printed at the bottom of each box. One 1x3 figure is
made per variable.

Input
    analysis/outputs/0day_taking_min_and_filling/3.75_events_with_prior_counts.csv
    ../data/baseflow_separation/usgs_12200500_discharge_with_baseflow.csv
Output
    plots/compounding_events/plots/daily_boxplot_compounding_scenarios_quickflow.png
    plots/compounding_events/plots/daily_boxplot_compounding_scenarios_streamflow.png

Usage
    python boxplots_compounding_daily.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "analysis" / "outputs" / "0day_taking_min_and_filling" / "3.75_events_with_prior_counts.csv"
QUICKFLOW_CSV = BASE_DIR.parent / "data" / "baseflow_separation" / "usgs_12200500_discharge_with_baseflow.csv"
OUT_DIR = BASE_DIR / "plots" / "compounding_events" / "plots"

# Font sizes
FONT_TICK = 8  # x-axis tick labels (one per day)
FONT_COUNT = 7  # event counts at the bottom of each box

def load_discharge_data():
    """Load quickflow and streamflow from baseflow-separated discharge data."""
    print(f"Loading discharge data from {QUICKFLOW_CSV}...")
    df = pd.read_csv(QUICKFLOW_CSV, parse_dates=['date'])
    print(f"  Loaded {len(df)} days of discharge data")
    result = df[['date', 'direct_runoff_cms', 'discharge_cms']].rename(
        columns={'direct_runoff_cms': 'quickflow_cms', 'discharge_cms': 'streamflow_cms'})
    return result

print(f"Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, parse_dates=['date'])

# Load and merge discharge data
discharge_df = load_discharge_data()
df = pd.merge(df, discharge_df, on='date', how='left')
print(f"  Merged discharge data for {df['quickflow_cms'].notna().sum()} events")

# Remove rows with missing data
df_clean = df.dropna(subset=['prism_7d_tot', 'quickflow_cms', 'streamflow_cms', 'events_in_prior_7days'])

print(f"Total events with complete data: {len(df_clean)}")
print("\nData ranges:")
print(f"  7-day antecedent precip (prism_7d_tot) - min: {df_clean['prism_7d_tot'].min():.1f}, "
        f"max: {df_clean['prism_7d_tot'].max():.1f}, "
        f"mean: {df_clean['prism_7d_tot'].mean():.1f}")
print(f"  Quickflow (cms) - min: {df_clean['quickflow_cms'].min():.2f}, "
        f"max: {df_clean['quickflow_cms'].max():.2f}, "
        f"mean: {df_clean['quickflow_cms'].mean():.2f}")
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
ar_ar = ar_ar.dropna(subset=["gap_days", "quickflow_cms"])
non_ar_ar = non_ar_ar.dropna(subset=["gap_days", "quickflow_cms"])
ar_non_ar = ar_non_ar.dropna(subset=["gap_days", "quickflow_cms"])

print("\nCompounding Event Scenarios:")
print(f"  AR followed by AR: {len(ar_ar)} events")
print(f"  non-AR followed by AR: {len(non_ar_ar)} events")
print(f"  AR followed by non-AR: {len(ar_non_ar)} events")

print("\nAR → AR:")
print(f"  n={len(ar_ar)}, median quickflow={ar_ar['quickflow_cms'].median():.2f} cms, median streamflow={ar_ar['streamflow_cms'].median():.2f} cms")

print("\nnon-AR → AR:")
print(f"  n={len(non_ar_ar)}, median quickflow={non_ar_ar['quickflow_cms'].median():.2f} cms, median streamflow={non_ar_ar['streamflow_cms'].median():.2f} cms")

print("\nAR → non-AR:")
print(f"  n={len(ar_non_ar)}, median quickflow={ar_non_ar['quickflow_cms'].median():.2f} cms, median streamflow={ar_non_ar['streamflow_cms'].median():.2f} cms")

def create_boxplot_figure(datasets, y_col, y_label, output_filename):
    """Create boxplot figure for a given variable."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    plot_datasets = [
        (datasets[0], "AR → AR", axes[0]),
        (datasets[1], "non-AR → AR", axes[1]),
        (datasets[2], "AR → non-AR", axes[2])
    ]

    for data, title, ax in plot_datasets:
        if len(data) > 0:
            sns.boxplot(data=data, x="gap_days", y=y_col, palette="Blues_r",
                        legend=False, showfliers=False, ax=ax)

            ax.set_xlabel("Days since previous event")
            ax.set_ylabel(y_label)
            ax.set_title(f"{title} (n={len(data)})")
            ax.tick_params(axis='x', rotation=45, labelsize=FONT_TICK)

            # Add counts above each bin
            counts = data["gap_days"].value_counts().sort_index()
            y_min = ax.get_ylim()[0]
            for i, (day, count) in enumerate(counts.items()):
                ax.text(i, y_min, f"{int(count)}", ha="center", va="bottom", fontsize=FONT_COUNT)

    plt.tight_layout()
    plt.savefig(OUT_DIR / output_filename, dpi=300)
    print(f"\nSaved to {OUT_DIR / output_filename}")

# Create plots for both quickflow and streamflow
create_boxplot_figure([ar_ar, non_ar_ar, ar_non_ar], "quickflow_cms", "Quickflow (cms)",
                     "daily_boxplot_compounding_scenarios_quickflow.png")
create_boxplot_figure([ar_ar, non_ar_ar, ar_non_ar], "streamflow_cms", "Streamflow (cms)",
                     "daily_boxplot_compounding_scenarios_streamflow.png")