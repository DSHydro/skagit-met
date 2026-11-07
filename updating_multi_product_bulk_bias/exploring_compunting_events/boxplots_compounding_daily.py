import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal

BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/updating_multi_product_bulk_bias"
INPUT_CSV = os.path.join(BASE_DIR, "outputs/3.75_events_with_prior_counts.csv")
QUICKFLOW_CSV = Path("/data0/hernanqd/plots_code/skagit-met/data/baseflow_separation/usgs_12200500_discharge_with_baseflow.csv")
OUT_DIR = Path("/data0/hernanqd/plots_code/skagit-met/updating_multi_product_bulk_bias/exploring_compunting_events")

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
if 'quickflow_cms' in df.columns:
    print(f"  Merged discharge data for {df['quickflow_cms'].notna().sum()} events")

# Remove rows with missing data
df_clean = df.dropna(subset=['prism_7d_tot', 'quickflow_cms', 'streamflow_cms', 'events_in_prior_7days'])

print(f"Total events with complete data: {len(df_clean)}")
print(f"\nData ranges:")
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
d["date"] = pd.to_datetime(d["date"])

# Track if each event is an AR event or not
d["is_ar"] = d["ar_scale"] > 0

# For each event, determine what type of event preceded it
d["prev_is_ar"] = d["is_ar"].shift(1)
d["next_is_ar"] = d["is_ar"].shift(-1)
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

print(f"\nCompounding Event Scenarios:")
print(f"  AR followed by AR: {len(ar_ar)} events")
print(f"  non-AR followed by AR: {len(non_ar_ar)} events")
print(f"  AR followed by non-AR: {len(ar_non_ar)} events")

print(f"\nAR → AR:")
print(f"  n={len(ar_ar)}, median quickflow={ar_ar['quickflow_cms'].median():.2f} cms, median streamflow={ar_ar['streamflow_cms'].median():.2f} cms")

print(f"\nnon-AR → AR:")
print(f"  n={len(non_ar_ar)}, median quickflow={non_ar_ar['quickflow_cms'].median():.2f} cms, median streamflow={non_ar_ar['streamflow_cms'].median():.2f} cms")

print(f"\nAR → non-AR:")
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
            ax.tick_params(axis='x', rotation=45, labelsize=8)

            # Add counts above each bin
            counts = data["gap_days"].value_counts().sort_index()
            y_min = ax.get_ylim()[0]
            for i, (day, count) in enumerate(counts.items()):
                ax.text(i, y_min, f"{int(count)}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, output_filename), dpi=300)
    print(f"\nSaved to {os.path.join(OUT_DIR, output_filename)}")

# Create plots for both quickflow and streamflow
create_boxplot_figure([ar_ar, non_ar_ar, ar_non_ar], "quickflow_cms", "Quickflow (cms)",
                     "daily_boxplot_compounding_scenarios_quickflow.png")
create_boxplot_figure([ar_ar, non_ar_ar, ar_non_ar], "streamflow_cms", "Streamflow (cms)",
                     "daily_boxplot_compounding_scenarios_streamflow.png")