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

# Exclude events where days to prior event > 10
ar_ar = ar_ar[ar_ar["gap_days"] <= 10]
non_ar_ar = non_ar_ar[non_ar_ar["gap_days"] <= 10]
ar_non_ar = ar_non_ar[ar_non_ar["gap_days"] <= 10]

def bin_gap_days_ar_ar(gap):
    """Bin gap_days for AR→AR: 1+2, 3, 4, 5+6, 7+8+9, >10"""
    if pd.isna(gap):
        return np.nan
    gap = int(gap)
    if gap in [1, 2]:
        return "1-2"
    elif gap == 3:
        return "3"
    elif gap == 4:
        return "4"
    elif gap in [5, 6]:
        return "5-6"
    elif gap in [7, 8, 9]:
        return "7-9"
    else:
        return ">10"

def bin_gap_days_non_ar_ar(gap):
    """Bin gap_days for non-AR→AR: 6+7, >8"""
    if pd.isna(gap):
        return np.nan
    gap = int(gap)
    if gap in [6, 7]:
        return "6-7"
    else:
        return ">8"

def bin_gap_days_ar_non_ar(gap):
    """Bin gap_days for AR→non-AR: 6, >7"""
    if pd.isna(gap):
        return np.nan
    gap = int(gap)
    if gap == 6:
        return "6"
    else:
        return ">7"

ar_ar["gap_bin"] = ar_ar["gap_days"].apply(bin_gap_days_ar_ar)
non_ar_ar["gap_bin"] = non_ar_ar["gap_days"].apply(bin_gap_days_non_ar_ar)
ar_non_ar["gap_bin"] = ar_non_ar["gap_days"].apply(bin_gap_days_ar_non_ar)

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

def create_ar_ar_scatter(data, y_col, y_label, output_filename):
    """Create scatterplot for AR→AR grouped by gap_days with trend line per group."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique gap_days and assign colors
    unique_gaps = sorted(data["gap_days"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_gaps)))
    gap_colors = {gap: colors[i] for i, gap in enumerate(unique_gaps)}

    # Plot each gap_days group with its own trend line
    for gap in unique_gaps:
        group_data = data[data["gap_days"] == gap]
        color = gap_colors[gap]

        # Scatter plot for this group
        ax.scatter(group_data["prism_7d_tot"], group_data[y_col],
                  c=[color], alpha=0.7, s=80, edgecolors="black", linewidth=0.5,
                  label=f"{int(gap)} days (n={len(group_data)})")

        # Trend line for this group (if more than 1 point)
        if len(group_data) > 1:
            z = np.polyfit(group_data["prism_7d_tot"], group_data[y_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(group_data["prism_7d_tot"].min(), group_data["prism_7d_tot"].max(), 100)
            ax.plot(x_trend, p(x_trend), color=color, linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("7-day antecedent precipitation (mm)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"AR → AR (n={len(data)})", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, output_filename), dpi=300)
    print(f"Saved to {os.path.join(OUT_DIR, output_filename)}")

# Create scatter plots for AR→AR with quickflow and streamflow
create_ar_ar_scatter(ar_ar, "quickflow_cms", "Quickflow (cms)",
                     "scatter_ar_ar_quickflow.png")
create_ar_ar_scatter(ar_ar, "streamflow_cms", "Streamflow (cms)",
                     "scatter_ar_ar_streamflow.png")