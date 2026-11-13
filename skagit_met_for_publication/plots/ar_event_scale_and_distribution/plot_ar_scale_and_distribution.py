"""Atmospheric river (AR) event frequency and AR-scale distribution figures.

Produces two stacked bar charts from the event table, using only events in
water years 1982-2024 (1 Oct 1981 - 30 Sep 2024; a water year runs from
1 Oct of the previous calendar year to 30 Sep):

1. Average frequency of AR events (scale 1-5) per water-year month
   (Oct-Sep), stacked by AR scale. Monthly counts are divided by the number
   of water years.
2. Percentage of events of each AR scale (0-5, 0 = non-AR) within each
   streamflow bucket (< 600, 600-1200, 1200-1700, > 1700 cms; USGS 12200500).

Summary tables are printed to the console.

Converted from the former ``figures.ipynb`` notebook in this folder.

Input
    analysis/outputs/5day_taking_min_and_filling/3_multi_product_bulk_bias_data.csv
Output
    plots/ar_event_scale_and_distribution/plots/ar_frequency_by_water_year_month.png
    plots/ar_event_scale_and_distribution/plots/ar_scale_by_streamflow_bucket.png

Usage
    python plot_ar_scale_and_distribution.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
EVENTS_CSV = BASE_DIR / "analysis" / "outputs" / "5day_taking_min_and_filling" / "3_multi_product_bulk_bias_data.csv"
OUT_DIR = Path(__file__).resolve().parent / "plots"

# Study period in water years (WY N = 1 Oct of year N-1 to 30 Sep of year N)
WY_START, WY_END = 1982, 2024

# Water-year months in order (Oct = 1, ..., Sep = 12)
WATER_YEAR_MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

# Colors for AR scales 1-5 in figure 1 (taken from the reversed inferno colormap)
AR_SCALE_COLORS = {1.0: "#fb9b06", 2.0: "#e05536", 3.0: "#a32c61", 4.0: "#5f136e", 5.0: "#160b39"}

# Streamflow buckets (cms), left-closed: [0, 600), [600, 1200), ...
CFS_PER_CMS = 35.3147
STREAMFLOW_BINS = [0, 600, 1200, 1700, float("inf")]
STREAMFLOW_LABELS = ["< 600", "600 - 1200", "1200 - 1700", "> 1700"]

# Figure styling
FIGSIZE = (14, 8)
BAR_WIDTH = 0.6
DPI = 300
FONT_TITLE = 20
FONT_LABEL = 18
FONT_TICK = 16
FONT_LEGEND = 15
FONT_LEGEND_TITLE = 16

OUT_DIR.mkdir(parents=True, exist_ok=True)

events = pd.read_csv(EVENTS_CSV, parse_dates=["date"])

# Assign each event to its water year (Oct-Dec belong to the next calendar year's WY)
# and keep only the study period.
events["water_year"] = events["date"].dt.year + (events["date"].dt.month >= 10)
events = events[events["water_year"].between(WY_START, WY_END)].copy()

# --------------------------------------------------------------------------- #
# Figure 1: AR event frequency by water-year month and scale
# --------------------------------------------------------------------------- #

# Keep AR events only (exclude scale 0)
events_filtered = events[events["ar_scale"] > 0].copy()

# Extract month and water-year month (Oct=1, Nov=2, ..., Sep=12)
events_filtered["month"] = events_filtered["date"].dt.month
events_filtered["water_year_month"] = events_filtered["month"].apply(
    lambda m: m - 9 if m >= 10 else m + 3
)

# Group by water-year month and ar_scale to get frequencies
freq_by_month_scale = events_filtered.groupby(["water_year_month", "ar_scale"]).size().reset_index(name="count")

# Divide by the number of water years to get the average frequency
num_water_years = WY_END - WY_START + 1
freq_by_month_scale["frequency"] = freq_by_month_scale["count"] / num_water_years

# Pivot table for the stacked bar chart
pivot_data = freq_by_month_scale.pivot(index="water_year_month", columns="ar_scale", values="frequency").fillna(0)

print(f"Number of water years: {num_water_years}")
print("\nAverage frequency of AR events (scale > 0) by water-year month and scale:")
print(pivot_data)
print(f"\nTotal AR events (scale > 0): {len(events_filtered)}")

x = np.arange(len(WATER_YEAR_MONTHS))
fig, ax = plt.subplots(figsize=FIGSIZE)

# Stacked bars, one layer per AR scale
bottom = np.zeros(len(WATER_YEAR_MONTHS))
for scale in sorted(pivot_data.columns):
    values = pivot_data[scale].values
    color = AR_SCALE_COLORS.get(scale, "#999999")
    ax.bar(x, values, BAR_WIDTH, label=f"Scale {int(scale)}", bottom=bottom, color=color, edgecolor="white", linewidth=1)
    bottom += values

ax.set_xlabel("Water Year Month", fontsize=FONT_LABEL, fontweight="bold")
ax.set_ylabel("Average Frequency (events per month)", fontsize=FONT_LABEL, fontweight="bold")
ax.set_title(f"Average Frequency of Atmospheric River Events by Water Year Month\n(WY {WY_START} - {WY_END})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
ax.set_xticks(x)
ax.set_xticklabels(WATER_YEAR_MONTHS, fontsize=FONT_TICK)
ax.tick_params(axis="y", labelsize=FONT_TICK)
ax.legend(title="AR Scale", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND_TITLE, loc="upper right")
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUT_DIR / "ar_frequency_by_water_year_month.png", dpi=DPI, bbox_inches="tight")
plt.close()

print("\nSummary statistics:")
print(f"Total water years: {num_water_years}")
total_events = pivot_data.sum().sum() * num_water_years
print(f"Total AR events (scale > 0): {int(total_events)}")
for scale in sorted(pivot_data.columns):
    total_for_scale = int(pivot_data[scale].sum() * num_water_years)
    avg_for_scale = pivot_data[scale].sum()
    print(f"Scale {int(scale)}: {total_for_scale} total events (avg {avg_for_scale:.2f} per month)")

# --------------------------------------------------------------------------- #
# Figure 2: AR scale distribution by streamflow bucket
# --------------------------------------------------------------------------- #

events["discharge_cms"] = events["discharge_cfs"] / CFS_PER_CMS

# Assign each event to a streamflow bucket
events["streamflow_bucket"] = pd.cut(events["discharge_cms"], bins=STREAMFLOW_BINS, labels=STREAMFLOW_LABELS, right=False)

# Count events per (bucket, AR scale) and per bucket
sf_ar_counts = events.groupby(["streamflow_bucket", "ar_scale"], observed=False).size().reset_index(name="count")
bucket_totals = events.groupby("streamflow_bucket", observed=False).size()

# Percentage of each AR scale within its bucket
sf_ar_counts["bucket_total"] = sf_ar_counts["streamflow_bucket"].map(bucket_totals).astype(float)
sf_ar_counts["percentage"] = (sf_ar_counts["count"] / sf_ar_counts["bucket_total"] * 100).astype(float)

pivot_percentages = sf_ar_counts.pivot(index="streamflow_bucket", columns="ar_scale", values="percentage").fillna(0)

print("Percentage of AR scales within each streamflow bucket:")
print(pivot_percentages)
print("\nTotal events per streamflow bucket:")
print(bucket_totals)

fig, ax = plt.subplots(figsize=FIGSIZE)
x = np.arange(len(STREAMFLOW_LABELS))

# Colors from the reversed inferno colormap:
# scale 0 (lowest) = bright yellow, scale 5 (highest) = dark violet
cmap = plt.cm.inferno
scales_list = sorted(pivot_percentages.columns)
n_scales = len(scales_list)
color_map = {scale: to_hex(cmap(0.95 - (i / (n_scales - 1)) * 0.85)) for i, scale in enumerate(scales_list)}

print("Color codes from inferno colormap (reversed):")
for scale, color in sorted(color_map.items()):
    print(f"Scale {int(scale)}: {color}")

# Stacked bars, one layer per AR scale
bottom = np.zeros(len(STREAMFLOW_LABELS))
for scale in scales_list:
    values = pivot_percentages[scale].values
    ax.bar(x, values, BAR_WIDTH, label=f"Scale {int(scale)}", bottom=bottom, color=color_map[scale], edgecolor="white", linewidth=1.5)
    bottom += values

ax.set_xlabel("Streamflow Bucket (cms)", fontsize=FONT_LABEL, fontweight="bold")
ax.set_ylabel("Percentage (%)", fontsize=FONT_LABEL, fontweight="bold")
ax.set_title(f"Distribution of AR Scales Across Streamflow Buckets\n(WY {WY_START} - {WY_END})", fontsize=FONT_TITLE, fontweight="bold", pad=20)
ax.set_xticks(x)
ax.set_xticklabels(STREAMFLOW_LABELS, fontsize=FONT_TICK)
ax.tick_params(axis="y", labelsize=FONT_TICK)
ax.legend(title="AR Scale", fontsize=FONT_LEGEND, title_fontsize=FONT_LEGEND_TITLE, loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUT_DIR / "ar_scale_by_streamflow_bucket.png", dpi=DPI, bbox_inches="tight")
plt.close()

print("\nStreamflow bucket statistics:")
for bucket in STREAMFLOW_LABELS:
    count = bucket_totals[bucket] if bucket in bucket_totals.index else 0
    print(f"{bucket} cms: {int(count)} events")
