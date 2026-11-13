"""Cross-product precipitation bias relative to PRISM (manuscript figure).

Builds a 2x2 panel of grouped boxplots showing the 3-day total precipitation
bias (product - PRISM) of five gridded products over 1981-2020:

    A) Bias by atmospheric river (AR) scale (AR events only, scale 1-5)
    B) Bias by PRISM precipitation intensity (non-AR events only)
    C) Bias by streamflow intensity (AR and non-AR events)
    D) Bias by PRISM precipitation intensity (AR and non-AR events)

The 3-day window is the event day plus the two prior days. Boxes hide
outliers; product/bin combinations with fewer than 3 events are additionally
drawn as individual points annotated with their count. X-axis tick labels
report the number of unique events in each bin. All panels share a common
y-axis range.

Input
    analysis/outputs/5day_taking_min_and_filling/5_prepared_bias_table.csv
Output
    plots/bias_boxplots/plots/grouped_bias_3_metrics_2_periods.png

Usage
    python 7_corrected_plot_prism_bias.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_CSV = BASE_DIR / "analysis" / "outputs" / "5day_taking_min_and_filling" / "5_prepared_bias_table.csv"
OUT_DIR = BASE_DIR / "plots" / "bias_boxplots" / "plots"
OUT_IMG = OUT_DIR / "grouped_bias_3_metrics_2_periods.png"

# Analysis settings
START_YEAR, END_YEAR = 1981, 2020
AR_SCALE_RANGE = (0, 5)  # valid AR scale values (0 = non-AR event)
MIN_N = 3  # product/bin combinations with fewer events are also drawn as points

PRODUCTS = ["PNNL", "Daymet", "CONUS404", "UCLA", "GridMET"]
PALETTE = {  # ColorBrewer "Dark2"
    "PNNL": "#d95f02",
    "Daymet": "#7570b3",
    "CONUS404": "#e7298a",
    "UCLA": "#66a61e",
    "GridMET": "#e6ab02",
}

# Display order of the categorical x-axis bins
STREAMFLOW_ORDER = ["< 600", "600 - 1200", "1200 - 1700", "> 1700"]
PRECIP_ORDER = ["< 20mm", "20 - 50mm", "50 - 100mm", "> 100mm"]

# Publication font sizes (figure is 14 in wide; expect ~0.5x reduction to journal page width)
FONT_SUPTITLE = 24
FONT_SUBTITLE = 18
FONT_TITLE = 17
FONT_LABEL = 18
FONT_TICK = 15
FONT_LEGEND = 16
FONT_ANNOT = 13  # "N=" labels on sparse bins


def plot_prism_bias():

    # This three functions are used to sort the x-axis categories in a custom order for each metric.
    def custom_sort_ar_scale(values):
        return sorted(values, key=lambda x: float(x))
    
    def custom_sort_streamflow(values):
        order = STREAMFLOW_ORDER
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))

    def custom_sort_precipitation(values):
        order = PRECIP_ORDER
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))

    print("Loading prepared bias table...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])

    # Keep only valid AR scale values.
    df = df.dropna(subset=["ar_scale"])
    df = df[(df["ar_scale"] >= AR_SCALE_RANGE[0]) & (df["ar_scale"] <= AR_SCALE_RANGE[1])]

    # Melt to long format for plotting.
    melted = df.melt(
        id_vars=["date", "ar_scale", "Streamflow Bucket (cms)", "Precipitation Bucket (mm)"],
        value_vars=PRODUCTS,
        var_name="Product",
        value_name="Bias",
    )
    melted = melted.dropna(subset=["Bias"])

    # Keep the study period.
    period1 = melted[(melted["date"].dt.year >= START_YEAR) & (melted["date"].dt.year <= END_YEAR)]

    def plot_panel(ax, data, x_col, sort_func):
        x_levels = sort_func(list(dict.fromkeys(data[x_col].astype(str).tolist())))

        sparse_bins = []
        for product in PRODUCTS:
            for x_level in x_levels:
                subset = data[(data[x_col].astype(str) == x_level) & (data["Product"] == product)]
                values = subset["Bias"].to_numpy()
                if 0 < len(values) < MIN_N:
                    sparse_bins.append((product, x_level, values))

        sns.boxplot(
            data=data,
            x=x_col,
            y="Bias",
            hue="Product",
            hue_order=PRODUCTS,
            palette=PALETTE,
            linewidth=1.2,
            ax=ax,
            showfliers=False,
            order=x_levels,
        )

        for product, x_level, values in sparse_bins:
            x_idx = x_levels.index(x_level)
            x_pos = x_idx + 0.18 * (PRODUCTS.index(product) - (len(PRODUCTS) - 1) / 2)
            ax.scatter([x_pos] * len(values), values, color=PALETTE[product], s=24, zorder=3)
            ax.text(x_pos, np.max(values) + 0.5, f"N={len(values)}", ha="center", va="bottom", fontsize=FONT_ANNOT, color="dimgray")

        # Add the number of unique events in each bin to the x-axis labels.
        bucket_counts = []
        for x_level in x_levels:
            count = data[data[x_col].astype(str) == x_level]["date"].nunique()
            bucket_counts.append(f"{x_level}\n(N={count})")
        ax.set_xticks(range(len(bucket_counts)))
        ax.set_xticklabels(bucket_counts)

        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_ylabel("Bias (mm)")

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Arial", "Liberation Sans"],
        "font.size": FONT_LABEL,
        "axes.labelsize": FONT_LABEL,
        "axes.titlesize": FONT_TITLE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    plot_panel(axes[0, 0], period1[period1["ar_scale"] > 0], "ar_scale", custom_sort_ar_scale)
    axes[0, 0].set_title("A) Bias by AR Scale", weight="bold")
    axes[0, 0].set_xlabel("Atmospheric River Scale (1-5)", labelpad=8)

    plot_panel(axes[1, 0], period1, "Streamflow Bucket (cms)", custom_sort_streamflow)
    axes[1, 0].set_title("C) Bias by Streamflow Intensity\nAR and non-AR events", weight="bold")
    axes[1, 0].set_xlabel("Streamflow Range (cms)", labelpad=8)

    plot_panel(axes[0, 1], period1[period1["ar_scale"] == 0], "Precipitation Bucket (mm)", custom_sort_precipitation)
    axes[0, 1].set_title("B) Bias by PRISM Precipitation Intensity\nNon-AR Events", weight="bold")
    axes[0, 1].set_xlabel("PRISM 3-Day Total Precipitation Range (mm)", labelpad=8)

    plot_panel(axes[1, 1], period1, "Precipitation Bucket (mm)", custom_sort_precipitation)
    axes[1, 1].set_title("D) Bias by PRISM Precipitation Intensity\nAR and Non-AR Events", weight="bold")
    axes[1, 1].set_xlabel("PRISM 3-Day Total Precipitation Range (mm)", labelpad=8)

    # One shared legend for the whole figure instead of one per panel.
    handles = [Patch(facecolor=PALETTE[p], edgecolor=PALETTE[p], label=p) for p in PRODUCTS]
    fig.legend(handles=handles, title="Product", title_fontsize=FONT_LEGEND,
                loc="upper center", bbox_to_anchor=(0.5, 0.95),
                ncol=6, frameon=True, edgecolor="#cccccc")

    for ax in axes.flat:
        ax.get_legend().remove()

    plt.suptitle(f"Cross-Product Precipitation Bias Analysis: {START_YEAR}-{END_YEAR}\n(Product - PRISM 3-Day Total)", fontsize=FONT_SUPTITLE, weight="bold", y=1.05)
    fig.text(0.5, 0.95, "3-day window computed as: the day of the event and the prior 2 days", ha="center", fontsize=FONT_SUBTITLE, color="dimgray")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Common y-axis range across all panels.
    y_min = min(ax.get_ylim()[0] for ax in axes.flat)
    y_max = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(y_min, y_max)

    plt.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUT_IMG}")


if __name__ == "__main__":
    plot_prism_bias()
