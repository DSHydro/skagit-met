"""Precipitation bias relative to PRISM for AR vs non-AR events, by prior events.

Builds 2x2 panels of grouped boxplots showing the 7-day total precipitation
bias (product - PRISM) of five gridded products over 1981-2020, comparing
atmospheric river (AR) events (left column) with non-AR events (right column):

    A) Bias by streamflow intensity, AR events
    B) Bias by streamflow intensity, non-AR events
    C) Bias by PRISM precipitation intensity, AR events
    D) Bias by PRISM precipitation intensity, non-AR events

One figure is made for each number of events in the prior 7 days
(``events_in_prior_7days`` = 0, 1, 2, ...), plus one figure with all events.
Boxes hide outliers; product/bin combinations with fewer than 3 events are
additionally drawn as individual points annotated with their count. X-axis
tick labels report the number of unique events in each bin. All panels of a
figure share a common y-axis range.

Input
    analysis/outputs/5day_taking_min_and_filling/5_prepared_bias_table.csv
    analysis/outputs/5day_taking_min_and_filling/3.5_events_with_prior_counts.csv
Output
    plots/bias_boxplots/plots/prior_events/9_prism_bias_ar_vs_nonar_prior_events_<N>.png
    plots/bias_boxplots/plots/prior_events/9_prism_bias_ar_vs_nonar_all_events.png

Usage
    python 9.5_prism_bias_ar_vs_nonar_by_prior_events.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "analysis" / "outputs" / "5day_taking_min_and_filling"
BIAS_CSV = DATA_DIR / "5_prepared_bias_table.csv"
PRIOR_EVENTS_CSV = DATA_DIR / "3.5_events_with_prior_counts.csv"
OUT_DIR = BASE_DIR / "plots" / "bias_boxplots" / "plots" / "prior_events"

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

# Font sizes
FONT_SUPTITLE = 22
FONT_SUBTITLE = 14
FONT_TITLE = 14
FONT_LABEL = 16
FONT_TICK = 14
FONT_LEGEND = 14
FONT_LEGEND_TITLE = 16
FONT_ANNOT = 10  # "N=" labels on sparse bins


def plot_prism_bias_by_prior_events():

    # These functions sort the x-axis categories in a custom order for each metric.
    def custom_sort_streamflow(values):
        order = STREAMFLOW_ORDER
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))

    def custom_sort_precipitation(values):
        order = PRECIP_ORDER
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))

    print("Loading prepared bias table...")
    df = pd.read_csv(BIAS_CSV)
    df["date"] = pd.to_datetime(df["date"])

    print("Loading prior events counts...")
    prior_events = pd.read_csv(PRIOR_EVENTS_CSV, parse_dates=["date"])
    prior_events = prior_events[["date", "events_in_prior_7days"]]

    # Merge prior events counts with bias data
    df = df.merge(prior_events, on="date", how="left")

    # Keep only valid AR scale values
    df = df.dropna(subset=["ar_scale"])
    df = df[(df["ar_scale"] >= AR_SCALE_RANGE[0]) & (df["ar_scale"] <= AR_SCALE_RANGE[1])]

    # Melt to long format for plotting
    melted = df.melt(
        id_vars=["date", "ar_scale", "Streamflow Bucket (cms)", "Precipitation Bucket (mm)", "events_in_prior_7days"],
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    handles = [Patch(facecolor=PALETTE[p], edgecolor=PALETTE[p], label=p) for p in PRODUCTS]

    # One figure per number of events in the prior 7 days.
    prior_counts = sorted(period1["events_in_prior_7days"].unique())
    print(f"Creating plots for prior event counts: {prior_counts}")

    for prior_count in prior_counts:
        print(f"\nCreating plot for events_in_prior_7days = {prior_count}")

        data_subset = period1[period1["events_in_prior_7days"] == prior_count]
        ar_events = data_subset[data_subset["ar_scale"] > 0]
        non_ar_events = data_subset[data_subset["ar_scale"] == 0]
        n_ar = ar_events["date"].nunique()
        n_nonar = non_ar_events["date"].nunique()
        n_total = n_ar + n_nonar

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

        plot_panel(axes[0, 0], ar_events, "Streamflow Bucket (cms)", custom_sort_streamflow)
        axes[0, 0].set_title(f"A) Bias by Streamflow Intensity ({START_YEAR} - {END_YEAR})\nAR Events", weight="bold")
        axes[0, 0].set_xlabel("Streamflow Range (cms)", labelpad=8)

        plot_panel(axes[0, 1], non_ar_events, "Streamflow Bucket (cms)", custom_sort_streamflow)
        axes[0, 1].set_title(f"B) Bias by Streamflow Intensity ({START_YEAR} - {END_YEAR})\nNon-AR Events", weight="bold")
        axes[0, 1].set_xlabel("Streamflow Range (cms)", labelpad=8)

        plot_panel(axes[1, 0], ar_events, "Precipitation Bucket (mm)", custom_sort_precipitation)
        axes[1, 0].set_title(f"C) Bias by PRISM Precipitation Intensity ({START_YEAR} - {END_YEAR})\nAR Events", weight="bold")
        axes[1, 0].set_xlabel("PRISM 7-Day Total Precipitation Range (mm)", labelpad=8)

        plot_panel(axes[1, 1], non_ar_events, "Precipitation Bucket (mm)", custom_sort_precipitation)
        axes[1, 1].set_title(f"D) Bias by PRISM Precipitation Intensity ({START_YEAR} - {END_YEAR})\nNon-AR Events", weight="bold")
        axes[1, 1].set_xlabel("PRISM 7-Day Total Precipitation Range (mm)", labelpad=8)

        # One shared legend for the whole figure instead of one per panel.
        fig.legend(handles=handles, title="Product", title_fontsize=FONT_LEGEND_TITLE,
                   loc="upper center", bbox_to_anchor=(0.5, 0.90),
                   ncol=6, frameon=True, edgecolor="#cccccc")
        for ax in axes.flat:
            if ax.get_legend() is not None:  # empty panels have no legend
                ax.get_legend().remove()

        prior_label = f"{prior_count} prior event{'s' if prior_count != 1 else ''}"
        plt.suptitle(f"Cross-Product Precipitation Bias Analysis: {START_YEAR}-{END_YEAR}\n(Events with {prior_label} in 7-day window)\n(Product - PRISM 7-Day Total)",
                     fontsize=FONT_SUPTITLE, weight="bold", y=1.05)
        fig.text(0.5, 0.92, f"AR events: {n_ar} | Non-AR events: {n_nonar} | Total: {n_total} | 7-day window: T-6 through T",
                 ha="center", fontsize=FONT_SUBTITLE, color="dimgray")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Common y-axis range across all panels.
        y_min = min(ax.get_ylim()[0] for ax in axes.flat)
        y_max = max(ax.get_ylim()[1] for ax in axes.flat)
        for ax in axes.flat:
            ax.set_ylim(y_min, y_max)

        out_img = OUT_DIR / f"9_prism_bias_ar_vs_nonar_prior_events_{prior_count}.png"
        plt.savefig(out_img, dpi=300, bbox_inches="tight")
        print(f"  Saved to {out_img}")
        plt.close()

    # One figure with all events (independent of prior event counts).
    print("\nCreating plot for all events combined...")
    ar_events = period1[period1["ar_scale"] > 0]
    non_ar_events = period1[period1["ar_scale"] == 0]
    n_ar = ar_events["date"].nunique()
    n_nonar = non_ar_events["date"].nunique()
    n_total = n_ar + n_nonar

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    plot_panel(axes[0, 0], ar_events, "Streamflow Bucket (cms)", custom_sort_streamflow)
    axes[0, 0].set_title(f"A) Bias by Streamflow Intensity ({START_YEAR} - {END_YEAR})\nAR Events", weight="bold")
    axes[0, 0].set_xlabel("Streamflow Range (cms)", labelpad=8)

    plot_panel(axes[0, 1], non_ar_events, "Streamflow Bucket (cms)", custom_sort_streamflow)
    axes[0, 1].set_title(f"B) Bias by Streamflow Intensity ({START_YEAR} - {END_YEAR})\nNon-AR Events", weight="bold")
    axes[0, 1].set_xlabel("Streamflow Range (cms)", labelpad=8)

    plot_panel(axes[1, 0], ar_events, "Precipitation Bucket (mm)", custom_sort_precipitation)
    axes[1, 0].set_title(f"C) Bias by PRISM Precipitation Intensity ({START_YEAR} - {END_YEAR})\nAR Events", weight="bold")
    axes[1, 0].set_xlabel("PRISM 7-Day Total Precipitation Range (mm)", labelpad=8)

    plot_panel(axes[1, 1], non_ar_events, "Precipitation Bucket (mm)", custom_sort_precipitation)
    axes[1, 1].set_title(f"D) Bias by PRISM Precipitation Intensity ({START_YEAR} - {END_YEAR})\nNon-AR Events", weight="bold")
    axes[1, 1].set_xlabel("PRISM 7-Day Total Precipitation Range (mm)", labelpad=8)

    # One shared legend for the whole figure instead of one per panel.
    fig.legend(handles=handles, title="Product", title_fontsize=FONT_LEGEND_TITLE,
               loc="upper center", bbox_to_anchor=(0.5, 0.9),
               ncol=6, frameon=True, edgecolor="#cccccc")
    for ax in axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.suptitle(f"Cross-Product Precipitation Bias Analysis: {START_YEAR}-{END_YEAR}\n(All Events - Independent of Prior Events)\n(Product - PRISM 7-Day Total)",
                 fontsize=FONT_SUPTITLE, weight="bold", y=1.05)
    fig.text(0.5, 0.92, f"AR events: {n_ar} | Non-AR events: {n_nonar} | Total: {n_total}",
             ha="center", fontsize=FONT_SUBTITLE, color="dimgray")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Common y-axis range across all panels.
    y_min = min(ax.get_ylim()[0] for ax in axes.flat)
    y_max = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_ylim(y_min, y_max)

    out_img = OUT_DIR / "9_prism_bias_ar_vs_nonar_all_events.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"  Saved to {out_img}")
    plt.close()

    print("\nDone! All plots created.")


if __name__ == "__main__":
    plot_prism_bias_by_prior_events()
