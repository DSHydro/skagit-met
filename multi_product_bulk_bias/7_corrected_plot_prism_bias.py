"""Generate the PRISM-based manuscript-style bias figure.

This script uses the prepared bias table and creates the multi-panel figure
for the main manuscript plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/multi_product_bulk_bias"
INPUT_CSV = os.path.join(BASE_DIR, "outputs", "5_prepared_bias_table.csv")
HERNAN_DIR = "/data0/hernanqd/plots_code"
OUT_DIR = os.path.join(BASE_DIR, "plots")
OUT_IMG = os.path.join(OUT_DIR, "grouped_bias_3_metrics_2_periods.png")


def plot_prism_bias():

    # This three functions are used to sort the x-axis categories in a custom order for each metric.
    def custom_sort_ar_scale(values):
        return sorted(values, key=lambda x: float(x))
    
    def custom_sort_streamflow(values):
        # Old order in cfs: ["< 10k", "10k - 20k", "20k - 40k", "> 40k"]
        # Old order in cfs: ["< 20k", "20k - 40k", "40k - 60k", "> 60k"]
        order = ["< 600", "600 - 1200", "1200 - 1700", "> 1700"]
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))
    
    def custom_sort_precipitation(values):
        order = ["< 20mm", "20 - 50mm", "50 - 100mm", "> 100mm"]
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))

    print("Loading prepared bias table...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])

    # Keep only valid AR scale values.
    df = df.dropna(subset=["ar_scale"])
    df = df[(df["ar_scale"] >= 0) & (df["ar_scale"] <= 5)]

    # Melt to long format for plotting.
    products = ["PNNL", "Daymet","CONUS404", "UCLA", "GridMET"] #, "HRRR"] #, "ORNL_mean", "ORNL_median"]
    melted = df.melt(
        id_vars=["date", "ar_scale", "Streamflow Bucket (cms)", "Precipitation Bucket (mm)", "prism_3d_tot"],
        value_vars=products,
        var_name="Product",
        value_name="Bias",
    )
    melted = melted.dropna(subset=["Bias"])

    palette = {
        "PNNL": "#d95f02",
        "Daymet": "#7570b3",
        "CONUS404": "#e7298a",
        "UCLA": "#66a61e",
        "GridMET": "#e6ab02",
        # "HRRR": "#a6761d",
    }

    # Split historical and modern periods.
    period1 = melted[(melted["date"].dt.year >= 1981) & (melted["date"].dt.year <= 2020)]
    period1 = period1[period1["Product"] != "HRRR"]
    period2 = melted[melted["date"].dt.year >= 2014]

    def plot_panel(ax, data, x_col, products_in, min_n=3, sort_func=None, show_bucket_size=False, show_mean_prism=False, prism_ylim=None):
        # x_levels = list(dict.fromkeys(data[x_col].astype(str).tolist()))

        x_levels = list(dict.fromkeys(data[x_col].astype(str).tolist()))
        if sort_func is not None:
            x_levels = sort_func(x_levels)

        sparse_bins = []
        for product in products_in:
            for x_level in x_levels:
                subset = data[(data[x_col].astype(str) == x_level) & (data["Product"] == product)]
                values = subset["Bias"].dropna().to_numpy()
                if len(values) > 0 and len(values) < min_n:
                    sparse_bins.append((product, x_level, values))

        sns.boxplot(
            data=data,
            x=x_col,
            y="Bias",
            hue="Product",
            hue_order=products_in,
            palette=palette,
            linewidth=1.2,
            fliersize=3,
            ax=ax,
            showfliers=False,
            order=x_levels,
        )

        for product, x_level, values in sparse_bins:
            x_idx = x_levels.index(x_level)
            x_pos = x_idx + 0.18 * (products_in.index(product) - (len(products_in) - 1) / 2)
            ax.scatter([x_pos] * len(values), values, color=palette[product], s=24, alpha=1.0, zorder=3)
            ax.text(x_pos, np.max(values) + 0.5, f"N={len(values)}", ha="center", va="bottom", fontsize=10, color="dimgray")

        # Add bucket sizes to x-axis labels if requested
        if show_bucket_size:
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
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white") #, sharey='row')

    plot_panel(axes[0, 0], period1[period1["ar_scale"] > 0], "ar_scale", [p for p in products], sort_func=custom_sort_ar_scale, show_bucket_size=True)
    axes[0, 0].set_title("A) Bias by AR Scale (1981 - 2020)", weight="bold")
    axes[0, 0].set_xlabel("Atmospheric River Scale (1-5)", labelpad=8)


    plot_panel(axes[1, 0], period1, "Streamflow Bucket (cms)", [p for p in products], sort_func=custom_sort_streamflow, show_bucket_size=True)
    axes[1, 0].set_title("C) Bias by Streamflow Intensity (1981 - 2020)\nAR and non-AR events", weight="bold", fontsize=14)
    axes[1, 0].set_xlabel("Streamflow Range (cms)", labelpad=8)


    plot_panel(axes[0, 1], period1[period1["ar_scale"] == 0], "Precipitation Bucket (mm)", [p for p in products], sort_func=custom_sort_precipitation, show_bucket_size=True)
    axes[0, 1].set_title("B) Bias by PRISM Precipitation Intensity (1981 - 2020)\nNon-AR Events", weight="bold", fontsize=14)
    axes[0, 1].set_xlabel("PRISM 3-Day Total Precipitation Range (mm)", labelpad=8)

    plot_panel(axes[1, 1], period1, "Precipitation Bucket (mm)", [p for p in products], sort_func=custom_sort_precipitation, show_bucket_size=True)
    axes[1, 1].set_title("D) Bias by PRISM Precipitation Intensity (1981 - 2020)\nAR and Non-AR Events", weight="bold", fontsize=14)
    axes[1, 1].set_xlabel("PRISM 3-Day Total Precipitation Range (mm)", labelpad=8)

    handles = [Patch(facecolor=palette[p], edgecolor=palette[p], alpha=1.0, label=p) for p in products]

    # axes[0, 0].legend(handles=handles, title="Product", title_fontsize=13,
    #                   loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, edgecolor="#cccccc")

    # Remove the subplot legend line and replace with:
    fig.legend(handles=handles, title="Product", title_fontsize=16,
                loc="upper center", bbox_to_anchor=(0.5, 0.95),
                ncol=6, frameon=True, edgecolor="#cccccc")

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.suptitle("Cross-Product Precipitation Bias Analysis: 1981-2020\n(Product - PRISM 3-Day Total)", fontsize=22, weight="bold", y=1.05)
    fig.text(0.5, 0.95, "3-day window computed as: the day of the event and the prior 2 days", ha="center", fontsize=16, color="dimgray")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    y_min = min(axes[i, j].get_ylim()[0] for i in range(2) for j in range(2))
    y_max = max(axes[i, j].get_ylim()[1] for i in range(2) for j in range(2))
    for i in range(2):
      for j in range(2):
        axes[i, j].set_ylim(y_min, y_max)

    
    plt.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUT_IMG}")


if __name__ == "__main__":
    plot_prism_bias()
