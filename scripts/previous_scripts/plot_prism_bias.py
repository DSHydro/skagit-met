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

BASE_DIR = "/home/nksp2/skagit/skagit_2/skagit-met"
INPUT_CSV = os.path.join(BASE_DIR, "reproducible_skagit_analysis", "data", "prepared_bias_table.csv")
OUT_DIR = os.path.join(BASE_DIR, "reproducible_skagit_analysis", "data")
OUT_IMG = os.path.join(OUT_DIR, "grouped_bias_3_metrics_2_periods.png")


def plot_prism_bias():
    print("Loading prepared bias table...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])

    # Keep only valid AR scale values.
    df = df.dropna(subset=["ar_scale"])
    df = df[(df["ar_scale"] >= 0) & (df["ar_scale"] <= 5)]

    # Melt to long format for plotting.
    products = ["PNNL", "CONUS404", "UCLA", "GridMET", "HRRR"]
    melted = df.melt(
        id_vars=["date", "ar_scale", "Streamflow Bucket (cfs)", "Precipitation Bucket (mm)"],
        value_vars=products,
        var_name="Product",
        value_name="Bias",
    )
    melted = melted.dropna(subset=["Bias"])

    palette = {
        "PNNL": "#d95f02",
        "CONUS404": "#e7298a",
        "UCLA": "#66a61e",
        "GridMET": "#e6ab02",
        "HRRR": "#a6761d",
    }

    # Split historical and modern periods.
    period1 = melted[(melted["date"].dt.year >= 1981) & (melted["date"].dt.year <= 2020)]
    period1 = period1[period1["Product"] != "HRRR"]
    period2 = melted[melted["date"].dt.year >= 2014]

    def plot_panel(ax, data, x_col, products_in, min_n=3):
        x_levels = list(dict.fromkeys(data[x_col].astype(str).tolist()))
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
        )

        for product, x_level, values in sparse_bins:
            x_idx = x_levels.index(x_level)
            x_pos = x_idx + 0.18 * (products_in.index(product) - (len(products_in) - 1) / 2)
            ax.scatter([x_pos] * len(values), values, color=palette[product], s=24, alpha=1.0, zorder=3)
            ax.text(x_pos, np.max(values) + 0.5, f"N={len(values)}", ha="center", va="bottom", fontsize=8, color="dimgray")

        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_ylabel("Bias (mm)", fontsize=14)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Arial", "Liberation Sans"],
    })
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 2, figsize=(24, 20), facecolor="white")

    plot_panel(axes[0, 0], period1, "ar_scale", [p for p in products if p != "HRRR"])
    axes[0, 0].set_title("Bias by AR Scale (1981 - 2020)", fontsize=16, weight="bold")
    axes[0, 0].set_xlabel("Atmospheric River Scale (0-5)", fontsize=14)

    plot_panel(axes[0, 1], period2, "ar_scale", products)
    axes[0, 1].set_title("Bias by AR Scale (2014 - Present)", fontsize=16, weight="bold")
    axes[0, 1].set_xlabel("Atmospheric River Scale (0-5)", fontsize=14)

    plot_panel(axes[1, 0], period1, "Streamflow Bucket (cfs)", [p for p in products if p != "HRRR"])
    axes[1, 0].set_title("Bias by Streamflow Intensity (1981 - 2020)", fontsize=16, weight="bold")
    axes[1, 0].set_xlabel("Streamflow Range (cfs)", fontsize=14)

    plot_panel(axes[1, 1], period2, "Streamflow Bucket (cfs)", products)
    axes[1, 1].set_title("Bias by Streamflow Intensity (2014 - Present)", fontsize=16, weight="bold")
    axes[1, 1].set_xlabel("Streamflow Range (cfs)", fontsize=14)

    plot_panel(axes[2, 0], period1, "Precipitation Bucket (mm)", [p for p in products if p != "HRRR"])
    axes[2, 0].set_title("Bias by PRISM Precipitation Intensity (1981 - 2020)", fontsize=16, weight="bold")
    axes[2, 0].set_xlabel("PRISM 3-Day Total Precipitation Range (mm)", fontsize=14)

    plot_panel(axes[2, 1], period2, "Precipitation Bucket (mm)", products)
    axes[2, 1].set_title("Bias by PRISM Precipitation Intensity (2014 - Present)", fontsize=16, weight="bold")
    axes[2, 1].set_xlabel("PRISM 3-Day Total Precipitation Range (mm)", fontsize=14)

    handles = [Patch(facecolor=palette[p], edgecolor=palette[p], alpha=1.0, label=p) for p in products]
    axes[0, 0].legend(handles=handles, title="Product", title_fontsize=13, fontsize=12,
                      loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, edgecolor="#cccccc")
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.suptitle("Cross-Product Precipitation Bias Analysis: 1981-2020 vs 2014-Present\n(Product - PRISM 3-Day Total)", fontsize=22, weight="bold", y=1.02)
    fig.text(0.5, 0.965, "HRRR: basin integration is still under review; bins with N<3 are shown as points.", ha="center", fontsize=10, color="dimgray")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUT_IMG}")


if __name__ == "__main__":
    plot_prism_bias()
