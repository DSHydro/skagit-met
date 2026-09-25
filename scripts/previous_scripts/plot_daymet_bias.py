"""Generate the Daymet-based appendix-style bias figure.

This script mirrors the PRISM-based plot but uses Daymet as the baseline.
It is useful for appendix-style comparisons.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

BASE_DIR = "/home/nksp2/skagit/skagit_2/skagit-met"
INPUT_CSV = os.path.join(BASE_DIR, "reproducible_skagit_analysis", "data", "prepared_bias_table.csv")
OUT_DIR = os.path.join(BASE_DIR, "reproducible_skagit_analysis", "data")
OUT_IMG = os.path.join(OUT_DIR, "grouped_bias_daymet_baseline.png")


def plot_daymet_bias():
    print("Loading prepared bias table...")
    df = pd.read_csv(INPUT_CSV)
    df["date"] = pd.to_datetime(df["date"])

    df = df.dropna(subset=["ar_scale"])
    df = df[(df["ar_scale"] >= 0) & (df["ar_scale"] <= 5)]

    products = ["PRISM", "PNNL", "CONUS404", "UCLA", "GridMET", "HRRR"]
    df["PRISM"] = df["prism_3d_tot"] - df["daymet_3d_tot"]
    df["PNNL"] = df["pnnl_3d_tot"] - df["daymet_3d_tot"]
    df["CONUS404"] = df["conus_3d_tot"] - df["daymet_3d_tot"]
    df["UCLA"] = df["ucla_3d_tot"] - df["daymet_3d_tot"]
    df["GridMET"] = df["gridmet_3d_tot"] - df["daymet_3d_tot"]
    df["HRRR"] = df["hrrr_3d_tot"] - df["daymet_3d_tot"]

    melted = df.melt(
        id_vars=["date", "ar_scale", "Streamflow Bucket (cfs)", "Precipitation Bucket (mm)"],
        value_vars=products,
        var_name="Product",
        value_name="Bias",
    )
    melted = melted.dropna(subset=["Bias"])

    palette = {
        "PRISM": "#1b9e77",
        "PNNL": "#d95f02",
        "CONUS404": "#e7298a",
        "UCLA": "#66a61e",
        "GridMET": "#e6ab02",
        "HRRR": "#a6761d",
    }

    period1 = melted[(melted["date"].dt.year >= 1981) & (melted["date"].dt.year <= 2020)]
    period1 = period1[period1["Product"] != "HRRR"]
    period2 = melted[melted["date"].dt.year >= 2014]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Arial", "Liberation Sans"],
    })
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(24, 20), facecolor="white")

    for ax, data, title in [
        (axes[0, 0], period1, "Bias by AR Scale (1981 - 2020)"),
        (axes[0, 1], period2, "Bias by AR Scale (2014 - Present)"),
        (axes[1, 0], period1, "Bias by Streamflow Intensity (1981 - 2020)"),
        (axes[1, 1], period2, "Bias by Streamflow Intensity (2014 - Present)"),
        (axes[2, 0], period1, "Bias by Daymet Precipitation Intensity (1981 - 2020)"),
        (axes[2, 1], period2, "Bias by Daymet Precipitation Intensity (2014 - Present)"),
    ]:
        sns.boxplot(data=data, x="ar_scale" if "Bias by AR Scale" in title else "Streamflow Bucket (cfs)" if "Streamflow" in title else "Precipitation Bucket (mm)", y="Bias", hue="Product", hue_order=products, palette=palette, linewidth=1.2, fliersize=3, ax=ax, showfliers=False)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_title(title, fontsize=16, weight="bold")
        ax.set_ylabel("Bias (mm)", fontsize=14)

    handles = [Patch(facecolor=palette[p], edgecolor=palette[p], alpha=1.0, label=p) for p in products]
    axes[0, 0].legend(handles=handles, title="Product", title_fontsize=13, fontsize=12,
                      loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, edgecolor="#cccccc")
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.suptitle("Cross-Product Precipitation Bias Analysis: 1981-2020 vs 2014-Present\n(Product - Daymet 3-Day Total)", fontsize=22, weight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUT_IMG}")


if __name__ == "__main__":
    plot_daymet_bias()
