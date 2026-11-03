"""Generate PRISM bias (AR vs non-AR) figures split by prior event counts.

Creates 4 separate plots, one for each value of events_in_prior_7days (0, 1, 2, 3).
Each plot compares AR and non-AR events for streamflow and precipitation bias.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

BASE_DIR = "/data0/hernanqd/plots_code/skagit-met/updating_multi_product_bulk_bias"
BIAS_CSV = os.path.join(BASE_DIR, "outputs", "5_prepared_bias_table.csv")
PRIOR_EVENTS_CSV = os.path.join(BASE_DIR, "outputs", "3.5_events_with_prior_counts.csv")
OUT_DIR = os.path.join(BASE_DIR, "plots")


def plot_prism_bias_by_prior_events():

    def custom_sort_ar_scale(values):
        return sorted(values, key=lambda x: float(x))

    def custom_sort_streamflow(values):
        order = ["< 600", "600 - 1200", "1200 - 1700", "> 1700"]
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))

    def custom_sort_precipitation(values):
        order = ["< 20mm", "20 - 50mm", "50 - 100mm", "> 100mm"]
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))

    print("Loading prepared bias table...")
    df = pd.read_csv(BIAS_CSV)
    df["date"] = pd.to_datetime(df["date"])

    print("Loading prior events counts...")
    prior_events = pd.read_csv(PRIOR_EVENTS_CSV, parse_dates=['date'])
    prior_events = prior_events[['date', 'events_in_prior_7days']]

    # Merge prior events counts with bias data
    df = df.merge(prior_events, on='date', how='left')

    # Keep only valid AR scale values
    df = df.dropna(subset=["ar_scale"])
    df = df[(df["ar_scale"] >= 0) & (df["ar_scale"] <= 5)]

    # Melt to long format for plotting
    products = ["PNNL", "Daymet", "CONUS404", "UCLA", "GridMET"]
    melted = df.melt(
        id_vars=["date", "ar_scale", "Streamflow Bucket (cms)", "Precipitation Bucket (mm)", "prism_7d_tot", "events_in_prior_7days"],
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
    }

    # Filter to historical period (1981-2020)
    period1 = melted[(melted["date"].dt.year >= 1981) & (melted["date"].dt.year <= 2020)]

    def plot_panel(ax, data, x_col, products_in, min_n=3, sort_func=None, show_bucket_size=False):
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

    # Get unique prior event counts and sort them
    prior_counts = sorted(period1['events_in_prior_7days'].unique())
    print(f"Creating plots for prior event counts: {prior_counts}")

    for prior_count in prior_counts:
        print(f"\nCreating plot for events_in_prior_7days = {prior_count}")

        # Filter data for this prior event count
        data_subset = period1[period1['events_in_prior_7days'] == prior_count]
        n_ar = (data_subset[data_subset["ar_scale"] > 0]["date"]).nunique()
        n_nonar = (data_subset[data_subset["ar_scale"] == 0]["date"]).nunique()
        n_total = n_ar + n_nonar

        if len(data_subset) == 0:
            print(f"  No data for prior_count={prior_count}, skipping...")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

        # Top-left: AR streamflow
        plot_panel(axes[0, 0], data_subset[data_subset["ar_scale"] > 0], "Streamflow Bucket (cms)",
                   [p for p in products], sort_func=custom_sort_streamflow, show_bucket_size=True)
        axes[0, 0].set_title("A) Bias by Streamflow Intensity (1981 - 2020)\nAR Events", weight="bold", fontsize=14)
        axes[0, 0].set_xlabel("Streamflow Range (cms)", labelpad=8)

        # Top-right: Non-AR streamflow
        plot_panel(axes[0, 1], data_subset[data_subset["ar_scale"] == 0], "Streamflow Bucket (cms)",
                   [p for p in products], sort_func=custom_sort_streamflow, show_bucket_size=True)
        axes[0, 1].set_title("B) Bias by Streamflow Intensity (1981 - 2020)\nNon-AR Events", weight="bold", fontsize=14)
        axes[0, 1].set_xlabel("Streamflow Range (cms)", labelpad=8)

        # Bottom-left: AR precipitation
        plot_panel(axes[1, 0], data_subset[data_subset["ar_scale"] > 0], "Precipitation Bucket (mm)",
                   [p for p in products], sort_func=custom_sort_precipitation, show_bucket_size=True)
        axes[1, 0].set_title("C) Bias by PRISM Precipitation Intensity (1981 - 2020)\nAR Events", weight="bold", fontsize=14)
        axes[1, 0].set_xlabel("PRISM 7-Day Total Precipitation Range (mm)", labelpad=8)

        # Bottom-right: Non-AR precipitation
        plot_panel(axes[1, 1], data_subset[data_subset["ar_scale"] == 0], "Precipitation Bucket (mm)",
                   [p for p in products], sort_func=custom_sort_precipitation, show_bucket_size=True)
        axes[1, 1].set_title("D) Bias by PRISM Precipitation Intensity (1981 - 2020)\nNon-AR Events", weight="bold", fontsize=14)
        axes[1, 1].set_xlabel("PRISM 7-Day Total Precipitation Range (mm)", labelpad=8)

        handles = [Patch(facecolor=palette[p], edgecolor=palette[p], alpha=1.0, label=p) for p in products]

        fig.legend(handles=handles, title="Product", title_fontsize=16,
                   loc="upper center", bbox_to_anchor=(0.5, 0.90),
                   ncol=6, frameon=True, edgecolor="#cccccc")

        for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        prior_label = f"{prior_count} prior event{'s' if prior_count != 1 else ''}"
        plt.suptitle(f"Cross-Product Precipitation Bias Analysis: 1981-2020\n(Events with {prior_label} in 7-day window)\n(Product - PRISM 7-Day Total)",
                     fontsize=22, weight="bold", y=1.05)
        fig.text(0.5, 0.92, f"AR events: {n_ar} | Non-AR events: {n_nonar} | Total: {n_total} | 7-day window: T-6 through T",
                 ha="center", fontsize=14, color="dimgray")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Set consistent y-axis limits across all subplots
        y_min = min(axes[i, j].get_ylim()[0] for i in range(2) for j in range(2))
        y_max = max(axes[i, j].get_ylim()[1] for i in range(2) for j in range(2))
        for i in range(2):
            for j in range(2):
                axes[i, j].set_ylim(y_min, y_max)

        # Save figure
        out_img = os.path.join(OUT_DIR, f"9_prism_bias_ar_vs_nonar_prior_events_{prior_count}.png")
        plt.savefig(out_img, dpi=300, bbox_inches="tight")
        print(f"  Saved to {out_img}")
        plt.close()

    # Create plot with ALL events (independent of prior event counts)
    print(f"\nCreating plot for all events combined...")
    data_all = period1
    n_ar = (data_all[data_all["ar_scale"] > 0]["date"]).nunique()
    n_nonar = (data_all[data_all["ar_scale"] == 0]["date"]).nunique()
    n_total = n_ar + n_nonar

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")

    # Top-left: AR streamflow
    plot_panel(axes[0, 0], data_all[data_all["ar_scale"] > 0], "Streamflow Bucket (cms)",
               [p for p in products], sort_func=custom_sort_streamflow, show_bucket_size=True)
    axes[0, 0].set_title("A) Bias by Streamflow Intensity (1981 - 2020)\nAR Events", weight="bold", fontsize=14)
    axes[0, 0].set_xlabel("Streamflow Range (cms)", labelpad=8)

    # Top-right: Non-AR streamflow
    plot_panel(axes[0, 1], data_all[data_all["ar_scale"] == 0], "Streamflow Bucket (cms)",
               [p for p in products], sort_func=custom_sort_streamflow, show_bucket_size=True)
    axes[0, 1].set_title("B) Bias by Streamflow Intensity (1981 - 2020)\nNon-AR Events", weight="bold", fontsize=14)
    axes[0, 1].set_xlabel("Streamflow Range (cms)", labelpad=8)

    # Bottom-left: AR precipitation
    plot_panel(axes[1, 0], data_all[data_all["ar_scale"] > 0], "Precipitation Bucket (mm)",
               [p for p in products], sort_func=custom_sort_precipitation, show_bucket_size=True)
    axes[1, 0].set_title("C) Bias by PRISM Precipitation Intensity (1981 - 2020)\nAR Events", weight="bold", fontsize=14)
    axes[1, 0].set_xlabel("PRISM 7-Day Total Precipitation Range (mm)", labelpad=8)

    # Bottom-right: Non-AR precipitation
    plot_panel(axes[1, 1], data_all[data_all["ar_scale"] == 0], "Precipitation Bucket (mm)",
               [p for p in products], sort_func=custom_sort_precipitation, show_bucket_size=True)
    axes[1, 1].set_title("D) Bias by PRISM Precipitation Intensity (1981 - 2020)\nNon-AR Events", weight="bold", fontsize=14)
    axes[1, 1].set_xlabel("PRISM 7-Day Total Precipitation Range (mm)", labelpad=8)

    handles = [Patch(facecolor=palette[p], edgecolor=palette[p], alpha=1.0, label=p) for p in products]

    fig.legend(handles=handles, title="Product", title_fontsize=16,
               loc="upper center", bbox_to_anchor=(0.5, 0.9),
               ncol=6, frameon=True, edgecolor="#cccccc")

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.suptitle(f"Cross-Product Precipitation Bias Analysis: 1981-2020\n(All Events - Independent of Prior Events)\n(Product - PRISM 7-Day Total)",
                 fontsize=22, weight="bold", y=1.05)
    fig.text(0.5, 0.92, f"AR events: {n_ar} | Non-AR events: {n_nonar} | Total: {n_total}",
             ha="center", fontsize=14, color="dimgray")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Set consistent y-axis limits across all subplots
    y_min = min(axes[i, j].get_ylim()[0] for i in range(2) for j in range(2))
    y_max = max(axes[i, j].get_ylim()[1] for i in range(2) for j in range(2))
    for i in range(2):
        for j in range(2):
            axes[i, j].set_ylim(y_min, y_max)

    # Save figure
    out_img = os.path.join(OUT_DIR, "9_prism_bias_ar_vs_nonar_all_events.png")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"  Saved to {out_img}")
    plt.close()

    print("\nDone! All plots created.")


if __name__ == "__main__":
    plot_prism_bias_by_prior_events()
