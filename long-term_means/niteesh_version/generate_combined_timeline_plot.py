#Created by Niteesh

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# CANONICAL SETTINGS (HRRR EXCLUDED)
# ============================================================
DATASET_ORDER = [
    "PRISM 4km",
    "Daymet-PC",
    "gridMET",
    "UCLA ERA5 d02",
    "PNNL hist",
    "CONUS404",
]

COLORS = {
    "PRISM 4km":     "#1B9E77",
    "Daymet-PC":     "#7570B3",
    "gridMET":       "#E7298A",
    "UCLA ERA5 d02": "#E6AB02",
    "PNNL hist":     "#D95F02",
    "CONUS404":      "#666666",
}

# Mapping for CSV dataset names
PPT_NAME_MAP = {
    "PRISM": "PRISM 4km",
    "Daymet v4 (Planetary Computer)": "Daymet-PC",
    "Daymet v4": "Daymet-PC",
    "Daymet": "Daymet-PC",
    "HRRR (f06)": "HRRR",
    "HRRR (f01)": "HRRR",
    "HRRR": "HRRR",
    "UCLA ERA5 WRF d02 (daily NetCDF)": "UCLA ERA5 d02",
    "PNNL (historical)": "PNNL hist",
    "CONUS404 (daily-osn)": "CONUS404",
}

REGIONS = ["Full Skagit", "Upper Skagit", "Sauk", "Lower Skagit"]

# Use extended CONUS404 data from workspace (2003-2020 forward-filled)
# Fallback to /data0 if not available
# TEMP_CSV_WORKSPACE = "/home/nksp2/skagit/skagit_2/skagit-met/wy_mean_temp_skagit_regions_1980_2024_ALL.csv"
# TEMP_CSV_ORIGINAL = "/data0/skagit_met/data_transfer/data/derived/wy_mean_temp_skagit_regions_1980_2024_ALL.csv"
# try:
#   TEMP_CSV = TEMP_CSV_WORKSPACE if Path(TEMP_CSV_WORKSPACE).exists() else TEMP_CSV_ORIGINAL
# except PermissionError:
#   TEMP_CSV = TEMP_CSV_ORIGINAL
PPT_CSV = "/data0/skagit_met/data_transfer/data/derived/wy_total_precip_skagit_regions_1983_2024_ALL.csv"

# OUT_DIR = Path("/home/nksp2/skagit/skagit_2/skagit-met/analysis/event_analysis/atmospheric_rivers/graphs_2")
OUT_DIR = Path("/data0/hernanqd/plots_code/skagit-met/long-term_means/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum year to include in plots (avoid incomplete WYs like 2021+)
MAX_YEAR = 2020
START_YEAR = 1983

def load_and_clean_data(region="Full Skagit"):
    # # 1. Load Temperature
    # df_temp = pd.read_csv(TEMP_CSV)
    # df_temp["dataset"] = df_temp["dataset"].map(lambda x: PPT_NAME_MAP.get(x, x))
    # df_temp = df_temp[df_temp["region"] == region].copy()

    # 2. Load Precipitation
    df_ppt = pd.read_csv(PPT_CSV)
    df_ppt["dataset"] = df_ppt["dataset"].map(lambda x: PPT_NAME_MAP.get(x, x))
    df_ppt = df_ppt[df_ppt["region"] == region].copy()

    # ----------------------------
    # EXCLUDE HRRR
    # ----------------------------
    # df_temp = df_temp[~df_temp["dataset"].str.contains("HRRR", case=False, na=False)].copy()
    df_ppt = df_ppt[~df_ppt["dataset"].str.contains("HRRR", case=False, na=False)].copy()

    # ----------------------------
    # CLEANING
    # ----------------------------
    # A) Limit to common period of record: START_YEAR to MAX_YEAR
    # df_temp = df_temp[(df_temp["year"] >= START_YEAR) & (df_temp["year"] <= MAX_YEAR)]
    df_ppt = df_ppt[(df_ppt["year"] >= START_YEAR) & (df_ppt["year"] <= MAX_YEAR)]

    # B) Remove zeros after MAX_YEAR if any slipped through (defensive)
    # df_temp = df_temp[~((df_temp["year"] > MAX_YEAR) & (df_temp["temp_c"] == 0))]
    df_ppt = df_ppt[~((df_ppt["year"] > MAX_YEAR) & (df_ppt["mm"] == 0))]

    # C) Remove specific outliers
    # df_temp = df_temp[~((df_temp["dataset"] == "Daymet-PC") & (df_temp["year"] == 2021))]
    df_ppt = df_ppt[~((df_ppt["dataset"] == "CONUS404") & (df_ppt["year"] == 2022))]
    df_ppt = df_ppt[~((df_ppt["dataset"] == "PRISM 4km") & (df_ppt["year"] == 1982))]

    # D) Remove invalid thresholds
    # df_temp = df_temp[df_temp["temp_c"] >= -5]
    df_ppt = df_ppt[df_ppt["mm"] >= 500]

    return df_ppt

# def plot_combined_panel(df_temp, df_ppt, region, out_filename, start_year=None):
#     if start_year:
#         df_temp = df_temp[df_temp["year"] >= start_year].copy()
#         df_ppt = df_ppt[df_ppt["year"] >= start_year].copy()
#
#     all_years = sorted(list(set(df_temp["year"]) | set(df_ppt["year"])))
#     if not all_years: return
#
#     xticks = np.arange(min(all_years), max(all_years) + 1, 2)
#     xlims = (min(all_years)-1, max(all_years)+1)
#
#     t_min, t_max = df_temp["temp_c"].min(), df_temp["temp_c"].max()
#     p_min, p_max = df_ppt["mm"].min(), df_ppt["mm"].max()
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
#
#     for ds in DATASET_ORDER:
#         # Plot all datasets that have any data for the region, including
#         # incomplete ones like CONUS404. Missing years will show as gaps.
#         d = df_temp[df_temp["dataset"] == ds].sort_values("year")
#         if not d.empty:
#             ax1.plot(d["year"], d["temp_c"], color=COLORS[ds], linewidth=2.5, label=ds)
#
#         d2 = df_ppt[df_ppt["dataset"] == ds].sort_values("year")
#         if not d2.empty:
#             ax2.plot(d2["year"], d2["mm"], color=COLORS[ds], linewidth=2.5, label=ds)
#
#     ax1.set_title(f"{region} | Water-Year Mean Temperature", fontsize=16, fontweight='bold', pad=20)
#     ax1.set_ylabel("Temperature (°C)", fontsize=13)
#     ax1.grid(True, which='both', color='grey', linestyle='-', alpha=0.4, linewidth=1.5)
#     ax1.set_ylim(max(0, np.floor(t_min - 0.5)), np.ceil(t_max + 0.5))
#
#     ax2.set_title(f"{region} | Water-Year Total Precipitation", fontsize=16, fontweight='bold', pad=20)
#     ax2.set_ylabel("Precipitation (mm)", fontsize=13)
#     ax2.set_xlabel("Water Year", fontsize=13)
#     ax2.grid(True, which='both', color='grey', linestyle='-', alpha=0.4, linewidth=1.5)
#     ax2.set_xticks(xticks)
#     ax2.set_xticklabels(xticks, rotation=45)
#     ax2.set_xlim(xlims)
#     ax2.set_ylim(int(np.floor((p_min - 100) / 250.0)) * 250, int(np.ceil((p_max + 100) / 250.0)) * 250)
#
#     h1, l1 = ax1.get_legend_handles_labels()
#     h2, l2 = ax2.get_legend_handles_labels()
#     all_h = []
#     all_l = []
#     for h, l in zip(h1 + h2, l1 + l2):
#         if l not in all_l:
#             all_h.append(h)
#             all_l.append(l)
#
#     # Sort legend by DATASET_ORDER
#     sorted_pairs = []
#     for ds in DATASET_ORDER:
#         if ds in all_l:
#             idx = all_l.index(ds)
#             sorted_pairs.append((all_h[idx], all_l[idx]))
#
#     final_h = [p[0] for p in sorted_pairs]
#     final_l = [p[1] for p in sorted_pairs]
#
#     fig.legend(final_h, final_l, loc='lower center', ncol=len(final_l), bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)
#     plt.tight_layout(rect=[0, 0.08, 1, 1])
#     plt.savefig(out_filename, dpi=300, bbox_inches='tight')
#     plt.close()

def plot_individual(df_ppt, region):
    all_years = sorted(list(df_ppt["year"]))
    if not all_years: return

    xticks = np.arange(min(all_years), max(all_years) + 1, 2)
    xlims = (min(all_years)-1, max(all_years)+1)

    reg_suffix = region.replace(" ", "_")

    p_min, p_max = df_ppt["mm"].min(), df_ppt["mm"].max()

    # # ----------------------------
    # # INDIVIDUAL TEMP PLOT
    # # ----------------------------
    # fig, ax = plt.subplots(figsize=(18, 8))
    # for ds in DATASET_ORDER:
    #     d = df_temp[df_temp["dataset"] == ds].sort_values("year")
    #     if not d.empty:
    #         ax.plot(d["year"], d["temp_c"], color=COLORS[ds], linewidth=2.5, label=ds)
    #
    # ax.set_title(f"{region} | Water-Year Mean Temperature", fontsize=18, fontweight='bold', pad=20)
    # ax.set_ylabel("Temperature (°C)", fontsize=14)
    # ax.set_xlabel("Water Year", fontsize=14)
    # ax.grid(True, which='both', color='grey', linestyle='-', alpha=0.4, linewidth=1.5)
    # ax.set_ylim(max(0, np.floor(t_min - 0.5)), np.ceil(t_max + 0.5))
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(xticks, rotation=45)
    # ax.set_xlim(xlims)
    # handles, labels = ax.get_legend_handles_labels()
    # sorted_pairs = [(h, l) for ds in DATASET_ORDER for h, l in zip(handles, labels) if l == ds]
    # if sorted_pairs:
    #     sh, sl = zip(*sorted_pairs)
    #     ax.legend(sh, sl, loc='upper center', ncol=len(sl), bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)
    #
    # plt.tight_layout()
    # plt.savefig(OUT_DIR / f"wy_mean_temp_{reg_suffix}_standardized.png", dpi=300, bbox_inches='tight')
    # plt.close()

    # ----------------------------
    # INDIVIDUAL PRECIP PLOT
    # ----------------------------
    fig, ax = plt.subplots(figsize=(18, 8))
    for ds in DATASET_ORDER:
        d = df_ppt[df_ppt["dataset"] == ds].sort_values("year")
        if not d.empty:
            ax.plot(d["year"], d["mm"], color=COLORS[ds], linewidth=2.5, label=ds)

    ax.set_title(f"{region} | Water-Year Total Precipitation", fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Precipitation (mm)", fontsize=14)
    ax.set_xlabel("Water Year", fontsize=14)
    ax.grid(True, which='both', color='grey', linestyle='-', alpha=0.4, linewidth=1.5)
    ax.set_ylim(int(np.floor((p_min - 100) / 250.0)) * 250, int(np.ceil((p_max + 100) / 250.0)) * 250)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45)
    ax.set_xlim(xlims)
    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = [(h, l) for ds in DATASET_ORDER for h, l in zip(handles, labels) if l == ds]
    if sorted_pairs:
        sh, sl = zip(*sorted_pairs)
        ax.legend(sh, sl, loc='upper center', ncol=len(sl), bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"wy_total_precip_{reg_suffix}_standardized.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_all():
    for region in REGIONS:
        df_ppt = load_and_clean_data(region=region)
        reg_suffix = region.replace(" ", "_")

        # # 1. Combined (Full timeline available)
        # plot_combined_panel(df_temp, df_ppt, region, OUT_DIR / f"wy_temp_precip_combined_{reg_suffix}_standardized.png")
        #
        # # 2. Combined (Starting from 1983)
        # plot_combined_panel(df_temp, df_ppt, region, OUT_DIR / f"wy_temp_precip_combined_{reg_suffix}_standardized_from1983.png", start_year=1983)

        # 3. Precipitation plot
        plot_individual(df_ppt, region)

    print(f"Generated precipitation plots for {REGIONS} in {OUT_DIR}")

if __name__ == "__main__":
    plot_all()
