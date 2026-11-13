"""
Build a bias table of long-term mean water-year precipitation for each
dataset relative to PRISM, by region.

Mean Bias (mm) = PRISM mean - Dataset mean
%Bias          = Mean Bias / PRISM mean * 100

Outputs (written next to this script):
    precipitation_bias_table.csv
    precipitation_bias_table.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path("/data0/hernanqd/plots_code/skagit-met/skagit_met_for_publication/plots/precipitation_long_term_means")
INPUT_CSV = BASE_DIR / "water_year_cumulative_precipitation.csv"
OUT_CSV = BASE_DIR / "bias_table" /  "precipitation_bias_table.csv"
OUT_PNG = BASE_DIR / "bias_table" / "precipitation_bias_table.png"

REFERENCE = "PRISM"

# CSV region prefix -> label in the table (order kept)
REGIONS = {
    "All": "Full Skagit",
    "Upper Skagit": "Upper Skagit",
    "Sauk": "Sauk",
    "Lower Skagit": "Lower Skagit",
}

# CSV dataset suffix -> label in the table (order kept).
DATASETS = {
    "DAYMET": "Daymet",
    "GRIDMET": "gridMET",
    "UCLA": "UCLA",
    "PNNL": "PNNL hist",
    "CONUS": "CONUS 404",
}


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    means = df.drop(columns="WaterYear").mean()
    rows = []
    for region_key, region_label in REGIONS.items():
        ref_mean = means[f"{region_key}_{REFERENCE}"]
        for ds_key, ds_label in DATASETS.items():
            ds_mean = means[f"{region_key}_{ds_key}"]
            bias = ref_mean - ds_mean
            rows.append({
                "Region": region_label,
                "Dataset": ds_label,
                f"{REFERENCE} mean (mm)": ref_mean,
                "Dataset mean (mm)": ds_mean,
                "Mean Bias (mm)": bias,
                "%Bias": bias / ref_mean * 100,
            })
    return pd.DataFrame(rows).round(1)


def plot_table(table: pd.DataFrame, path: Path) -> None:
    display = table.copy()
    # Show the region name only on the first row of each block
    display.loc[display["Region"].duplicated(), "Region"] = ""
    cell_text = [
        [v if isinstance(v, str) else f"{v:.1f}" for v in row]
        for row in display.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(9, 0.3 * (len(display) + 1) + 0.4))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=list(display.columns),
                   loc="center", cellLoc="right", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.3)
    tbl.auto_set_column_width(list(range(len(display.columns))))

    for (r, c), cell in tbl.get_celld().items():
        if c < 2:
            cell.set_text_props(ha="left")
        if r == 0 or (c == 0 and cell.get_text().get_text()):
            cell.set_text_props(weight="bold")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    table = build_table(df)
    print(f"Water years {df['WaterYear'].min()}-{df['WaterYear'].max()}")
    print(table.to_string(index=False))
    table.to_csv(OUT_CSV, index=False)
    plot_table(table, OUT_PNG)
    print(f"\nSaved {OUT_CSV}\nSaved {OUT_PNG}")


if __name__ == "__main__":
    main()
