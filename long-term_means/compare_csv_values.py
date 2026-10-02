"""
Compare precipitation values between two CSV files (Niteesh's vs User's version).

Analyzes differences in water year cumulative precipitation data and produces
a detailed comparison report with statistics.
"""

import os
import pandas as pd
import numpy as np

NITEESH_CSV = "/data0/hernanqd/plots_code/skagit-met/long-term_means/niteesh_version/wy_total_precip_skagit_regions_1983_2024_ALL.csv"
USER_CSV = "/data0/hernanqd/plots_code/skagit-met/long-term_means/cumulative_precipitation_plot/water_year_cumulative_precipitation.csv"
OUTPUT_DIR = "/data0/hernanqd/plots_code/skagit-met/long-term_means"


def load_and_normalize_data():
    """Load both CSVs and convert to comparable format."""
    # Load Niteesh's version (long format)
    niteesh_df = pd.read_csv(NITEESH_CSV)
    print(f"Loaded Niteesh's CSV: {len(niteesh_df)} rows")
    print(f"Columns: {niteesh_df.columns.tolist()}\n")

    # Load user's version (wide format)
    user_df = pd.read_csv(USER_CSV)
    print(f"Loaded User's CSV: {len(user_df)} rows")
    print(f"Columns: {user_df.columns.tolist()}\n")

    # Convert user's wide format to long format
    user_long = user_df.melt(
        id_vars=["WaterYear"],
        var_name="dataset_region",
        value_name="mm",
    )

    # Parse dataset_region into dataset and region
    user_long[["region", "dataset"]] = user_long["dataset_region"].str.rsplit(
        "_", n=1, expand=True
    )
    user_long = user_long.drop("dataset_region", axis=1)
    user_long = user_long.rename(columns={"WaterYear": "year"})
    user_long = user_long[["dataset", "region", "year", "mm"]]

    # Normalize region names to match Niteesh's format
    region_mapping = {
        "All": "Full Skagit",
        "Upper Skagit": "Upper Skagit",
        "Sauk": "Sauk",
        "Lower Skagit": "Lower Skagit",
    }
    user_long["region"] = user_long["region"].map(region_mapping)

    # Normalize dataset names
    dataset_mapping = {
        "CONUS": "CONUS404 (daily-osn)",
        "DAYMET": "Daymet v4 (Planetary Computer)",
        "GRIDMET": "GridMET",
        "PNNL": "PNNL (historical)",
        "PRISM": "PRISM",
        "UCLA": "UCLA ERA5 WRF d02 (daily NetCDF)",
    }
    user_long["dataset"] = user_long["dataset"].map(dataset_mapping)

    return niteesh_df, user_long


def compare_data(niteesh_df, user_long):
    """Compare the two datasets and identify differences."""
    # Merge on dataset, region, year
    comparison = niteesh_df.merge(
        user_long,
        on=["dataset", "region", "year"],
        how="outer",
        suffixes=("_niteesh", "_user"),
    )

    comparison["difference"] = comparison["mm_user"] - comparison["mm_niteesh"]
    comparison["abs_difference"] = np.abs(comparison["difference"])
    comparison["percent_diff"] = (
        (comparison["difference"] / comparison["mm_niteesh"]) * 100
    )

    return comparison


def generate_report(comparison):
    """Generate and print comparison report for common years and geometries only."""
    print("=" * 80)
    print("COMPARISON REPORT: Niteesh's vs User's CSV Values")
    print("(Common Years and Geometries Only)")
    print("=" * 80)

    # Only compare rows where both datasets have data
    valid_comparisons = comparison[
        (comparison["mm_niteesh"].notna()) & (comparison["mm_user"].notna())
    ].copy()

    print(f"\nTotal comparable rows (both datasets have data): {len(valid_comparisons)}")

    # Show which years and regions are being compared
    if len(valid_comparisons) > 0:
        years = sorted(valid_comparisons["year"].unique())
        regions = sorted(valid_comparisons["region"].unique())
        print(f"Years in comparison: {years}")
        print(f"Regions in comparison: {regions}\n")

    print("### OVERALL STATISTICS ###\n")
    if len(valid_comparisons) > 0:
        print(
            f"Mean absolute difference: {valid_comparisons['abs_difference'].mean():.2f} mm"
        )
        print(
            f"Max absolute difference: {valid_comparisons['abs_difference'].max():.2f} mm"
        )
        print(
            f"Median absolute difference: {valid_comparisons['abs_difference'].median():.2f} mm"
        )
        print(f"Std dev of difference: {valid_comparisons['abs_difference'].std():.2f} mm")

        # Rows with large differences (>50mm)
        large_diff = valid_comparisons[
            valid_comparisons["abs_difference"] > 50
        ].copy()
        print(f"\n### ROWS WITH LARGE DIFFERENCES (>50 mm) ###\n")
        print(f"Count: {len(large_diff)}\n")
        if len(large_diff) > 0:
            large_diff_sorted = large_diff.sort_values(
                "abs_difference", ascending=False
            )
            print(
                large_diff_sorted[
                    ["dataset", "region", "year", "mm_niteesh", "mm_user", "difference"]
                ].to_string(index=False)
            )

        # Differences by year and region
        print(f"\n### DIFFERENCES BY YEAR AND REGION ###\n")
        by_year_region = (
            valid_comparisons.groupby(["year", "region"])[
                ["abs_difference", "percent_diff"]
            ]
            .agg(["mean", "count"])
            .round(2)
        )
        print(by_year_region)

        # Differences by dataset
        print(f"\n### MEAN DIFFERENCE BY DATASET ###\n")
        by_dataset = (
            valid_comparisons.groupby("dataset")[["abs_difference", "percent_diff"]]
            .agg(["mean", "max", "count"])
            .round(2)
        )
        print(by_dataset)

        # Differences by region
        print(f"\n### MEAN DIFFERENCE BY REGION ###\n")
        by_region = (
            valid_comparisons.groupby("region")[["abs_difference", "percent_diff"]]
            .agg(["mean", "max", "count"])
            .round(2)
        )
        print(by_region)

    # Summary table organized by year, region, dataset
    if len(valid_comparisons) > 0:
        print("\n### DETAILED COMPARISON BY YEAR, REGION, AND DATASET ###\n")
        summary = valid_comparisons[
            ["year", "region", "dataset", "mm_niteesh", "mm_user", "difference"]
        ].copy()
        summary["difference"] = summary["difference"].round(2)
        summary = summary.sort_values(["year", "region", "dataset"])
        print(summary.to_string(index=False))

    # Missing data
    missing_niteesh = comparison[comparison["mm_niteesh"].isna()]
    missing_user = comparison[comparison["mm_user"].isna()]

    print(f"\n### DATA AVAILABILITY ###\n")
    print(f"Rows in User's CSV but not in Niteesh's: {len(missing_niteesh)}")
    print(f"Rows in Niteesh's CSV but not in User's: {len(missing_user)}")

    if len(missing_niteesh) > 0:
        print("\nYears/Regions in User's CSV but missing in Niteesh's:")
        print(
            missing_niteesh[["dataset", "region", "year"]]
            .drop_duplicates()
            .sort_values(["year", "region"])
            .to_string(index=False)
        )

    if len(missing_user) > 0:
        print("\nYears/Regions in Niteesh's CSV but missing in User's:")
        print(
            missing_user[["dataset", "region", "year"]]
            .drop_duplicates()
            .sort_values(["year", "region"])
            .to_string(index=False)
        )

    # Save detailed comparison to CSV
    output_path = os.path.join(OUTPUT_DIR, "comparison_detailed.csv")
    comparison.to_csv(output_path, index=False)
    print(f"\n### OUTPUT ###\nDetailed comparison saved to: {output_path}")

    return comparison


if __name__ == "__main__":
    niteesh_df, user_long = load_and_normalize_data()
    comparison = compare_data(niteesh_df, user_long)
    generate_report(comparison)
