"""Total precipitation per water year for six products and four basin areas (CSV).

For every water year from 1983 to 2020 (WY N = 1 Oct of year N-1 to 30 Sep of
year N), daily basin-mean precipitation is extracted from six gridded
products (PRISM, PNNL, Daymet, CONUS404, UCLA, GridMET) and summed over the
water year. This is done separately for the Upper Skagit, Sauk and Lower
Skagit HUC8 sub-basins and for the three combined ("All"). The result is one
row per water year with one column per <area>_<PRODUCT> pair.

Daily data come from common/prism_day.py (shared with step 3 and the other plot
scripts). With ALIGN = True all products are on the PRISM day (12 UTC on D-1 to
12 UTC on D; GitHub issue #61), so a water year runs from 12 UTC on Sep 30 to
12 UTC on Sep 30. With ALIGN = False each product's own date labels are used.
A water-year total is the sum of the days available; days missing in a product
(e.g. Daymet's Dec 31 in leap years) are left out, not filled.

This CSV is the input of plot_water_year_cumulative_data.py and
plot_water_year_cumulative_panel.py.

Input
    Precipitation datasets - see the paths in common/prism_day.py
    data/GIS/SkagitSubBasin_HUC8.geojson
Output
    plots/precipitation_long_term_means/water_year_cumulative_precipitation.csv

Usage
    python water_year_cumulative.py      (slow: reads every day of 38 water years)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Shared daily loaders (common/prism_day.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import prism_day  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV_NAME = 'water_year_cumulative_precipitation.csv'

PRODUCTS = ['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']

# Day alignment of the products (GitHub issue #61): True = all products on the PRISM
# day (12 UTC on D-1 to 12 UTC on D); False = each product's own date labels (as before)
ALIGN = True

# Sub-basins (HUC8 names), and the areas written to the CSV ('All' = the three combined)
SUBBASINS = ["Upper Skagit", "Sauk", "Lower Skagit"]
GEOMETRIES = SUBBASINS + ['All']

# Water years to process (water year is Oct 1 - Sep 30)
# Water year 1983 = Oct 1, 1982 to Sep 30, 1983
WATER_YEAR_START = 1983
WATER_YEAR_END = 2020


def get_water_year_dates(water_year):
    """Daily dates of a water year: Oct 1 of the previous calendar year to Sep 30."""
    start_date = pd.Timestamp(year=water_year - 1, month=10, day=1)
    end_date = pd.Timestamp(year=water_year, month=9, day=30)
    return pd.date_range(start_date, end_date, freq='D')


def water_year_totals(water_year, products=PRODUCTS):
    """Water-year total of each product for each area in GEOMETRIES.

    Each product is loaded once and averaged over every area's mask.
    Returns {"<area>_<PRODUCT>": total}; areas/products without data are omitted.
    """
    dates = get_water_year_dates(water_year)
    totals = {}
    for product in products:
        da = prism_day.load_daily(product, dates, align=ALIGN)
        if da is None:
            print(f"    {product}: no data")
            continue
        print(f"    {product}: {da.sizes['time']}/{len(dates)} days")
        for geometry in GEOMETRIES:
            regions = SUBBASINS if geometry == 'All' else [geometry]
            daily = prism_day.basin_mean(da, prism_day.basin_mask(da, regions))
            total = daily.sum()  # NaN days are skipped
            if daily.notna().any() and not np.isnan(total):
                totals[f"{geometry}_{product.upper()}"] = total
    return totals


def create_water_year_cumulative_csv():
    """Extract and save the water-year totals of every product and area to CSV."""
    print(f"Extracting water year data from {WATER_YEAR_START} to {WATER_YEAR_END} (ALIGN={ALIGN})...")
    water_years = range(WATER_YEAR_START, WATER_YEAR_END + 1)

    csv_data = {}
    for wy in water_years:
        print(f"  Water year {wy}...")
        csv_data[wy] = water_year_totals(wy)

    # Export to CSV
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_df = pd.DataFrame(csv_data).T
    csv_df.index.name = 'WaterYear'
    csv_df = csv_df.sort_index(axis=1)
    csv_df.to_csv(csv_path)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    create_water_year_cumulative_csv()
