# Cumulative Precipitation Plot - Skagit Basin AR Events

This directory contains analysis and visualization of cumulative precipitation timeseries from multiple weather products during high-impact atmospheric river (AR) events in the Skagit Basin. The analysis compares various precipitation datasets including hourly data (CONUS404, UCLA, PNNL) and daily data (PRISM, Daymet, GridMET, CONUS404).

## Overview

The scripts extract precipitation data over the Skagit Basin for specific atmospheric river events, compute cumulative precipitation, and generate comparative visualizations. Each plot includes:
- Cumulative precipitation time series from multiple products
- AR scale classification
- Peak streamflow discharge from USGS gauge 12200500

## Directory Structure

```
cumulative_precipitation_plot/
├── README.md                                          # This file
├── plot_specific_ar_and_non_ar_events/               # AR and non-AR event analysis
│   ├── plot_specific_ar_events.py                    # Extract and plot daily-based cumulative precipitation for AR events
│   ├── plot_specific_non_ar_events.py                # Extract and plot cumulative precipitation for non-AR events
│   ├── plot_specific_ar_events_and_streamflow.py     # Combined precipitation and streamflow analysis
│   ├── specific_ar_events_cumulative_precipitation.png
│   ├── specific_non_ar_events_cumulative_precipitation.png
│   └── streamflow_and_specific_ar_events_cumulative_precipitation.png
├── daily_cumulative_from_hourly/                     # Hourly-derived cumulative precipitation
│   ├── plot_hourly_cumulative_ar_events.py           # Extract and plot hourly-based cumulative precipitation
│   └── hourly_cumulative_ar_events.png
├── comparing_daily_vs_hourly/                        # Comparison of hourly-derived vs daily-derived data
│   ├── compare_cumulative_data.py                    # Compare hourly-derived vs daily-derived cumulative precipitation
│   └── comparison_results.csv                        # Summary of hourly vs daily percent differences
├── top_50_ar_vs_non-AR/                              # Top 50 AR vs non-AR event comparison
│   ├── ar_vs_non_ar_comparison.py                    # Compare top 50 AR and non-AR events
│   ├── ar_vs_non_ar_peak_discharge.py                # Analyze peak discharge for AR vs non-AR events
│   └── comparison.ipynb                              # Exploratory comparison notebook
├── timeseries_data/                                  # Output directory for CSV timeseries data
│   ├── cumulative_precip_*.csv                       # Daily cumulative precipitation from daily data
│   ├── cumulative_from_hourly_precip_*.csv           # Daily cumulative precipitation computed from hourly data
│   └── hourly_precip_*.csv                           # Hourly precipitation increments
├── old_code/                                         # Archived versions of previous scripts
└── exploring/                                        # Exploratory analysis outputs
```

## Key Scripts

### AR & Non-AR Event Analysis

#### `plot_specific_ar_and_non_ar_events/plot_specific_ar_events.py`
Extracts precipitation from daily data sources and plots cumulative precipitation for 6 high-impact AR events.

**Products analyzed:**
- PRISM 
- PNNL 
- Daymet 
- CONUS404 
- UCLA 
- GridMET 

#### `plot_specific_ar_and_non_ar_events/plot_specific_non_ar_events.py`
Analyzes precipitation for non-AR events (for contrast with AR events).

#### `plot_specific_ar_and_non_ar_events/plot_specific_ar_events_and_streamflow.py`
Creates a combined visualization showing both cumulative precipitation and USGS streamflow discharge on dual axes.

### Hourly-Derived Cumulative Precipitation

#### `daily_cumulative_from_hourly/plot_hourly_cumulative_ar_events.py`
Extracts hourly precipitation data and derives daily cumulative values. Allows comparison of hourly-derived vs daily-derived precipitation.

**Products analyzed:**
- CONUS404 (hourly WRF)
- UCLA (hourly WRF)
- PNNL (hourly WRF)

### Comparison & Analysis

#### `comparing_daily_vs_hourly/compare_cumulative_data.py`
Compares cumulative precipitation derived from hourly vs daily data for the same events. Computes mean percent differences by product across all events.

**Output:** `comparing_daily_vs_hourly/comparison_results.csv` - Mean percent difference for each event and product

#### `top_50_ar_vs_non-AR/ar_vs_non_ar_comparison.py`
Analyzes and compares the top 50 AR and non-AR events by cumulative precipitation.

#### `top_50_ar_vs_non-AR/ar_vs_non_ar_peak_discharge.py`
Analyzes peak discharge differences between top AR and non-AR events.

## Required Data Sources

### Daily precipitation data:
- **PRISM** - `/data0/skagit_met/data_transfer/data/prism_new_hq/` (TIFF format zips)
- **Daymet** - `/data0/skagit_met/data_transfer/data/daymet_new_hq/` (netCDF)
- **CONUS404** - `/data0/hernanqd/plots_code/skagit_basin_de/data/weather_data/conus404_skagit_precip_daily_full.zarr`
- **UCLA** - `/data0/skagit_met/data_transfer/data/ucla_era5_d02_daily/` (netCDF)
- **GridMET** - `/data0/skagit_met/data_transfer/data/gridmet/` (Zarr)

### Hourly precipitation data:
- **CONUS404** - `/data0/hernanqd/instance_2021_data/preparing_datasets/CONUS404/hourly_ar_non_ar_events/`
- **UCLA** - `/data0/hernanqd/instance_2021_data/hourly_ar_non_ar_events/`
- **PNNL** - `/data0/skagit_met/data_transfer/data/PNNL/historical/`

### Auxiliary data:
- **HUC8 sub-basin geometry** - `../data/GIS/SkagitSubBasin_HUC8.geojson`
- **Events metadata** - `../multi_product_bulk_bias/outputs/4_clean_bias_table.csv`
- **USGS discharge data** - `/data0/nksp2/skagit/skagit_2/skagit-met/experiments_2/data/usgs_12200500_discharge.rdb`

## Key Methods

### Spatial Masking
- Uses HUC8 sub-basin boundaries (Upper Skagit, Sauk, Lower Skagit) to mask precipitation grids
- Computes basin-mean precipitation using regionmask
- Handles multiple coordinate systems and grid resolutions

### Cumulative Computation
- **Daily data:** Directly sums daily precipitation values
- **Hourly data:** Sums hourly increments, then resamples to daily totals
- Cumulative sum computed as running total over the event window

### Basin Mean Calculation
Accounts for 2D/3D data arrays, handles NaN values, and applies spatial masks consistently across products.

## Output Files

### Visualizations
PNG files located in their respective subdirectories:
- `plot_specific_ar_and_non_ar_events/*.png` - AR and non-AR event plots
- `daily_cumulative_from_hourly/*.png` - Hourly-derived cumulative precipitation plots

### Timeseries Data (`timeseries_data/`)
- `cumulative_precip_YYYYMMDD.csv` - Daily cumulative precipitation from daily data
  - Columns: `datetime`, `conus`, `ucla`, `pnnl`, `prism`, `daymet`, `gridmet`
- `cumulative_from_hourly_precip_YYYYMMDD.csv` - Daily cumulative from hourly-derived data
  - Columns: `datetime`, `CONUS_daily_cumsum_mm`, `UCLA_daily_cumsum_mm`, `PNNL_daily_cumsum_mm`
- `hourly_precip_YYYYMMDD.csv` - Hourly precipitation increments
  - Columns: `datetime`, `CONUS_hourly_precip_mm`, `UCLA_hourly_precip_mm`, `PNNL_hourly_precip_mm`

### Comparison Data
- `comparison_results.csv` - Mean percent difference (hourly vs daily) for each event/product
  - Formula: `((hourly - daily) / daily) * 100`
