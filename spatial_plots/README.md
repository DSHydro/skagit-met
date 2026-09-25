# Spatial Plots: Skagit Basin Precipitation Visualization

This directory contains scripts for generating spatial visualizations of precipitation data and biases across multiple weather datasets for the Skagit Basin.

## Overview

The spatial_plots module produces maps comparing precipitation estimates from multiple weather datasets (PRISM, Daymet, PNNL, CONUS404, UCLA, GridMET) against each other and relative to a chosen baseline (PRISM). Visualizations focus on:

- **Atmospheric River (AR) events**: Cumulative precipitation during extreme weather events
- **Seasonal patterns**: Average precipitation across seasons (Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec)
- **Percentiles**: Spatial distribution of precipitation percentiles
- **Bias comparisons**: Product-specific and hourly vs. daily calculation differences

## Directory Structure

```
spatial_plots/
├── README.md                                    # This file
├── *.py                                         # Plotting scripts
├── plots/                                       # Output spatial maps (PNG figures)
│   ├── plot_spatial_seasonal_avg_prism/        # Seasonal average maps and biases
│   └── plot_spatial_percentiles/               # Percentile distribution maps
├── hourly_means/                               # Intermediate hourly precipitation data
├── daily_versions/                             # Daily precipitation calculations
│   ├── daily_means/
│   └── plots/
├── comparison_results/                         # Comparison outputs (8-day events)
└── comparison_results_3days/                   # Comparison outputs (3-day events)
```

## Key Scripts

### AR Event Visualizations (8-day windows)
- **plot_ar_events_cumulative_spatial_grid_8_days_updated.py**
  - 6×6 grid: rows = AR events, columns = weather products
  - Shows raw cumulative precipitation (T-2 to T+5 days) for selected AR events
  - Outputs: PNG to `plots/`

- **plot_ar_events_cumulative_spatial_bias_grid_8_days_updated.py**
  - Same structure as above but shows product-minus-PRISM bias (%)
  - Highlights systematic differences in precipitation estimates

### AR Event Visualizations (3-day windows)
- **plot_ar_events_cumulative_spatial_grid_3_days_updated.py**
  - Same as 8-day version but with shorter time windows
  - Focuses on peak precipitation periods

- **plot_ar_events_cumulative_spatial_bias_grid_3_days_updated.py**
  - Bias grid for 3-day AR events

### Non-AR Event Comparisons
- **plot_non_ar_events_cumulative_spatial_grid_8_days_updated.py**
  - 8-day cumulative precipitation for non-AR precipitation events
  - Compares baseline vs. alternative products

- **plot_non_ar_events_cumulative_spatial_grid_3_days_updated.py**
  - 3-day version of non-AR event comparison

- **plot_non_ar_events_cumulative_spatial_bias_grid_8_days_updated.py**
- **plot_non_ar_events_cumulative_spatial_bias_grid_3_days_updated.py**
  - Bias grids for non-AR events

### Seasonal and Statistical Analysis
- **plot_spatial_seasonal_avg.py**
  - Generates two publication-quality maps:
    1. Raw seasonal average precipitation from each product (1981–2019)
    2. Product-minus-PRISM bias maps
  - Outputs: PNG to `plots/plot_spatial_seasonal_avg_prism/`

- **plot_spatial_percentiles.py**
  - Spatial distribution of precipitation percentiles across products
  - Outputs: PNG to `plots/plot_spatial_percentiles/`

### Hourly vs. Daily Bias Differences
- **plot_bias_difference_grid_hourly_vs_daily.py**
  - 6×5 grid showing the DIFFERENCE in bias calculations (hourly - daily)
  - Highlights which products have meaningful differences between hourly and daily approaches

- **plot_bias_difference_grid_hourly_vs_daily_3days.py**
  - Same as above but for 3-day AR event windows

## Input Data Sources

Scripts expect data from:
- **VAULT_DIR**: `/data0/skagit_met/data_transfer/data/` (PRISM, Daymet, PNNL, CONUS404, UCLA, GridMET)
- **GIS data**: Skagit Basin boundary and subbasin shapefiles
- **Event lists**: AR event dates and time windows (defined in script configuration)

## Output Locations

- **plots/**: Primary output directory for all PNG figures
- **hourly_means/**: Cached hourly-averaged intermediate data
- **daily_versions/**: Daily aggregations and related plots
- **comparison_results/**: Metadata from 8-day event comparisons
- **comparison_results_3days/**: Metadata from 3-day event comparisons

