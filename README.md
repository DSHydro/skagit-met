# Skagit Basin Precipitation Analysis

Comprehensive analysis of precipitation products and atmospheric river events in the Skagit Basin.

## Overview

This repository contains reproducible workflows for:
- Comparing multiple precipitation products (daily and hourly)
- Quantifying bias against PRISM and Daymet baselines
- Analyzing high-impact atmospheric river (AR) events
- Generating manuscript-style visualizations

## Directory Structure

### Analysis Modules

- **`cumulative_precipitation_plot/`** - Cumulative precipitation analysis for AR and non-AR events
  - Daily and hourly-derived cumulative precipitation
  - Combined precipitation-streamflow plots
  - Top 50 AR vs non-AR event comparison

- **`multi_product_bulk_bias/`** - Bulk bias analysis across all precipitation products
  - Bias calculation against PRISM and Daymet baselines
  - Bias metrics (MAE, RMSE, correlation)
  - Manuscript-style bias plots with time period and event type breakdown

- **`spatial_plots/`** - Spatial visualizations of precipitation products
  - Basin-mean precipitation maps
  - Spatial extent and resolution comparisons

- **`houly_timeseries_plots/`** - Hourly precipitation timeseries analysis
  - Event-level hourly precipitation patterns
  - Product comparisons at high temporal resolution

- **`mask_comparison/`** - Sub-basin masking and spatial analysis
  - HUC8 sub-basin comparisons (Upper Skagit, Sauk, Lower Skagit)
  - Sensitivity analysis for basin boundaries

### Supporting Structure

- **`data/`** - Data preparation and file structure
  - Expected input data locations
  - GIS boundaries and reference data
  - Output data storage

- **`scripts/`** - Utility and helper scripts
  - Common functions for data processing
  - Download wrappers for external datasets

## Key Outputs

- `multi_product_bulk_bias/grouped_bias_3_metrics_2_periods.png` - Main bias comparison figure
- `cumulative_precipitation_plot/specific_ar_events_cumulative_precipitation.png` - AR event analysis
- `spatial_plots/` - Basin-scale precipitation maps

## Quick Start

1. Begin with `multi_product_bulk_bias/` for overall product bias assessment
2. Move to `cumulative_precipitation_plot/` for event-specific analysis
3. Explore `spatial_plots/` for geographic patterns
4. Check `houly_timeseries_plots/` for high-resolution event details

## Data Products Analyzed

**Daily:** PRISM, Daymet, GridMET, CONUS404, UCLA, PNNL

**Hourly:** CONUS404, UCLA, PNNL

**Discharge:** USGS gauge 12200500 (Skagit River at Marblemount)
