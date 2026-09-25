# Daily Versions of Spatial Plot Scripts

This directory contains the original versions of the 8 spatial plot scripts that use **daily** versions of UCLA and CONUS404 datasets (before updating to hourly versions).

## Files

1. **plot_ar_events_cumulative_spatial_bias_grid_3_days_daily.py** - AR events bias maps (3-day windows)
2. **plot_ar_events_cumulative_spatial_bias_grid_8_days_daily.py** - AR events bias maps (8-day windows)
3. **plot_ar_events_cumulative_spatial_grid_3_days_daily.py** - AR events precipitation maps (3-day windows)
4. **plot_ar_events_cumulative_spatial_grid_8_days_daily.py** - AR events precipitation maps (8-day windows)
5. **plot_non_ar_events_cumulative_spatial_bias_grid_3_days_daily.py** - Non-AR events bias maps (3-day windows)
6. **plot_non_ar_events_cumulative_spatial_bias_grid_8_days_daily.py** - Non-AR events bias maps (8-day windows)
7. **plot_non_ar_events_cumulative_spatial_grid_3_days_daily.py** - Non-AR events precipitation maps (3-day windows)
8. **plot_non_ar_events_cumulative_spatial_grid_8_days_daily.py** - Non-AR events precipitation maps (8-day windows)

## Key Differences from Updated Versions

### CONUS404 Loading
- **Daily version**: Loads from pre-computed daily zarr file `conus404_skagit_precip_daily_full.zarr`
- **Hourly version**: Loads hourly netCDF files and sums them

### UCLA Loading
- **Daily version**: Loads from daily netCDF files (structure: auxhist files without precipitation increments)
- **Hourly version**: Loads hourly netCDF files, calculates increments using RAINC+RAINNC.diff(), then sums

## Usage

These scripts can be run independently to generate plots using the daily data approach. They are provided for comparison with the updated hourly versions.
