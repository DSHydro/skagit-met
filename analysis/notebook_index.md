# Notebook Index

This file is a quick map of the notebooks under `analysis/`.


This index covers the main notebooks and a few closely related prep files. It does not list generated plots, GIF frames, or `.ipynb_checkpoints`.

Note:
- most product downloader scripts live under `scripts/`, not under `analysis/`
- this index is mainly for analysis notebooks plus a few analysis-adjacent prep files
- data that used to be under `/data0/balaji24/` has been moved under `/data0/skagit_met/data_transfer/`
- some notebooks may still contain old absolute paths and may need path cleanup

## Data Prep

- `analysis/data_prep/extraction/CONUS_Downloader.ipynb`
  - Purpose: pulls CONUS404 data for Skagit and writes local derived Zarr outputs for precipitation and temperature.
  - Status: `final` but we switched to an online source for CONUS
  - Outputs: `conus404_skagit_precip_daily.zarr`, `conus404_skagit_temp_daily.zarr`

- `analysis/data_prep/mask_regions/pnnl_skagit_basins_mask_creation.ipynb`
  - Purpose: builds Upper Skagit, Sauk, and Lower Skagit masks on the PNNL 450x450 grid.
  - Status: `final`
  - Outputs: `skagit_huc8_3mask_pnnl_450x450.nc`

- `analysis/data_prep/downloads/download_canesm2.sh`
  - Purpose: bulk-downloads yearly hourly CanESM2 NetCDF files from NERSC into year-based folders from Dr. Xiaoding's server.
  - Status: `utility`

- `analysis/data_prep/downloads/download_canesm2_psfc.sh`
  - Purpose: bulk-downloads hourly CanESM2 `PSFC` NetCDF files from Dr. Xiaoding's server.
  - Status: `utility`

### Related scripts outside `analysis/`

- `scripts/conus_downloader.py`
  - Product downloader/prep script for CONUS-related data work.

- `scripts/daymet_pc_downloader.py`
  - Product downloader script for Daymet data from Planetary Computer.

- `scripts/hrrr_downloader.py`
  - Product downloader script for HRRR data.

- `scripts/ornl_downloader.py`
  - Product downloader script for ORNL data.

- `scripts/prism_downloader.py`
  - Product downloader script for PRISM data.

- `scripts/snotel_downloader.py`
  - Product downloader script for SNOTEL data.

- `scripts/wrf_downloader.py`
  - Product downloader script for WRF/UCLA-related data.

### Related prep files inside `analysis/`

- `analysis/data_prep/downloads/daymet_pc_downloader.py`
  - Copied/download-related Daymet prep utility currently stored under `analysis/data_prep/downloads/`.

## Data Checks

- `analysis/data_checks/temporal_resolution/AR_6h.ipynb`
  - Purpose: audits precipitation datasets to see which sources are hourly, daily, 6-hourly, or otherwise.
  - Status: `final`
  - Outputs: `dataset_time_resolution_audit_all_sources.csv`

- `analysis/data_checks/source_qc/ornl_check.ipynb`
  - Purpose: troubleshooting notebook for Daymet/ORNL access and cross-checks against UCLA-related totals.
  - Status: `working`

- `analysis/data_checks/variable_availability/temperature_var_check.ipynb`
  - Purpose: checks temperature variable availability, units, QC, and trial comparison logic across sources.
  - Status: `working`

## Event Analysis

- `analysis/event_analysis/atmospheric_rivers/AR_Events.ipynb`
  - Purpose: compares cumulative precipitation totals across datasets for selected AR events, with sub-basin plots and MAE against PRISM 4 km.
  - Status: `final`

- `analysis/event_analysis/atmospheric_rivers/ProductAnalysis_Precip.ipynb`
  - Purpose: older AR-event notebook that builds basin-mean cumulative precipitation curves across products.
  - Status: `old version of AR_Events.ipynb`

- `analysis/event_analysis/atmospheric_rivers/hrrr_gif.ipynb`
  - Purpose: creates an HRRR precipitation GIF over the Skagit basin for a selected event window.
  - Status: `final`
  - Outputs: GIF plus frame PNGs

## Regional Analysis

- `analysis/regional_analysis/seasonal/era5_analysis.ipynb`
  - Purpose: seasonal precipitation comparison across datasets for Upper Skagit, Sauk, Lower Skagit, and Full Skagit, with PRISM-based error summaries.
  - Status: `final`

- `analysis/regional_analysis/water_year/water_year_era5.ipynb`
  - Purpose: water-year precipitation comparison across datasets for the three Skagit sub-basins, plus later basin-wide summary plots.
  - Status: `final`

- `analysis/regional_analysis/subbasins/Sub_basin_analysis.ipynb`
  - Purpose: main working notebook for the 3-region Skagit sub-basin setup, including seasonal, water-year, event, and PNNL-specific analyses.
  - Status: `working`

## Summary Stats

- `analysis/summary_stats/precipitation/ppt_stats.ipynb`
  - Purpose: computes long-record precipitation summary metrics by dataset and region.
  - Status: `need to develop this (has some inconsistencies)`
  - Outputs: annual metrics CSV, summary metrics CSV, region summary tables

- `analysis/summary_stats/precipitation/Precipitation_Plots.ipynb`
  - Purpose: makes annual precipitation comparison plots, including all-dataset and PRISM 4 km vs 800 m views.
  - Status: `working`

- `analysis/summary_stats/temperature/temp_stats.ipynb`
  - Purpose: computes long-record air-temperature summary metrics for the three Skagit sub-basins.
  - Status: `need to develop this (has some inconsistencies)`
  - Outputs: annual metrics CSV, summary metrics CSV, region summary tables

- `analysis/summary_stats/stations/snotel_analysis.ipynb`
  - Purpose: computes SNOTEL water-year precipitation by station and groups the stations by Skagit sub-basin.
  - Status: `final`
  - Outputs: station-level CSV plus one plot per region

## Exploration

- `analysis/exploration/scratch/Playground.ipynb`
  - Purpose: general scratchpad for one-off checks across multiple datasets.
  - Status: `scratch`

- `analysis/exploration/scratch/playground1.ipynb`
  - Purpose: development notebook for AR-event comparison workflows, file availability checks, verification, and MAE logic.
  - Status: `scratch`

## Long Timeline

- `analysis/full_timeline_analysis/full_timeline_ppt_plot.ipynb`
  - Purpose: full 1983-2024 seasonal precipitation comparison across datasets, using 4 regions and windowed plots for readability.
  - Status: `final`
  - Outputs: seasonal precip CSVs, PRISM-based MAE CSVs, windowed plots

- `analysis/full_timeline_analysis/full_timeline_temperature_plot.ipynb`
  - Purpose: long-timeline temperature comparison notebook, combining water-year mean temperature and later seasonal/MAE workflows.
  - Status: `final`
