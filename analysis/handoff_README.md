# Handoff README

For a notebook-by-notebook list, see:
- [`analysis/notebook_index.md`](/home/balaji24/skagit-met/analysis/notebook_index.md)

## What This Folder Is

This folder contains the analysis notebooks I used for the Skagit basin work including:

- precipitation comparison across products
- temperature comparison across products
- atmospheric river event analysis
- seasonal and water-year summaries
- sub-basin analysis for Upper Skagit, Sauk, and Lower Skagit
- data preparation and dataset checks for some sources

Not every notebook here is equally polished. Some are final enough to reuse, some are still working notebooks, and a few are just scratch notebooks that I kept because they contain useful logic.

Important:
- most of the product downloader scripts are under `scripts/`, not under `analysis/`
- `analysis/` is mainly for notebooks and a few prep/QC utilities
- the data that used to live under `/data0/balaji24/` has been moved under `/data0/skagit_met/data_transfer/`

## Current Folder Structure

The folder is partly reorganized already.

- `analysis/data_prep/`
  - extraction and utility notebooks/scripts
- `analysis/data_checks/`
  - variable checks, QC, and temporal-resolution checks
- `analysis/event_analysis/`
  - AR-event comparison notebooks
- `analysis/regional_analysis/`
  - seasonal, water-year, and sub-basin notebooks
- `analysis/full_timeline_analysis/`
  - long-timeline precip and temperature notebooks
- `analysis/summary_stats/`
  - long-record summary metrics and plotting notebooks
- `analysis/exploration/`
  - scratch notebooks


## Where To Start

If someone is trying to understand the main analysis workflow, I would start here:

1. [`analysis/notebook_index.md`](/home/balaji24/skagit-met/analysis/notebook_index.md)
2. notebooks in `analysis/regional_analysis/`
3. notebooks in `analysis/event_analysis/`
4. notebooks in `analysis/summary_stats/`

If someone is trying to understand how data got prepared or clipped:

1. `analysis/data_prep/extraction/CONUS_Downloader.ipynb`
2. `analysis/data_prep/mask_regions/pnnl_skagit_basins_mask_creation.ipynb`
3. product downloader scripts under `scripts/`
4. CanESM2 shell downloaders and some copied prep utilities under `analysis/data_prep/downloads/`

If someone is trying to debug data issues:

1. `analysis/data_checks/temporal_resolution/AR_6h.ipynb`
2. `analysis/data_checks/variable_availability/temperature_var_check.ipynb`
3. `analysis/data_checks/source_qc/ornl_check.ipynb`

## What Feels Most Reusable

These are the notebooks that feel closest to “main workflow” status:

- `analysis/event_analysis/atmospheric_rivers/AR_Events.ipynb`
- `analysis/regional_analysis/seasonal/era5_analysis.ipynb`
- `analysis/regional_analysis/water_year/water_year_era5.ipynb`
- `analysis/full_timeline_analysis/full_timeline_ppt_plot.ipynb`
- `analysis/full_timeline_analysis/full_timeline_temperature_plot.ipynb`
- `analysis/summary_stats/stations/snotel_analysis.ipynb`
- `analysis/data_prep/extraction/CONUS_Downloader.ipynb`
- `analysis/data_prep/mask_regions/pnnl_skagit_basins_mask_creation.ipynb`

If someone needs the actual dataset download scripts, check:

- `scripts/conus_downloader.py`
- `scripts/daymet_pc_downloader.py`
- `scripts/hrrr_downloader.py`
- `scripts/ornl_downloader.py`
- `scripts/prism_downloader.py`
- `scripts/snotel_downloader.py`
- `scripts/wrf_downloader.py`

## What Still Needs Work


- `analysis/summary_stats/precipitation/ppt_stats.ipynb`
  - has some inconsistencies and probably needs cleanup before being treated as final

- `analysis/summary_stats/temperature/temp_stats.ipynb`
  - same story as the precipitation stats notebook


## Notes About Paths

Many notebooks use relative paths like:
- `../data/...`
- absolute paths under `/data0/...`

So before moving notebooks around again, check path usage carefully.

Important:
- some notebooks still rely on older absolute local paths under `/data0/balaji24/data`
- the current data location is under `/data0/skagit_met/data_transfer/`
- code paths may still need to be updated after the data move is fully finished

If the folder structure is cleaned up more later, path updates should be done at the same time.
