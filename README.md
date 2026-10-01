# skagit-met

Exploring meteorology data for use in hydrologic modeling of the Skagit River basin

This repository hosts code to construct analysis ready data cubes for the Skagit River basin from a variety of model and observational datasets. There are also Jupyter Notebooks with detailed analysis and figures.

## Datasets
|Dataset          |Spatial Resolution     |Range/Availability|Data Granularity|Temperature|Precipitation|Wind Speed            |Relative Humidity         |Long Wave Radiation|Short Wave Radiation|
|-----------------|-----------------------|------------------|----------------|-----------|-------------|----------------------|--------------------------|-------------------|--------------------|
|[HRRRv4](https://rapidrefresh.noaa.gov/hrrr/)             |3 km                   |2014 - present    |Hourly          |✔️         |✔️           |✔️                    |✔️                        |✔️                 |✔️                  |
|[PRISM](https://www.prism.oregonstate.edu/)            |4 km or 800 m                   |1981 - present    |Daily           |✔️         |✔️           |X                     |Via Vapor Pressure Deficit|X                  |X                   |
|[UCLA CMIP-6 (WRF)](https://dept.atmos.ucla.edu/alexhall/downscaling-cmip6)|9 km                   |1980 - 2100       |Hourly          |✔️         |✔️           |Via U and V components|Via Specific Humidity     |✔️                 |✔️                  |
|[ORNL](https://hydrosource.ornl.gov/data/datasets/9505v3_1/)             |4 km                   |1980-2040         |Daily           |✔️         |✔️           |✔️                    |✔️                        |✔️                 |✔️                  |
|[SNOTEL](https://metloom.readthedocs.io/en/latest/)           |Point Data (9 stations)|2010 - present    |Hourly          |✔️         |✔️           |X                     |X                         |X                  |X                   |
|PNNL*            |6 km                   |1981 - 2020       |Hourly          |✔️         |✔️           |Via U and V components|Via Specific Humidity     |✔️                 |✔️                  |
|CONUS404 (HyTest)            |4 km                   |2014 - 2020       |Hourly/Daily         |✔️         |✔️           |✔️ |✔️      |✔️                 |✔️                  |

*This is a private experimental dataset hosted at the Pacific Northwest National Laboratory

## Environment Setup
We recommend using [pixi](https://pixi.sh/latest/), an alternate to conda environments that uses a "per-project" rather than "global" environment paradigm. Pixi helps isolate software environments specific to a repository through the use of configuration and lock files to manage dependencies all in one, but backed by the conda-forge repos and pypi.

To use pixi, install it on your machine, clone this repo, then run `pixi install` in the root of the repo.
You may get an error big and long that mentions 'clang' or something like that. If that's the case run `export CFLAGS="-Wno-incompatible-function-pointer-types -Wno-implicit-function-declaration"` and try again. This is pre-run as a part of the setup script for the conda environment.

To play around with the jupyter notebooks, just run `pixi run nb`, and a local instance of jupyter will be launched for you with all the necessary packages and dependencies. It's quite magic.

### DSHydro JupyterHub

[The DSHydro JupyterHub](https://dshydro.ce.washington.edu/jupyter/hub) is a limited-access JupyterHub running on servers at the University of Washington.

1. Run `curl -fsSL https://pixi.sh/install.sh | bash` to install pixi for your user
2. Install the data-download environment or analysis environment using `pixi install -e download-data` and `pixi install -e analysis`
3. To look at the analysis notebook, install the analysis kernel with the following command: `./skagit-met/.pixi/envs/analysis/bin/python3 -m ipykernel install --user --name=skagit_analysis`
4. Once installed, open the .ipynb file you wish to look at, and select the skagit_analyis kernel.

### Cloud-hosted JupyterHub:

[Cryocloud](https://book.cryointhecloud.com/content/Getting_Started.html) is a NASA-supported JupyterHub running in AWS us-west-2. The environment setup is the same as above!

### Conda environment:
If you prefer to use conda, there is also a script to install necessary packages in a global conda environment
1. Clone this repo
2. In the root of the repo, run `./setup.sh` - this creates the conda environment, installs whats needed, etc.
3. Enter the environment using `conda activate skagit-met`
4. If you're done, don't forget to `conda deactivate`.

## Additional Resources
* https://rapidrefresh.noaa.gov/Diag-vars-NOAA-TechMemo.pdf
* https://rapidrefresh.noaa.gov/hrrr/HRRR/Welcome.cgi?dsKey=hrrr_ncep_jet
* https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
* https://www.nco.ncep.noaa.gov/pmb/docs/on388/table2.html
* https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2.shtml

## Precipitation analysis (plots and bias)

Comprehensive analysis of precipitation products and atmospheric river events in the Skagit Basin.

### Overview

This repository contains reproducible workflows for:
- Comparing multiple precipitation products (daily and hourly)
- Quantifying bias against PRISM and Daymet baselines
- Analyzing high-impact atmospheric river (AR) events
- Generating manuscript-style visualizations

### Directory Structure

#### Analysis Modules

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

#### Supporting Structure

- **`data/`** - Data preparation and file structure
  - Expected input data locations
  - GIS boundaries and reference data
  - Output data storage

- **`scripts/`** - Utility and helper scripts
  - Common functions for data processing
  - Download wrappers for external datasets

### Key Outputs

- `multi_product_bulk_bias/grouped_bias_3_metrics_2_periods.png` - Main bias comparison figure
- `cumulative_precipitation_plot/specific_ar_events_cumulative_precipitation.png` - AR event analysis
- `spatial_plots/` - Basin-scale precipitation maps

### Quick Start

1. Begin with `multi_product_bulk_bias/` for overall product bias assessment
2. Move to `cumulative_precipitation_plot/` for event-specific analysis
3. Explore `spatial_plots/` for geographic patterns
4. Check `houly_timeseries_plots/` for high-resolution event details

### Data Products Analyzed

**Daily:** PRISM, Daymet, GridMET, CONUS404, UCLA, PNNL

**Hourly:** CONUS404, UCLA, PNNL

**Discharge:** USGS gauge 12200500 (Skagit River at Marblemount)
