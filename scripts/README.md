# Data download scripts

## hrrr_downloader.py
To download bulk data, we have a python module/script that can be run.

It:
1. Downloads select parameters from hrrr archives over a specified  using fast herbie for parallel computation
2. Geographically subsets that downloaded data using a provided geojson (geojson polygon boundary - see skagit_boundaries.json for more)
3. Cleans up all data that is not geographically subsetted
4. Saves the data as a zarr store to be read and manipulated - see hrr_model_downloader_notebook.ipynb for example usage

When done this way, each day of data takes only a few MB of disk space.

To run:
1. Activate the conda environment in the root of this repo (see setup above)
2. Run from the command line using the following command - be sure to adjust the dates and parameters as needed -  `pixi run hrrr --startDate 2023-02-01 --endDate 2023-02-08 --parameters 'TMP:surface,RH:2 m above ground,WIND:10 m above ground,APCP:surface:0-1 hour acc fcst,DSWRF:surface,DLWRF:surface'`
3. For help with parameters, run `pixi run hrrr -h`

## wrf_downloader.py
This script downloads and formats bulk, downscaled WRF output data from the [UCLA downscaled cmip6 archive](https://dept.atmos.ucla.edu/alexhall/downscaling-cmip6). You can read more about the data tiers and various domains [here](https://dept.atmos.ucla.edu/sites/default/files/alexhall/files/aws_tiers_dirstructure_nov22.pdf)

Similar to the hrr_downloader script, it:
1. Downloads select parameters from UCLA archives over a specified date range using multi-theraded connections to AWS.
2. Geographically subsets that downloaded data using a provided geojson (geojson polygon boundary - see skagit_bound_poly.json for more)
3. Uses masking to establish boundaries
4. Saves the data as a zarr store to be read and manipulated - see WRF_Downloader.ipynb for example usage

All 22 variables for WRF are about 56MB for a weeks worth of data, so 2.9GB as a zarr per year for that hourly data.

To run:
1. Activate the conda environment in the root of this repo (see setup above) or have pixi installed
2. Run from the command line using the following command - be sure to adjust the dates and parameters as needed - `python wrf_downloader.py --model cesm2_r11i1p1f1_ssp245 --startDate 2023-01-01 --endDate 2023-01-08`
3. If using pixi, run `pixi run wrf --model cesm2_r11i1p1f1_ssp245 --startDate 2023-01-01 --endDate 2023-01-08 --outputDir data/weather_data/`
3. For help with parameters, run pixi run wrf_downloader.py -h`

## prism_downloader.py
This script downloads and formats bulk, downscaled PRISM output data from the [PRISM archives](https://www.prism.oregonstate.edu/). You can read more about the data [here](https://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf).

Similar to the the other scripts, it:
1. Downloads select parameters from PRISM archives over a specified date range using multi-theraded connections to PRISM FTP servers.
2. Geographically subsets that downloaded data using a provided geojson
3. Uses masking to establish boundaries
4. Saves the data as a zarr store to be read and manipulated - see PRISM_Downloader.ipynb for example usage

To run:
1. Activate the conda environment in the root of this repo (see setup above) or have pixi installed
2. Run from the command line using the following command - be sure to adjust the dates and parameters as needed - `python prism_downloader.py --startDate 2023-01-01 --endDate 2023-01-08 --outputDir data/weather_data/`
3. If using pixi, run `pixi run prism --startDate 2023-01-01 --endDate 2023-01-08 --outputDir data/weather_data/`
3. For help with parameters, run `pixi run prism_downloader.py -h`

## snotel_downloader.py
This script downloads and formats bulk SNOTEL data using metloom. You can read more about metloom [here](https://metloom.readthedocs.io/en/latest/).

Similar to the the other scripts, it:
1. Downloads select parameters from SNOTEL archives over a specified date range using metloom, given Snotel site ids or a geojson to specify the boundaries where you want to get snotel sites within.
2. Saves the data as a zarr store to be read and manipulated.

To run:
1. Using pixi: `pixi run snotel --frequency daily --startDate 2023-01-01 --endDate 2023-01-08 --outputDir data/weather_data/`
2. For help with parameters, run `pixi run snotel_downloader.py -h`
