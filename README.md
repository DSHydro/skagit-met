# skagit-met
Exploring meteorology data for use in hydrologic modeling of the Skagit River basin

# Datasets
|Dataset          |Spatial Resolution     |Range/Availability|Data Granularity|Temperature|Precipitation|Wind Speed            |Relative Humidity         |Long Wave Radiation|Short Wave Radiation|
|-----------------|-----------------------|------------------|----------------|-----------|-------------|----------------------|--------------------------|-------------------|--------------------|
|HRRR             |3 km                   |2014 - present    |Hourly          |✔️         |✔️           |✔️                    |✔️                        |✔️                 |✔️                  |
|PRISM            |4 km                   |1981 - present    |Daily           |✔️         |✔️           |X                     |Via Vapor Pressure Deficit|X                  |X                   |
|UCLA CMIP-6 (WRF)|9 km                   |1980 - 2100       |Hourly          |✔️         |✔️           |Via U and V components|Via Specific Humidity     |✔️                 |✔️                  |
|ORNL             |4 km                   |1980-2040         |Daily           |✔️         |✔️           |✔️                    |✔️                        |✔️                 |✔️                  |
|SNOTEL           |Point Data (9 stations)|2010 - present    |Hourly          |✔️         |✔️           |X                     |X                         |X                  |X                   |
|PNNL            |6 km                   |1981 - 2020       |Hourly          |✔️         |✔️           |Via U and V components|Via Specific Humidity     |✔️                 |✔️                  |


# Setup
This repo relies on conda environments - I recommend [miniforge](https://github.com/conda-forge/miniforge). I also fast explored/protoyped herbie and other hrrr data in jupyter/ipython notebooks. 

My setup installs jupyter in my base conda environment, then runs notebooks in the sub-envirionment kernels. You can do the same with:

```
conda install jupyterlab
conda install nb_conda_kernels
```

From there run, `jupyter-lab` in your base environment, and you should see the `skagit-met` kernel as an option

## Alternate option
Thanks to Scott, this works really well with [pixi](https://pixi.sh/latest/), an alternate to conda environments that uses a project rather than environment paradigm. Pixi helps isolate and run entire projects through the use of configuration and lock files to manage dependencies all in one, but backed by the conda-forge repos and pypi. 

To use pixi, install it on your machine, clone this repo, then run `pixi install` in the root of the repo.
You may get an error big and long that mentions 'clang' or something like that. If that's the case run `export CFLAGS="-Wno-incompatible-function-pointer-types -Wno-implicit-function-declaration"` and try again. This is pre-run as a part of the setup script for the conda environment. 

To play around with the jupyter notebooks, just run `pixi run nb`, and a local instance of jupyter will be launched for you with all the necessary packages and dependencies. It's quite magic. 

### Assumptions:
1. I assume you have conda, and all its dependencies installed, and a base conda environment going
2. I assume you are using bash, but if you're using zsh or another shell, this will run better if you modify setup.sh to use the environment on line 7 of setup.sh

1. Clone this repo
2. In the root of the repo, run `./setup.sh` - this creates the conda environment, installs whats needed, etc.  
3. Enter the environment using `conda activate skagit-met`
4. If you're done, don't forget to `conda deactivate`.

# hrrr_downloader script
To download bulk data, we have a python module/script that can be run.

It:
1. Downloads select parameters from hrrr archives over a specified  using fast herbie for parallel computation
2. Geographically subsets that downloaded data using a provided geojson (geojson polygon boundary - see skagit_boundaries.json for more)
3. Cleans up all data that is not geographically subsetted
4. Saves the data as a zarr store to be read and manipulated - see hrr_model_downloader_notebook.ipynb for example usage

When done this way, each day of data takes only a few MB of disk space. 

To run: 
1. Activate the conda environment in the root of this repo (see setup above)
2. Run from the command line using the following command - be sure to adjust the dates and parameters as needed -  `python hrrr_downloader.py --startDate 2023-02-01 --endDate 2023-02-08 --parameters 'TMP:surface,RH:2 m above ground,WIND:10 m above ground,APCP:surface:0-1 hour acc fcst,DSWRF:surface,DLWRF:surface'`
3. For help with parameters, run `python hrrr_downloader.py -h`

# wrf_downloader script
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
3. For help with parameters, run `python wrf_downloader.py -h`

# prism_downloader script
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
3. For help with parameters, run `python prism_downloader.py -h`

# snotel_downloader script
This script downloads and formats bulk SNOTEL data using metloom. You can read more about metloom [here](https://metloom.readthedocs.io/en/latest/).

Similar to the the other scripts, it:
1. Downloads select parameters from SNOTEL archives over a specified date range using metloom, given Snotel site ids or a geojson to specify the boundaries where you want to get snotel sites within. 
2. Saves the data as a zarr store to be read and manipulated. 

Note that this script doesn't keep the encoded geometries of the snotel sites. If you're interested in getting those, I recommended using a script like this:

```python
import geopandas as gpd

from metloom.pointdata import SnotelPointData
from metloom.variables import SnotelVariables

GEOJSON = 'path/to/your/geojson'

if __name__ == '__main__':
    geometry = gpd.read_file(GEOJSON)
    points = SnotelPointData.points_from_geometry(geometry, [SnotelVariables.PRECIPITATION])
    points.to_dataframe().to_file('path/to/output/snotel_points.json', driver='GeoJSON')

```

to save your points file as a geojson and just using geopnadas for handling that.

To run:
1. Using pixi: `pixi run snotel --frequency daily --startDate 2023-01-01 --endDate 2023-01-08 --outputDir data/weather_data/`
2. For help with parameters, run `pixi run snotel_downloader.py -h`

# Another Environment:
* [Cryocloud](https://book.cryointhecloud.com/content/Getting_Started.html): Built-in environment to access and manipulate data.

# Resources
* https://rapidrefresh.noaa.gov/Diag-vars-NOAA-TechMemo.pdf
* https://rapidrefresh.noaa.gov/hrrr/HRRR/Welcome.cgi?dsKey=hrrr_ncep_jet
* https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
* https://www.nco.ncep.noaa.gov/pmb/docs/on388/table2.html
* https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2.shtml
