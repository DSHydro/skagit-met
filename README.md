# skagit-met
Exploring meteorology data for use in hydrologic modeling of the Skagit River basin

# Setup
This repo relies on conda environments - I recommend [miniforge](https://github.com/conda-forge/miniforge). I also fast explored/protoyped herbie and other hrrr data in jupyter/ipython notebooks. 

My setup installs jupyter in my base conda environment, then runs notebooks in the sub-envirionment kernels. You can do the same with:

```
conda install jupyterlab
conda install nb_conda_kernels
```

From there run, `jupyterlab` in your base environment, and you should see the `skagit-met` kernel as an option

### Assumptions:
1. I assume you have conda, and all its dependencies installed, and a base conda environment going
2. I assume you are using bash, but if you're using zsh or another shell, this will run better if you modify setup.sh to use the environment on line 7 of setup.sh

1. Clone this repo
2. In the root of the repo, run `./setup.sh` - this creates the conda environment, installs whats needed, etc.  
3. Enter the environment using `conda activate skagit-met`
4. If you're done, don't forget to `conda deactivate`.

### Another Environment:
* [Cryocloud](https://book.cryointhecloud.com/content/Getting_Started.html): Built-in environment to access and manipulate data.

# Resources
* https://rapidrefresh.noaa.gov/Diag-vars-NOAA-TechMemo.pdf
* https://rapidrefresh.noaa.gov/hrrr/HRRR/Welcome.cgi?dsKey=hrrr_ncep_jet
* https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfsfcf00.grib2.shtml
* https://www.nco.ncep.noaa.gov/pmb/docs/on388/table2.html
* https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2.shtml
