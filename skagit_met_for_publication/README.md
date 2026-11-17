# Skagit Basin precipitation products: bias during atmospheric river and non-AR events

> **SKETCH: not final.** Anything marked **TODO** is missing information or work still to do.

Code and data used for:

> **TODO:** Authors (Year). *Title*. Journal. DOI.

> **TODO:** One paragraph on the study: what is compared (six gridded precipitation
> products against PRISM, plus SNOTEL), over what region (Skagit Basin, WA: Upper
> Skagit, Sauk and Lower Skagit HUC8 sub-basins) and period (water years 1982–2024),
> and the main question (how product bias differs between AR and non-AR events,
> and with streamflow and precipitation intensity).

## Repository layout

```
data/          Small inputs shipped with the repo (AR catalogs, USGS gage, GIS)
downloaders/   Scripts that download the large gridded products (not shipped)
analysis/      Numbered pipeline that builds the event tables (1 → 5)
plots/         One folder per figure group; each script writes its PNG next to it
```

## Requirements

> **TODO:** Environment file (`environment.yml` or `pixi.toml`) with pinned versions.
> Main packages: pandas, numpy, xarray, geopandas, rioxarray, regionmask, cartopy,
> matplotlib, seaborn, scipy, joblib, metloom, zeep, requests.

## Data

### Included in this repository (`data/`)

| File | Content | Source |
|---|---|---|
| `ar_event_data_475_1245.csv`, `_480_`, `_485_` | AR event catalogs at 47.5°N, 48.0°N, 48.5°N (124.5°W) | **TODO:** catalog name, reference, download date, column description (the CSVs have no header) |
| `usgs_12200500_discharge.rdb` | Daily mean discharge, Skagit R. nr Mount Vernon | USGS NWIS, parameter 00060. **TODO:** download date |
| `usgs_12200500_gage_height.rdb` | Daily gage height | USGS NWIS, parameter 00065 |
| `GIS/SkagitSubBasin_HUC8.geojson` | HUC8 sub-basins used for basin means | **TODO:** source (USGS WBD?) |
| `GIS/SkagitBoundary.json` | Basin outline | **TODO:** source |
| `GIS/SkagitRiver_90mDEM.tif`, `skagit_elevation_3520ft.*` | DEM and elevation contour | **TODO:** source, and which script uses them |
| `GIS/skagit_snotel_points.json` | SNOTEL stations in the basin | NRCS SNOTEL |

### Not included: download with `downloaders/`

| Product | Downloader | Access | Size |
|---|---|---|---|
| PRISM AN81d daily 4 km (reference) | `download_prism_data.py` | public | **TODO** |
| Daymet V4R1 daily | `download_daymet_data.py` | NASA Earthdata login (`~/.netrc`) | ~4.5 GB/yr before subsetting |
| CONUS404 daily | `conus404_daily_downloader.py` | public (USGS OSN) | ~89 GB |
| CONUS404 hourly (event windows) | `conus_hourly_download.py` | gdex.ucar.edu. **TODO:** account needed? Docstring is a placeholder |
| UCLA WRF-ERA5 d02 daily (+ `wrfinput_d02_coord.nc`) | `ucla_era5_daily_download.py` | public (AWS S3) | ~130 MB/file |
| UCLA WRF-ERA5 d02 hourly (event windows) | `ucla_hourly_download.py` | public (AWS S3) | **TODO** |
| gridMET daily | `gridmet_downloader.py` | public | ~60 MB/yr |
| PNNL WRF historical (hourly) | **TODO: no downloader.** Where can readers get it? | | |
| SNOTEL daily/hourly | `snotel_downloader.py` (called on demand by plot scripts) | public | small. **TODO:** add docstring |


## Configuration

> **TODO:** All scripts currently hardcode `VAULT_DIR = /data0/skagit_met/data_transfer/data`
> (and a few other absolute paths). Replace these with one setting, e.g. an environment variable:
>
> ```bash
> export SKAGIT_DATA_DIR=/path/to/downloaded/products
> ```

## Reproducing the analysis

### 1. Event tables (`analysis/`)

| Step | Script | Output (`analysis/outputs/`) |
|---|---|---|
| 1 | `1_generate_daily_dataset.py` | daily USGS + AR scale at 3 points, 1980–2025 |
| 2 | `2_ar_scale_and_sampling.py` | combined AR scale; non-AR days within ±`WINDOW_DAYS` of an AR day removed |
| 3 | `3_bulk_bias_new_version.py` | basin-mean 3- and 7-day precipitation of each product per event (slow; needs the downloaded products) |
| 3.5 | `3.5_count_prior_events.py` | + events in the prior 7 days |
| 3.75 | `3.75_count_prior_events.py` | + prior 7/30-day counts and days to the prior event |
| 4 | `4_data_preparation.py` | trimmed bias table |
| 5 | `5_preprocess_products.py` | + bias columns and streamflow/precipitation classes |

The figures use two event samples, stored in separate subfolders of `analysis/outputs/`:

| Folder | `WINDOW_DAYS` | Used by |
|---|---|---|
| `5day_taking_min_and_filling/` | 5 (isolated non-AR events) | most figures |
| `0day_taking_min_and_filling/` | 0 (all non-AR events) | compounding events; row (a) of the AR vs non-AR panel |
| `5day_taking_min/` | 5, **TODO:** how does it differ from `..._and_filling`? | `plot_specific_non_ar_events.py` only |

> **TODO:**
> - Explain "taking_min" (minimum AR scale across the 3 points) and "filling" in plain words.
> - Today the scripts write to the top level of `outputs/`, and the run was copied into
>   the subfolder by hand. Document that step, or have the scripts write straight to the folder.

### 2. Figures (`plots/`)

> **TODO:** Fill in the paper figure numbers, and drop any row that is not in the paper.

| Paper fig. | Script | Output |
|---|---|---|
| **TODO** | `precipitation_long_term_means/water_year_cumulative.py` → `plot_water_year_cumulative_panel.py` | `water_year_cumulative_precipitation_panel.png` |
| **TODO** | `precipitation_long_term_means/plot_water_year_cumulative_data.py` | `water_year_cumulative_precipitation_<area>.png` |
| **TODO** | `precipitation_long_term_means/bias_table/bias_table.py` | `precipitation_bias_table.png/.csv` |
| **TODO** | `spatial_bias_distribution/plot_spatial_seasonal_avg.py` | `spatial_seasonal_{products,biases}.png` |
| **TODO** | `spatial_bias_distribution/plot_spatial_percentiles.py` | `spatial_percentiles_{products,biases}.png` |
| **TODO** | `spatial_bias_distribution/plot_{ar,non_ar}_events_cumulative_spatial_bias_grid_8_days_updated.py` | `{ar,non_ar}_events_cumulative_spatial_bias_grid_8_days.png` |
| **TODO** | `ar_event_scale_and_distribution/plot_ar_scale_and_distribution.py` | `ar_frequency_by_water_year_month.png`, `ar_scale_by_streamflow_bucket.png` |
| **TODO** | `top_50_ar_vs_non-AR/ar_vs_non_ar_panel_figure.py` | `ar_vs_non_ar_panel_prism.png` |
| **TODO** | `bias_boxplots/7_corrected_plot_prism_bias.py` | `grouped_bias_3_metrics_2_periods.png` |
| **TODO** | `bias_boxplots/9_prism_bias_ar_vs_nonar.py` | `9_prism_bias_ar_vs_nonar.png` |
| **TODO** | `bias_boxplots/9.5_prism_bias_ar_vs_nonar_by_prior_events.py` | `prior_events/*.png` |
| **TODO** | `plot_specific_ar_and_non_ar_events/plot_specific_{ar_events,non_ar_events,months}.py` | `specific_*_cumulative_precipitation.png` |
| **TODO** | `storm_patterns/create_3ar_3nonar_events_grid_plot.py` | `ar_nonar_events_hourly_grid.png` |
| **TODO** | `storm_patterns/create_ar_events_grid_plot.py`, `create_ar_events_plots.py` | `ar_events_hourly_grid.png`, `ar_event_NN_*.png` |
| **TODO** | `compounding_events/boxplots_compounding.py` / `_daily.py` | `*boxplot_compounding_scenarios_streamflow.png` |

Slow steps cache intermediate results (`*.pkl`, `panel_data_row_*.csv`); delete them to recompute.

> **TODO:** A `run_all.sh` that runs the steps above in order.

## License

> **TODO:** Code license (e.g. MIT) and data license/attribution (e.g. CC-BY 4.0 for derived
> tables; the original product licenses apply to the downloaded data).

## Citation

> **TODO:** `CITATION.cff` + Zenodo DOI for the archived release.

## Contact

> **TODO:** Name, affiliation, email or GitHub issues.
