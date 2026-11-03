# Multi-Product Bulk Bias Analysis - Skagit Basin

## Overview
This directory contains a pipeline for analyzing precipitation bias across multiple weather/climate datasets within the Skagit Basin: PNNL, Daymet, CONUS404, UCLA and GridMET. The analysis integrates USGS hydrological data with atmospheric river (AR) event classifications to evaluate how different precipitation products perform during various atmospheric conditions.

## Directory Structure
```
multi_product_bulk_bias/
├── 1_generate_daily_dataset.py       
├── 2_ar_scale_and_sampling.py
├── 3_bulk_bias_new_version.py
├── 4_data_preparation.py
├── 5_preprocess_products.py
├── 6_plot_prism_bias_old.py
├── 7_corrected_plot_prism_bias.py
├── 8_corrected_plot_prism_bias_percent.py
├── outputs/                          # Generated data files
│   ├── 1_skagit_daily_integrated_dataset_1980_2025.csv
│   ├── 2_ar_scale_and_sampling.csv
│   ├── 3_multi_product_bulk_bias_data.csv
│   ├── 4_clean_bias_table.csv
│   └── 5_prepared_bias_table.csv
├── plots/                            # Generated visualizations
│   ├── grouped_bias_3_metrics_2_periods.png
│   └── grouped_bias_percent_3_metrics_2_periods.png
├── old_code/                         # Previous versions
└── downloading_missing_data/         # Data acquisition scripts
```

---

## Workflow Pipeline

### 1. **Generate Daily Dataset** (`1_generate_daily_dataset.py`)
Create the a CSV file combining discharge meassurements from a USGS station and AR event data.

#### What it does:
- Loads USGS discharge (Q) and gage height (H) measurements from station 12200500
- Loads atmospheric river event classifications from three monitoring stations:
  - 47.5°N, 124.5°W
  - 48.0°N, 124.5°W
  - 48.5°N, 124.5°W
- Merges all data sources on date, filling AR scale values for days within event duration windows
- Outputs a comprehensive daily dataset suitable for downstream bias analysis

#### Input Data:
- **USGS Discharge**: `usgs_12200500_discharge.rdb` (RDB format)
- **USGS Gage Height**: `usgs_12200500_gage_height.rdb` (RDB format)
- **AR Event Data**: `ar_event_data_47*_124*.csv` (three station files)
  - Each file contains: start/max/end dates, duration, max IVT, AR scale, IWV values, wind components

#### Output:
- `1_skagit_daily_integrated_dataset_1980_2025.csv`
  - Columns: `date`, `discharge_cfs`, `gage_height_ft`, `ar_scale_47.5N_124.5W`, `ar_scale_48.0N_124.5W`, `ar_scale_48.5N_124.5W`
  - ~16,700 daily records
  - Discharge available for most records; AR scales 1-5 on event days, 0 otherwise

---

### 2. **AR Scale & Sampling** (`2_ar_scale_and_sampling.py`)
Aggregates AR scales across stations (picks the maxumum scale across the 3 stations for that date) and creates balanced sampling subset (all AR events + 2,000 non-AR days).

**Output**: `2_ar_scale_and_sampling.csv`

---

### 3. **Bulk Bias Analysis** (`3_bulk_bias_new_version.py`)
Computes bias metrics (MAE, correlation, percent bias) for multiple precipitation products.

**Output**: `3_multi_product_bulk_bias_data.csv`

---

### 4-5. **Data Preparation & Preprocessing** (`4_data_preparation.py`, `5_preprocess_products.py`)
Cleans bias table and prepares data for visualization.

**Outputs**: 
- `4_clean_bias_table.csv`
- `5_prepared_bias_table.csv`

---

### 6-8. **Visualization & Analysis** (`6_plot_prism_bias_old.py`, `7_corrected_plot_prism_bias.py`, `8_corrected_plot_prism_bias_percent.py`)
Generates diagnostic plots showing:
- Bias metrics across precipitation products
- Comparison between AR and non-AR periods
- Percentage bias visualizations

**Outputs**: 
- `grouped_bias_3_metrics_2_periods.png`
- `grouped_bias_percent_3_metrics_2_periods.png`

---

## Output Locations
- **Data files**: `outputs/`
- **Plots**: `plots/`

---

*Last updated: September 2025*
