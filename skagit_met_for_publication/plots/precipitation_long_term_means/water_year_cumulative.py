"""Total precipitation per water year for six products and four basin areas (CSV).

For every water year from 1983 to 2020 (WY N = 1 Oct of year N-1 to 30 Sep of
year N), daily basin-mean precipitation is extracted from six gridded
products (PRISM, PNNL, Daymet, CONUS404, UCLA, GridMET) and summed over the
water year. This is done separately for the Upper Skagit, Sauk and Lower
Skagit HUC8 sub-basins and for the three combined ("All"). The result is one
row per water year with one column per <area>_<PRODUCT> pair.

This CSV is the input of plot_water_year_cumulative_data.py and
plot_water_year_cumulative_panel.py.

Main functions:
  extract_water_year_data: Extract daily precipitation for a specific water year
  create_water_year_cumulative_csv: Generate and save CSV with cumulative precipitation

Input
    /data0/skagit_met/data_transfer/data/...      (gridded precipitation products)
    data/GIS/SkagitSubBasin_HUC8.geojson
    data/weather_data/conus404_skagit_precip_daily_full.zarr
Output
    plots/precipitation_long_term_means/water_year_cumulative_precipitation.csv

Usage
    python water_year_cumulative.py      (slow: reads every day of 38 water years, 4 times)
"""

import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import rioxarray
import xarray as xr

# Shared raw-data archive (outside this repository)
VAULT_DIR = "/data0/skagit_met/data_transfer/data"

# Paths are resolved relative to this file: <repo>/skagit_met_for_publication/
BASE_DIR = Path(__file__).resolve().parents[2]
HUC8_GEO = BASE_DIR / "data" / "GIS" / "SkagitSubBasin_HUC8.geojson"
OUTPUT_DIR = Path(__file__).resolve().parent

# Precipitation product locations
PRISM_DIR = os.path.join(VAULT_DIR, "prism_new_hq")  # daily TIFF zips, by decade folder
DAYMET_DIR = os.path.join(VAULT_DIR, "daymet_new_hq")  # prcp_<year>_subset.nc
PNNL_DIR = os.path.join(VAULT_DIR, "PNNL/historical")  # <year>/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.<year>.nc
PNNL_GEO = os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc")  # PNNL grid coordinates
CONUS_ZARR = BASE_DIR / "data" / "weather_data" / "conus404_skagit_precip_daily_full.zarr"
UCLA_PREC_DIR = os.path.join(VAULT_DIR, "ucla_era5_d02_daily", "prec")  # prec.daily.era5.d02.<year>.nc
UCLA_STATIC = os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc")  # UCLA grid coordinates
GRIDMET_DIR = os.path.join(VAULT_DIR, "gridmet")  # <year>_daily_4km_gridMET_data.zarr
GRIDMET_SAMPLE = os.path.join(GRIDMET_DIR, "2021_daily_4km_gridMET_data.zarr")  # used only for its grid

PRODUCTS = ['prism', 'pnnl', 'daymet', 'conus', 'ucla', 'gridmet']
PNNL_LAST_YEAR = 2020  # PNNL historical run ends in 2020

# Sub-basins (HUC8 names) in mask order, and the areas written to the CSV ('All' = the three combined)
SUBBASINS = ["Upper Skagit", "Sauk", "Lower Skagit"]
GEOMETRIES = SUBBASINS + ['All']

OUTPUT_CSV_NAME = 'water_year_cumulative_precipitation.csv'

# Water years to process (water year is Oct 1 - Sep 30)
# Water year 1983 = Oct 1, 1982 to Sep 30, 1983
WATER_YEAR_START = 1983
WATER_YEAR_END = 2020


def load_prism_from_new_format(date, prism_root):
    """Load PRISM data from new format (TIFF) zip file for a specific date"""
    date_str = date.strftime('%Y%m%d')
    year = date.year
    decade_folder = get_decade_folder(year)
    zip_path = os.path.join(prism_root, decade_folder, f"prism_ppt_us_25m_{date_str}.zip")

    if not os.path.exists(zip_path):
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith('.tif'):
                            filepath = os.path.join(root, f)
                            da = rioxarray.open_rasterio(filepath)
                            return da.squeeze().copy(deep=True)
    except Exception:
        pass

    return None


def load_daymet_from_netcdf(year):
    """Load Daymet data from netCDF file for a specific year"""
    daymet_nc_path = os.path.join(DAYMET_DIR, f"prcp_{year}_subset.nc")

    if not os.path.exists(daymet_nc_path):
        return None

    try:
        ds = xr.open_dataset(daymet_nc_path)
        return ds
    except Exception as e:
        print(f"  Error loading Daymet netCDF for {year}: {e}")
        return None


def load_regions():
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    return gdf


def get_mask(gdf, lon, lat):
    """Get the mask of the three sub-basins, with one entry per sub-basin along 'region'."""
    mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    for r in SUBBASINS:
        try:
            idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
            region_indices.append(idx)
        except Exception:
            pass
    return mask.sel(region=region_indices)


def calculate_basin_mean(da, mask_2d, region_idx=None):
    """Calculate basin mean, optionally for a specific region.
    If region_idx is an integer, extract that region from the mask."""
    data = da.values
    mask = mask_2d.values if hasattr(mask_2d, 'values') else mask_2d

    # Handle multi-region mask
    if mask.ndim == 3 and region_idx is not None:
        mask = mask[region_idx]
    elif mask.ndim == 3:
        mask = (mask > 0).any(axis=0).astype(float)

    if data.ndim == 3:
        time_steps = data.shape[0]
        data_flat = data.reshape(time_steps, -1)
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            mean_vals = np.nanmean(data_flat[:, mask_idx], axis=1)
        else:
            mean_vals = np.full(time_steps, np.nan)
        time_coord = da.time.values if 'time' in da.coords else range(time_steps)
        return pd.Series(mean_vals, index=pd.to_datetime(time_coord))
    elif data.ndim == 2:
        data = data.astype(float)
        data[data < 0] = np.nan

        data_flat = data.flatten()
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            mean_val = np.nanmean(data_flat[mask_idx])
        else:
            mean_val = np.nan
        return mean_val
    else:
        raise ValueError(f"Unexpected data dimensions: {data.ndim}")


def get_decade_folder(year: int) -> str:
    if 2021 <= year <= 2024:
        return "2021-2024"
    start_decade = (year // 10) * 10
    if start_decade == 1980 and year > 1980:
        return "1981-1990"
    if year % 10 == 0 and year > 1980:
        return f"{start_decade-9}-{start_decade}"
    if start_decade % 10 == 0:
        return f"{start_decade+1}-{start_decade+10}"
    return f"{start_decade}-{(year//10)*10+9}"


def get_water_year_dates(water_year):
    """
    Get the date range for a water year.
    Water year starts Oct 1 of previous calendar year and ends Sep 30 of given year.
    E.g., water year 2010 = Oct 1, 2009 to Sep 30, 2010
    """
    start_date = pd.Timestamp(year=water_year-1, month=10, day=1)
    end_date = pd.Timestamp(year=water_year, month=9, day=30)
    return pd.date_range(start_date, end_date, freq='D').tolist()


def extract_water_year_data(water_year, products_to_extract=PRODUCTS, geometry=None):
    """Extract daily basin-mean precipitation for a complete water year across all products.

    ``geometry`` is one of SUBBASINS, or None/'All' for the three combined.
    Returns a DataFrame indexed by date string with one column per product.
    """

    window_dates = get_water_year_dates(water_year)
    results = {f'{date.strftime("%Y-%m-%d")}': {} for date in window_dates}

    gdf = load_regions()
    masks_2d = {}

    # Which sub-basin to use (None = all combined, 0 = Upper Skagit, 1 = Sauk, 2 = Lower Skagit)
    region_idx = None
    if geometry and geometry.lower() != 'all':
        region_map = {name.lower(): i for i, name in enumerate(SUBBASINS)}
        region_idx = region_map.get(geometry.lower())

    # Basin masks on each product's grid - keep the region dimension if extracting one sub-basin.
    # PRISM and Daymet masks are built later from the coordinates of the data actually loaded.
    try:
        sample_geo = xr.open_dataset(PNNL_GEO)
        mask = get_mask(gdf, sample_geo.XLONG_M.values[0], sample_geo.XLAT_M.values[0])
        masks_2d['PNNL'] = mask if region_idx is not None else mask.any(dim='region')
    except Exception as e:
        print(f"  Warning: Could not load PNNL mask: {e}")

    try:
        sample_conus = xr.open_zarr(CONUS_ZARR)
        mask = get_mask(gdf, sample_conus.lon, sample_conus.lat)
        masks_2d['CONUS404'] = mask if region_idx is not None else mask.any(dim='region')
        sample_conus.close()
    except Exception:
        pass

    try:
        ucla_static = xr.open_dataset(UCLA_STATIC)
        mask = get_mask(gdf, ucla_static.lon2d.values, ucla_static.lat2d.values)
        masks_2d['UCLA'] = mask if region_idx is not None else mask.any(dim='region')
    except Exception as e:
        print(f"  Warning: Could not load UCLA mask: {e}")

    try:
        sample_gridmet = xr.open_zarr(GRIDMET_SAMPLE)
        mask = get_mask(gdf, sample_gridmet.lon, sample_gridmet.lat)
        masks_2d['GridMET'] = mask if region_idx is not None else mask.any(dim='region')
    except Exception:
        pass

    # Extract PRISM from new format (TIFF)
    if 'prism' in products_to_extract:
        prism_data = {}

        for date in window_dates:
            try:
                da = load_prism_from_new_format(date, PRISM_DIR)
                if da is None:
                    continue
                lon = da.x.values
                lat = da.y.values
                m_prism = get_mask(gdf, lon, lat)
                mean_val = calculate_basin_mean(da, m_prism, region_idx=region_idx)
                prism_data[date.normalize()] = mean_val
            except Exception:
                pass

        for date in window_dates:
            normalized_date = date.normalize()
            if normalized_date in prism_data:
                results[f'{date.strftime("%Y-%m-%d")}']['prism'] = float(prism_data[normalized_date])

        prism_count = sum(1 for d in prism_data.values() if not np.isnan(d))
        if prism_count > 0:
            print(f"    PRISM: {prism_count}/{len(window_dates)} dates extracted")

    # Extract PNNL (hourly, summed to daily; a water year spans two calendar-year files)
    if 'pnnl' in products_to_extract and water_year <= PNNL_LAST_YEAR:
        try:
            for cal_year in [water_year - 1, water_year]:
                pnnl_path = os.path.join(PNNL_DIR, str(cal_year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{cal_year}.nc")
                if os.path.exists(pnnl_path):
                    try:
                        ds = xr.open_dataset(pnnl_path)
                        hourly_dates = []
                        for d in window_dates:
                            hourly_dates.extend(pd.date_range(d, d + pd.Timedelta(hours=23), freq='h'))
                        available_hourly = [d for d in hourly_dates if d in ds.time.values]
                        if available_hourly and 'PNNL' in masks_2d:
                            da_subset = ds['PREC_ACC_NC'].sel(time=available_hourly)
                            s_hourly = calculate_basin_mean(da_subset, masks_2d['PNNL'], region_idx=region_idx)
                            pnnl_daily = s_hourly.resample('1D').sum()
                            pnnl_daily.index = pnnl_daily.index.normalize()
                            for date in window_dates:
                                normalized_date = date.normalize()
                                if normalized_date in pnnl_daily.index:
                                    val = pnnl_daily[normalized_date]
                                    results[f'{date.strftime("%Y-%m-%d")}']['pnnl'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                        ds.close()
                    except Exception as e:
                        print(f"  PNNL Error for {cal_year}: {e}")
            pnnl_count = sum(1 for d in window_dates if f'{d.strftime("%Y-%m-%d")}' in results and 'pnnl' in results[f'{d.strftime("%Y-%m-%d")}'])
            if pnnl_count > 0:
                print(f"    PNNL: {pnnl_count}/{len(window_dates)} dates extracted")
        except Exception as e:
            print(f"  PNNL Error for {water_year}: {e}")

    # Extract Daymet (new format - netCDF files)
    if 'daymet' in products_to_extract:
        try:
            # For water year, we need data from two calendar years
            for cal_year in [water_year - 1, water_year]:
                ds = load_daymet_from_netcdf(cal_year)
                if ds is not None:
                    ds['time'] = pd.to_datetime(ds.time.values).normalize()
                    da = ds['prcp']

                    if 'Daymet' not in masks_2d:
                        lon_2d = ds['lon'].values
                        lat_2d = ds['lat'].values
                        mask = get_mask(gdf, lon_2d, lat_2d)
                        masks_2d['Daymet'] = mask if region_idx is not None else mask.any(dim='region')

                    year_window = [d for d in window_dates if d.year == cal_year]
                    if year_window:
                        da_window = da.sel(time=year_window, method='nearest')
                        daymet_daily = calculate_basin_mean(da_window, masks_2d['Daymet'], region_idx=region_idx)
                        daymet_daily.index = daymet_daily.index.normalize()
                        daymet_daily = daymet_daily[~daymet_daily.index.duplicated(keep='first')]
                        for date in year_window:
                            normalized_date = date.normalize()
                            if normalized_date in daymet_daily.index:
                                val = daymet_daily[normalized_date]
                                results[f'{date.strftime("%Y-%m-%d")}']['daymet'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                    ds.close()
        except Exception:
            pass

    # Extract CONUS404
    if 'conus' in products_to_extract:
        try:
            if os.path.exists(CONUS_ZARR):
                ds = xr.open_zarr(CONUS_ZARR)
                ds['time'] = pd.to_datetime(ds.time.values).normalize()
                da = ds['precip_daily'].sel(time=window_dates, method='nearest')
                if 'CONUS404' in masks_2d:
                    conus_daily = calculate_basin_mean(da, masks_2d['CONUS404'], region_idx=region_idx)
                    conus_daily.index = conus_daily.index.normalize()
                    conus_daily = conus_daily[~conus_daily.index.duplicated(keep='first')]
                    for date in window_dates:
                        normalized_date = date.normalize()
                        if normalized_date in conus_daily.index:
                            val = conus_daily[normalized_date]
                            results[f'{date.strftime("%Y-%m-%d")}']['conus'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ds.close()
        except Exception:
            pass

    # Extract UCLA
    if 'ucla' in products_to_extract:
        try:
            series_list = []
            for cal_year in [water_year - 1, water_year]:
                u_path = os.path.join(UCLA_PREC_DIR, f"prec.daily.era5.d02.{cal_year}.nc")
                if os.path.exists(u_path):
                    try:
                        ds = xr.open_dataset(u_path)
                        u_var = "prec" if "prec" in ds.data_vars else "pr"
                        if 'day' in ds.dims:
                            ds = ds.rename({'day': 'time'})
                        ds['time'] = pd.to_datetime(ds.time.values).normalize()

                        available_dates = [d for d in window_dates if d in ds.time.values]
                        if available_dates and 'UCLA' in masks_2d:
                            da_subset = ds[u_var].sel(time=available_dates)
                            s = calculate_basin_mean(da_subset, masks_2d['UCLA'], region_idx=region_idx)
                            s.index = s.index.normalize()
                            series_list.append(s)
                        ds.close()
                    except Exception as e:
                        print(f"  UCLA Error for {cal_year}: {e}")
            if series_list:
                ucla_daily = pd.concat(series_list).sort_index()
                ucla_daily = ucla_daily[~ucla_daily.index.duplicated(keep='first')]
                for date in window_dates:
                    normalized_date = date.normalize()
                    if normalized_date in ucla_daily.index:
                        val = ucla_daily[normalized_date]
                        results[f'{date.strftime("%Y-%m-%d")}']['ucla'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                ucla_count = sum(1 for d in window_dates if f'{d.strftime("%Y-%m-%d")}' in results and 'ucla' in results[f'{d.strftime("%Y-%m-%d")}'])
                if ucla_count > 0:
                    print(f"    UCLA: {ucla_count}/{len(window_dates)} dates extracted")
        except Exception as e:
            print(f"  UCLA Error for {water_year}: {e}")

    # Extract GridMET
    if 'gridmet' in products_to_extract:
        try:
            for cal_year in [water_year - 1, water_year]:
                gridmet_path = os.path.join(GRIDMET_DIR, f"{cal_year}_daily_4km_gridMET_data.zarr")
                if os.path.exists(gridmet_path):
                    ds = xr.open_zarr(gridmet_path)
                    if 'day' in ds.dims:
                        ds = ds.rename({'day': 'time'})
                    ds['time'] = pd.to_datetime(ds.time.values).normalize()
                    var = 'prcp' if 'prcp' in ds.data_vars else 'precipitation_amount'

                    year_window = [d for d in window_dates if d.year == cal_year]
                    if year_window:
                        da = ds[var].sel(time=year_window, method='nearest')
                        if 'GridMET' in masks_2d:
                            gridmet_daily = calculate_basin_mean(da, masks_2d['GridMET'], region_idx=region_idx)
                            gridmet_daily.index = gridmet_daily.index.normalize()
                            gridmet_daily = gridmet_daily[~gridmet_daily.index.duplicated(keep='first')]
                            for date in year_window:
                                normalized_date = date.normalize()
                                if normalized_date in gridmet_daily.index:
                                    val = gridmet_daily[normalized_date]
                                    results[f'{date.strftime("%Y-%m-%d")}']['gridmet'] = float(val) if not isinstance(val, pd.Series) else float(val.iloc[0])
                    ds.close()
        except Exception:
            pass

    return pd.DataFrame(results).T


def create_water_year_cumulative_csv():
    """Extract and save cumulative precipitation data to CSV for each geometry."""
    print(f"Extracting water year data from {WATER_YEAR_START} to {WATER_YEAR_END}...")

    products = PRODUCTS
    water_years = range(WATER_YEAR_START, WATER_YEAR_END + 1)

    # Water-year totals, {water_year: {"<area>_<PRODUCT>": total}}
    csv_data = {wy: {} for wy in water_years}

    for geometry in GEOMETRIES:
        for wy in water_years:
            print(f"  Extracting water year {wy} for {geometry}...")
            wy_data = extract_water_year_data(wy, products_to_extract=products, geometry=geometry)

            # Total over the water year for each product
            for product in products:
                if product in wy_data.columns:
                    total = wy_data[product].sum()
                    if not np.isnan(total):
                        csv_data[wy][f"{geometry}_{product.upper()}"] = total

    # Export to CSV
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_df = pd.DataFrame(csv_data).T
    csv_df.index.name = 'WaterYear'
    csv_df = csv_df.sort_index(axis=1)
    csv_df.to_csv(csv_path)
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    create_water_year_cumulative_csv()
