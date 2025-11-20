import rioxarray as rxr
import xarray as xr
import os
import requests
import geopandas as gpd
import pandas as pd
from datetime import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile
import pathlib  # Python >= 3.4
import dask as dask
import argparse

BASE_URL = "https://services.nacse.org/prism/data/get"
# Format options, we need
DEFAULT_REGION = "us"
REGION_OPTIONS = ["us", "ak", "hi", "pr"]
DEFAULT_RESOLUTION = "4km"
RESOLUTION_OPTIONS = ["4km", "800m", "400m"]
DEFAULT_FORMAT = "nc"
FORMAT_OPTIONS = ["nc", "bil", "asc", "geotiff"]
DEFAULT_FREQUENCY = "daily"
FREQUENCY_OPTIONS = ["daily", "monthly", "annual"]
DEFAULT_PARAMS = ["tmean", "tmax", "tmin", "ppt", "vpdmax", "vpdmin", "tdmean"]


def setupArgs() -> None:
  parser = argparse.ArgumentParser(
    description="""Download PRISM downsampled data and clip to region and save as zarr. See https://www.prism.oregonstate.edu/ and https://www.prism.oregonstate.edu/documents/PRISM_datasets.pdf
                                     Uses the prism Webservice -- https://prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf"""
  )
  parser.add_argument(
    "--parameters",
    type=str,
    default=DEFAULT_PARAMS,
    choices=DEFAULT_PARAMS,
    nargs="+",
    help="the variables that will be downloaded - these are PRISM output vars and defaults to all",
  )
  parser.add_argument(
    "--startDate",
    type=str,
    required=True,
    help="Start date of data to download e.g. 2020-10-01 for October 1, 2020",
  )
  parser.add_argument(
    "--endDate",
    type=str,
    required=True,
    help="End date of data to download e.g. 2020-10-01 for October 1, 2020",
  )
  parser.add_argument(
    "--outputDir",
    default="data/weather_data/",
    type=str,
    help="Directory/path to download data/output zarr to.",
  )
  parser.add_argument(
    "--geojson",
    default="data/GIS/SkagitBoundary.json",
    type=str,
    help="Path to/name of geo_json file that geogrpahically limits the downloaded data",
  )
  parser.add_argument(
    "--region",
    default=DEFAULT_REGION,
    type=str,
    choices=REGION_OPTIONS,
    help="Region to download data for. Default is US",
  )
  parser.add_argument(
    "--resolution",
    default=DEFAULT_RESOLUTION,
    type=str,
    choices=RESOLUTION_OPTIONS,
    help="Resolution of data to download. Default is 4km. Only 4km and 800m are available.",
  )
  parser.add_argument(
    "--format",
    default=DEFAULT_FORMAT,
    type=str,
    choices=FORMAT_OPTIONS,
    help="Format of data to download. Default is nc",
  )
  parser.add_argument(
    "--frequency",
    default=DEFAULT_FREQUENCY,
    type=str,
    choices=FREQUENCY_OPTIONS,
    help="Frequency of data to download. Default is daily",
  )
  parser.add_argument(
    "--keepZip",
    type=bool,
    default=False,
    help="Keep the zipped files after download. Default is False",
  )

  return parser.parse_args()


def create_prism_dataset(
  min_date: str,
  max_date: str,
  dest_path: str,
  boundaries_gdf: gpd.GeoDataFrame,
  zip_paths: list[str],
  frequency: str,
  resolution: str,
) -> xr.Dataset:
  # Output Zarr
  output_file = "%s/%s_%s_%s_%s_PRISM_data.zarr" % (
    dest_path,
    min_date,
    max_date,
    frequency,
    resolution,
  )

  # Collect Individual Variable Data arrays
  rasters = []
  nc_files = []
  for zip_path in zip_paths:
    with ZipFile(zip_path, "r") as zip_ref:
      nc_file = [f for f in zip_ref.namelist() if f.endswith(".nc")][0]
      zip_ref.extract(nc_file, path=dest_path)
      full_path = os.path.join(dest_path, nc_file)
      variable = os.path.basename(nc_file).split("_")[1]
      date = os.path.basename(nc_file).split("_")[4].split(".")[0]
      nc_files.append({"full_path": full_path, "variable": variable, "date": date})

  for f in nc_files:
    # open weather file and clip to watershed boundaries
    raster = rxr.open_rasterio(f["full_path"], masked=True)
    raster = raster.rio.clip(boundaries_gdf.to_crs(raster.rio.crs).geometry)

    # get date from filename and add as a time coordinate
    # if frequency is daily, date is in YYYYMMDD format, else YYYYMM
    # if frequency is annual, date is in YYYY format
    if frequency == "daily":
      date = dt.strptime(f["date"], "%Y%m%d")
    elif frequency == "monthly":
      date = dt.strptime(f["date"], "%Y%m")
    elif frequency == "annual":
      date = dt.strptime(f["date"], "%Y")
    raster = raster.expand_dims(dim="time")
    raster.coords["time"] = ("time", [date])
    raster = (
      raster.drop_vars(["spatial_ref"])
      .sel(band=1)
      .drop_vars(["band"])
      .rename(f["variable"])
      .rename({"x": "lon", "y": "lat"})
    )
    raster.drop_attrs()

    # add timestamp to list
    rasters.append(raster)

  weather_dataset = xr.merge(rasters)
  weather_dataset.to_zarr(output_file, mode="w")

  return weather_dataset


def parseDateRange(startDateString: str, endDateString: str, frequency: str) -> pd.DatetimeIndex:
  if frequency == "daily":
    start_date = dt.strptime(startDateString, "%Y-%m-%d")
    end_date = dt.strptime(endDateString, "%Y-%m-%d")
    freq = "1D"
    date_range = pd.date_range(start_date, end_date, freq=freq, normalize=True).strftime("%Y%m%d")
  elif frequency == "monthly":
    start_date = dt.strptime(startDateString, "%Y-%m")
    end_date = dt.strptime(endDateString, "%Y-%m")
    freq = "1ME"
    date_range = pd.date_range(start_date, end_date, freq=freq, normalize=True).strftime("%Y%m")
  elif frequency == "annual":
    start_date = dt.strptime(startDateString, "%Y")
    end_date = dt.strptime(endDateString, "%Y")
    freq = "1Y"
    date_range = pd.date_range(start_date, end_date, freq=freq, normalize=True).strftime("%Y")

  if start_date > end_date:
    raise ValueError(f"Start date {start_date} must be before end date {end_date}")

  return date_range


def clean_up_files(files: list) -> None:
  [pathlib.Path(f).unlink(missing_ok=True) for f in files]


if __name__ == "__main__":
  # Get Arguments - model, variables, product, date range, and geo_json
  args = setupArgs()
  parameters = args.parameters
  dates = parseDateRange(args.startDate, args.endDate, args.frequency)
  output_dir = args.outputDir
  if output_dir[-1] == "/":
    output_dir = output_dir[:-1]

  # No Longer using FTP Client
  # '/us/4km/tmin/202204?format=nc'
  url_params = f"/{args.region}/{args.resolution}/"
  query_params = {"format": args.format}

  def download(var, date, output_dir) -> str:
    try:
      zip_file_path = os.path.join(output_dir, f"{var}_{date}_{args.resolution}.zip")
      with requests.get(BASE_URL + url_params + var + "/" + date, params=query_params) as response:
        response.raise_for_status()
        with open(zip_file_path, "wb") as zip_file:
          for chunk in response.iter_content(chunk_size=8192):
            zip_file.write(chunk)
      return zip_file_path
    except Exception:
      print(f"Failed to download for {var}")

  start_time = dt.now()
  futures = []
  zip_paths = []
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
      executor.submit(download, var, date, output_dir) for var in parameters for date in dates
    ]
    zip_paths = [future.result() for future in futures]

  end_time = dt.now()

  print(
    "Time to download {} {}(s): {} seconds".format(
      len(dates), args.frequency, (end_time - start_time).seconds
    )
  )

  # Create Dataset, write out to zarr
  mask = gpd.read_file(args.geojson)
  if args.format != "nc":
    print("Will only merge and write out to zarr if format is nc for now")
  else:
    print("Creating zarr dataset...")
    create_prism_dataset(
      args.startDate, args.endDate, output_dir, mask, zip_paths, args.frequency, args.resolution
    )
    print("Zarr dataset created...")

  # cleanup
  if args.keepZip:
    print("Keeping zipped files...")
  else:
    print("Cleaning up zipped files...")
    clean_up_files(zip_paths)

  print("Cleaning up extracted files...")
  clean_up_files([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".nc")])
