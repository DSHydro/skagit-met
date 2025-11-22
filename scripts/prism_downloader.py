import argparse
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import requests
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import datadir as pyproj_datadir


DEFAULT_PROJ = os.environ.get("PROJ_LIB") or os.path.join(sys.prefix, "share", "proj")
DEFAULT_GDAL = os.environ.get("GDAL_DATA") or os.path.join(sys.prefix, "share", "gdal")
os.environ.setdefault("PROJ_LIB", DEFAULT_PROJ)
os.environ.setdefault("GDAL_DATA", DEFAULT_GDAL)
pyproj_datadir.set_data_dir(os.environ["PROJ_LIB"])
os.environ.setdefault("PROJ_NETWORK", "ON")

gpd.options.io_engine = "fiona"
os.environ["GEOPANDAS_USE_PYOGRIO"] = "0"

BASE_URL = "https://services.nacse.org/prism/data/get"
DEFAULT_REGION = "us"
REGION_OPTIONS = ["us", "ak", "hi", "pr"]
DEFAULT_RESOLUTION = "4km"
RESOLUTION_OPTIONS = ["4km", "800m", "400m"]
DEFAULT_FORMAT = "nc"
FORMAT_OPTIONS = ["nc", "bil", "asc", "geotiff"]
DEFAULT_FREQUENCY = "daily"
FREQUENCY_OPTIONS = ["daily", "monthly", "annual"]
DEFAULT_PARAMS = ["tmean", "tmax", "tmin", "ppt", "vpdmax", "vpdmin", "tdmean"]


def configure_spatial_env(proj_dir: str | None, gdal_dir: str | None) -> None:
  if proj_dir:
    os.environ["PROJ_LIB"] = proj_dir
    try:
      pyproj_datadir.set_data_dir(proj_dir)
    except Exception:
      pass
  if gdal_dir:
    os.environ["GDAL_DATA"] = gdal_dir
  if proj_dir or gdal_dir:
    os.environ.setdefault("PROJ_NETWORK", "ON")


def setupArgs() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Download PRISM downsampled data, clip to region, and save as Zarr. "
      "See https://www.prism.oregonstate.edu/ for dataset details."
    )
  )
  parser.add_argument(
    "--parameters",
    type=str,
    default=DEFAULT_PARAMS,
    choices=DEFAULT_PARAMS,
    nargs="+",
    help="PRISM variables to download (default: all).",
  )
  parser.add_argument("--startDate", type=str, required=True, help="Start date e.g. 2020-10-01")
  parser.add_argument("--endDate", type=str, required=True, help="End date e.g. 2020-10-02")
  parser.add_argument(
    "--outputDir",
    default="data/weather_data/",
    type=str,
    help="Directory/path to download data/output Zarr to.",
  )
  parser.add_argument(
    "--geojson",
    default="data/GIS/SkagitBoundary.json",
    type=str,
    help="GeoJSON file limiting the downloaded data.",
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
    help="Download file format (default: nc).",
  )
  parser.add_argument(
    "--frequency",
    default=DEFAULT_FREQUENCY,
    type=str,
    choices=FREQUENCY_OPTIONS,
    help="Temporal frequency (daily/monthly/annual).",
  )
  parser.add_argument(
    "--keepZip",
    type=bool,
    default=False,
    help="Keep downloaded ZIP files after processing.",
  )
  parser.add_argument(
    "--projLib",
    type=str,
    required=False,
    default=DEFAULT_PROJ,
    help="Path to the PROJ data directory.",
  )
  parser.add_argument(
    "--gdalData",
    type=str,
    required=False,
    default=DEFAULT_GDAL,
    help="Path to the GDAL data directory.",
  )
  return parser.parse_args()


def create_prism_dataset(
  min_date: str,
  max_date: str,
  dest_path: str,
  boundaries_gdf: gpd.GeoDataFrame,
  data_paths: list[str],
  frequency: str,
  resolution: str,
) -> xr.Dataset:
  output_file = f"{dest_path}/{min_date}_{max_date}_{frequency}_{resolution}_PRISM_data.zarr"
  rasters = []
  nc_files = []

  for path in filter(None, data_paths):
    if path.endswith(".zip"):
      with ZipFile(path, "r") as archive:
        nc_members = [member for member in archive.namelist() if member.endswith(".nc")]
        if not nc_members:
          continue
        member = nc_members[0]
        archive.extract(member, path=dest_path)
        nc_path = os.path.join(dest_path, member)
    else:
      nc_path = path

    fname = os.path.basename(nc_path)
    parts = fname.split("_")
    try:
      variable = parts[1]
      date_part = parts[4].split(".")[0] if len(parts) > 4 else parts[-1].split(".")[0]
    except Exception:
      variable = "var"
      date_part = "".join(c for c in fname if c.isdigit())[:8]

    nc_files.append({"full_path": nc_path, "variable": variable, "date": date_part})

  for entry in nc_files:
    dataset = xr.open_dataset(entry["full_path"], engine="netcdf4")
    dataset = dataset.drop_vars(["crs", "spatial_ref"], errors="ignore")

    var_name = (
      entry["variable"] if entry["variable"] in dataset.data_vars else list(dataset.data_vars)[0]
    )
    data_array = dataset[var_name]

    if frequency == "daily":
      date = dt.strptime(entry["date"], "%Y%m%d")
    elif frequency == "monthly":
      date = dt.strptime(entry["date"], "%Y%m")
    else:
      date = dt.strptime(entry["date"], "%Y")

    data_array = data_array.expand_dims(dim="time").assign_coords(time=("time", [date]))

    rename_map = {}
    if "x" in data_array.dims:
      rename_map["x"] = "lon"
    if "y" in data_array.dims:
      rename_map["y"] = "lat"
    if "longitude" in data_array.dims:
      rename_map["longitude"] = "lon"
    if "latitude" in data_array.dims:
      rename_map["latitude"] = "lat"
    if rename_map:
      data_array = data_array.rename(rename_map)

    data_array = data_array.rio.write_crs("EPSG:4326", inplace=False)
    data_array = data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    data_array = data_array.rio.clip(boundaries_gdf.to_crs("EPSG:4326").geometry)

    rasters.append(data_array.rename(entry["variable"]))

  weather_dataset = xr.merge(rasters)
  weather_dataset.to_zarr(output_file, mode="w")
  return weather_dataset


def parseDateRange(startDateString: str, endDateString: str, frequency: str) -> pd.DatetimeIndex:
  if frequency == "daily":
    start_date = dt.strptime(startDateString, "%Y-%m-%d")
    end_date = dt.strptime(endDateString, "%Y-%m-%d")
    date_range = pd.date_range(start_date, end_date, freq="1D", normalize=True).strftime("%Y%m%d")
  elif frequency == "monthly":
    start_date = dt.strptime(startDateString, "%Y-%m")
    end_date = dt.strptime(endDateString, "%Y-%m")
    date_range = pd.date_range(start_date, end_date, freq="1ME", normalize=True).strftime("%Y%m")
  else:
    start_date = dt.strptime(startDateString, "%Y")
    end_date = dt.strptime(endDateString, "%Y")
    date_range = pd.date_range(start_date, end_date, freq="1Y", normalize=True).strftime("%Y")

  if start_date > end_date:
    raise ValueError(f"Start date {start_date} must be before end date {end_date}")

  return date_range


def clean_up_files(files: list) -> None:
  for file_path in files:
    pathlib.Path(file_path).unlink(missing_ok=True)


def download_file(var: str, date: str, output_dir: str, args) -> str | None:
  url = f"{BASE_URL}/{args.region}/{args.resolution}/{var}/{date}"
  try:
    response = requests.get(url, params={"format": args.format}, stream=True, timeout=60)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()

    base = os.path.join(output_dir, f"{var}_{date}_{args.resolution}")
    out_path = base + (".zip" if "zip" in content_type else ".nc")

    with open(out_path, "wb") as handle:
      for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
          handle.write(chunk)

    print(f"[DOWNLOAD] {url} -> {out_path} ({content_type or 'unknown content-type'})")
    return out_path
  except Exception as exc:
    print(f"[DOWNLOAD-ERROR] {url} -> {exc}")
    return None


if __name__ == "__main__":
  args = setupArgs()
  configure_spatial_env(args.projLib, args.gdalData)
  parameters = args.parameters
  dates = parseDateRange(args.startDate, args.endDate, args.frequency)
  output_dir = args.outputDir.rstrip("/")

  start_time = dt.now()
  with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [
      executor.submit(download_file, var, date, output_dir, args)
      for var in parameters
      for date in dates
    ]
    zip_paths = [future.result() for future in futures]
  end_time = dt.now()

  print(
    "Time to download {} {}(s): {} seconds".format(
      len(dates), args.frequency, (end_time - start_time).seconds
    )
  )

  mask = gpd.read_file(args.geojson)
  if args.format != "nc":
    print("Will only merge and write to Zarr when format is NetCDF.")
  else:
    print("Creating Zarr dataset...")
    create_prism_dataset(
      args.startDate,
      args.endDate,
      output_dir,
      mask,
      zip_paths,
      args.frequency,
      args.resolution,
    )
    print("Zarr dataset created.")

  if args.keepZip:
    print("Keeping zipped files...")
  else:
    print("Cleaning up zipped files...")
    clean_up_files([path for path in zip_paths if path])

  print("Cleaning up extracted files...")
  clean_up_files(
    [os.path.join(output_dir, file) for file in os.listdir(output_dir) if file.endswith(".nc")]
  )
