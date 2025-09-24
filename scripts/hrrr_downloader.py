from herbie import Herbie, FastHerbie, wgrib2
import shapely
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import dask as dask
import cfgrib
import argparse
import os


# Parse command arguments from script run in the command line
def setupArgs() -> None:
  parser = argparse.ArgumentParser(
    description="Download HRRR data using Herbie, and segment to a specific geographic region"
  )
  parser.add_argument(
    "--model",
    default="hrrr",
    type=str,
    required=False,
    help="Model of data to download, defaults to hrrr",
  )
  parser.add_argument(
    "--product",
    default="sfc",
    type=str,
    required=False,
    help="Model product to download, defuaults to surface level features",
  )
  parser.add_argument(
    "--parameters",
    type=str,
    required=True,
    help="Comma seperated tring containing the variables and level of the vars that will be downloaded e.g. TMP:surface,RH:2 m above ground",
  )
  parser.add_argument(
    "--startDate",
    default="2020-01-01",
    type=str,
    required=True,
    help="Start date of data to download e.g. 2020-10-01 for October 1, 2020",
  )
  parser.add_argument(
    "--endDate",
    default="2020-01-02",
    type=str,
    required=True,
    help="End date of data to download e.g. 2020-10-01 for October 1, 2020",
  )
  parser.add_argument(
    "--geoJson",
    default="data/GIS/SkagitBoundary.json",
    type=str,
    required=False,
    help="Path to/name of geo_json file that geogrpahically limits the downloaded data",
  )
  parser.add_argument(
    "--outputDir",
    default="data/weather_data/",
    type=str,
    help="Directory/path to download data/output zarr to.",
  )
  return parser.parse_args()


def getFastHerbie(
  start_date: str, end_date: str, model: str, product: str, save_dir: str
) -> FastHerbie:
  date_range = pd.date_range(start=start_date, end=end_date, freq="1h")
  return FastHerbie(date_range, model=model, product=product, fxx=range(0, 2), save_dir=save_dir)


# Parse GeoJson File into tuple containing boundaries
def parseGeoJson(geojson_path: str) -> tuple[float, float, float, float]:
  mask = gpd.read_file(geojson_path)
  minLon, minLat, maxLon, maxLat = mask.total_bounds
  return (minLon, maxLon, minLat, maxLat)


def limitGeographicRange(bounds: tuple[float, float, float, float], subsetFiles: list) -> list:
  return [wgrib2.region(f, bounds, name="skagit-basin") for f in subsetFiles]


# Use Fast herbie to subset and download parameters
def downloadParameters(parameters: list[str], fh: FastHerbie) -> list[Herbie]:
  fields = [f":{param}" for param in parameters]
  param_regex = rf"^(?:{'|'.join(fields)})"
  print("Search String: " + param_regex)
  return fh.download(param_regex)


def parseParameters(paramString: str) -> list[str]:
  return paramString.split(",")


def cleanUpFiles(subsetFiles: list) -> None:
  [os.unlink(f) for f in subsetFiles]


def maskDataset(ds: xr.Dataset, mask_file: str) -> xr.Dataset:
  mask_shape = gpd.read_file(mask_file)
  mask = shapely.contains_xy(mask_shape.geometry[0], ds.longitude.values, ds.latitude.values)
  masked_data_set = ds.where(mask)

  return masked_data_set


def mergeDatasets(regionSubsetGribFiles: list) -> xr.Dataset:
  datasets = []
  dropVars = ["surface", "heightAboveGround", "valid_time", "step"]
  # if f001, grab just the accumlated precip by dropping the other forecast variables
  dropVarsStep = dropVars + ["t", "r2", "si10", "sdswrf", "sdlwrf"]
  for f in regionSubsetGribFiles:
    unMergedDatasets = cfgrib.open_datasets(f, indexpath="", decode_timedelta=False)
    mergedDataset = xr.merge(
      [
        ds.drop_vars(dropVarsStep, errors="ignore")
        if ds.step.values == np.timedelta64(1, "h")
        else ds.drop_vars(dropVars, errors="ignore")
        for ds in unMergedDatasets
      ]
    )
    mergedDataset.load()
    datasets.append(mergedDataset)

  other_vars = [ds for ds in datasets if "tp" not in ds.variables]
  tp_f001 = [ds for ds in datasets if "tp" in ds.variables]

  tp_ds = xr.concat(tp_f001, dim="time")
  other_ds = xr.concat(other_vars, dim="time")
  combined_ds = xr.combine_by_coords([tp_ds, other_ds], compat="override")
  # Set Longitude to be in correct space
  combined_ds["longitude"] = combined_ds.longitude - 360

  return combined_ds


def write_to_zarr(dataset: xr.Dataset, output_dir: str, path: str) -> None:
  if output_dir[-1] == "/":
    output_dir = output_dir[:-1]

  dataset.to_zarr(output_dir + "/" + path, mode="w")


if __name__ == "__main__":
  # Get Arguments - model, variables, product, date range, and geo_json
  args = setupArgs()
  parameters = parseParameters(args.parameters)
  fh = getFastHerbie(args.startDate, args.endDate, args.model, args.product, args.outputDir)
  fh_files = downloadParameters(parameters, fh)
  bounds = parseGeoJson(args.geoJson)
  geo_limited_files = limitGeographicRange(bounds, fh_files)
  mergedDs = mergeDatasets(geo_limited_files)
  maskedDs = maskDataset(mergedDs, args.geoJson)
  write_to_zarr(maskedDs, args.outputDir, args.startDate + "_" + args.endDate + "_HRRR_data.zarr")
  cleanUpFiles(fh_files)
  cleanUpFiles([str(f) + ".idx" for f in geo_limited_files])
  cleanUpFiles(geo_limited_files)
