import argparse
import glob
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime as dt

import fsspec
import geopandas as gpd
import rioxarray  # noqa: F401  # registers .rio accessor
import xarray as xr
from pyproj import datadir as pyproj_datadir

import helper.ornl_mapper as mapper


def first_existing(paths):
  for candidate in paths:
    if candidate and os.path.isdir(candidate):
      return candidate
  return None


def configure_spatial_env(proj_override=None, gdal_override=None):
  conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
  candidates_proj = [
    proj_override,
    os.environ.get("PROJ_LIB"),
    os.path.join(conda_prefix, "share", "proj"),
    os.path.join(sys.prefix, "share", "proj"),
    "/usr/share/proj",
  ]
  candidates_gdal = [
    gdal_override,
    os.environ.get("GDAL_DATA"),
    os.path.join(conda_prefix, "share", "gdal"),
    os.path.join(sys.prefix, "share", "gdal"),
    "/usr/share/gdal",
  ]

  proj_dir = first_existing(candidates_proj)
  gdal_dir = first_existing(candidates_gdal)

  if not proj_dir:
    raise RuntimeError(f"Could not locate PROJ data dir. Tried: {candidates_proj}")
  if not gdal_dir:
    print(f"Warning: Could not locate GDAL data dir. Tried: {candidates_gdal}")

  os.environ["PROJ_LIB"] = proj_dir
  if gdal_dir:
    os.environ["GDAL_DATA"] = gdal_dir
  os.environ.setdefault("PROJ_NETWORK", "ON")

  try:
    pyproj_datadir.set_data_dir(proj_dir)
  except Exception as exc:
    print(f"Warning: failed to set pyproj data dir: {exc}")

  print(f"[DEBUG] Using PROJ_LIB={proj_dir}")
  if gdal_dir:
    print(f"[DEBUG] Using GDAL_DATA={gdal_dir}")


def open_and_clip_nc(nc_path, mask_gdf):
  dataset = None
  for engine in ("h5netcdf", "netcdf4"):
    try:
      dataset = xr.open_dataset(nc_path, engine=engine, chunks={})
      break
    except Exception:
      dataset = None
  if dataset is None:
    raise RuntimeError(f"Could not open {nc_path} with h5netcdf or netcdf4")

  var = list(dataset.data_vars)[0]
  data_array = dataset[var]

  rename_map = {}
  if "x" in data_array.dims:
    rename_map["x"] = "lon"
  if "y" in data_array.dims:
    rename_map["y"] = "lat"
  if rename_map:
    data_array = data_array.rename(rename_map)

  if {"lon", "lat"}.issubset(data_array.dims):
    data_array = data_array.rio.write_crs("EPSG:4326", inplace=False)
    data_array = data_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    data_array = data_array.rio.clip(mask_gdf.geometry, mask_gdf.crs)
  else:
    raise RuntimeError(f"{nc_path} missing expected lon/lat dims")

  return data_array.to_dataset(name=var)


CACHE_DIR = os.path.expanduser("~/.cache/fsspec_ornl")
os.makedirs(CACHE_DIR, exist_ok=True)


# Parse command arguments from script run in the command line
def setupArgs() -> None:
  parser = argparse.ArgumentParser(
    description="Download Daily ORNL 4KM downsampled data and clip to region and save as zarr. See https://hydrosource.ornl.gov/data/datasets/9505v3_1/"
  )
  parser.add_argument(
    "--parameters",
    type=str,
    default="",
    help="Comma seperated string containing the variables that will be downloaded - defaults to prcp,tmax,tmin,wind,rhum,srad,lrad",
  )
  parser.add_argument(
    "--startYear", type=str, required=True, help="Start year of daily data to download e.g. 1999"
  )
  parser.add_argument(
    "--endYear", type=str, required=True, help="End year of data to download e.g 2001"
  )
  parser.add_argument(
    "--outputDir",
    default=mapper.DEFAULT_OUTPUT_PATH,
    type=str,
    help="Directory/path to download data/output zarr to.",
  )
  parser.add_argument(
    "--geojson",
    default=mapper.DEFAULT_SKAGIT_GEOJSON,
    type=str,
    help="Path to/name of geo_json file that geogrpahically limits the downloaded data",
  )
  parser.add_argument(
    "--reference",
    default=mapper.DEFAULT_REF_SIM,
    type=str,
    choices=mapper.ALLOWED_REF_MET_OBS,
    help="""Reference meteorological observations to use e.g. DaymetV4 or Livneh
                        Default is DaymetV4, if only ths is priovided, will use simulation driven only from the reference data
                        If using a GCM, include the GCM name, climate scenario, and downscaling method""",
  )
  parser.add_argument(
    "--hydroModel",
    default=mapper.DEFAULT_HYDRO,
    choices=mapper.ALLOWED_HYDRO_MODELS,
    type=str,
    help="Hydro model to use e.g. VIC4 (currently only one supported)",
  )
  parser.add_argument(
    "--gcm",
    choices=mapper.ALLOWED_GCMS,
    type=str,
    help="Global climate model to use e.g. ACCESS-CM2, CNRM-ESM-1, etc",
  )
  parser.add_argument(
    "--climateScenario",
    choices=mapper.ALLOWED_CLIMATE_SCENARIOS,
    type=str,
    help="Climate scenario to use e.g. ssp585, ssp245, etc",
  )
  parser.add_argument(
    "--downscalingMethod",
    choices=mapper.ALLOWED_DOWNSCALING_METHODS,
    type=str,
    help="Downscaling method used to downscale GCM data to 4KM resolution, e.g. DBCCA",
  )
  parser.add_argument(
    "--projLib", type=str, required=False, help="Path to the PROJ data directory (sets PROJ_LIB)."
  )
  parser.add_argument(
    "--gdalData", type=str, required=False, help="Path to the GDAL data directory (sets GDAL_DATA)."
  )
  return parser.parse_args()


def pull_from_globus(url: str, retries: int = 5, blocksize: int = 1024 * 1024) -> str:
  """
  Download to simplecache, force a full read to populate the cache,
  then verify local size matches remote. Retry on mismatch.
  """
  httpfs = fsspec.filesystem("http")
  target_name = os.path.basename(url)
  local_path = os.path.join(CACHE_DIR, target_name)

  for attempt in range(1, retries + 1):
    # open via simplecache and stream the whole file to ensure it fully caches
    try:
      with fsspec.open(
        f"simplecache::{url}",
        "rb",
        simplecache={"cache_storage": CACHE_DIR, "same_names": True, "check_files": True},
      ) as f:
        while True:
          chunk = f.read(blocksize)
          if not chunk:
            break
      # verify size
      remote_size = httpfs.info(url).get("size", None)
      local_size = os.path.getsize(local_path) if os.path.exists(local_path) else -1
      if remote_size is None or local_size == remote_size:
        return local_path  # success
      else:
        # size mismatch -> retry after removing partial file
        try:
          os.remove(local_path)
        except OSError:
          pass
        time.sleep(min(2**attempt, 10))
    except Exception:
      # network hiccup: clean and retry
      try:
        if os.path.exists(local_path):
          os.remove(local_path)
      except OSError:
        pass
      if attempt == retries:
        raise
      time.sleep(min(2**attempt, 10))

  raise RuntimeError(f"Failed to download (verified) after {retries} attempts: {url}")


def create_ornl_dataset(
  start_year: str,
  end_year: str,
  dest_path: str,
  geojson: str,
  reference: str,
  gcm: str,
  climate_scenario: str,
  downscaling_method: str,
) -> xr.Dataset:
  # reference-only vs GCM path
  ref = not (gcm and climate_scenario and downscaling_method)

  # Output Zarr path
  out_name = (
    f"{start_year}_{end_year}"
    f"{('_ref_' + reference) if ref else f'_{gcm}_{climate_scenario}_{downscaling_method}'}"
    "_ORNL_data.zarr"
  )
  output_file = os.path.join(dest_path.rstrip("/"), out_name)

  # Mask geometry (force to WGS84 so it matches lon/lat grids)
  mask = gpd.read_file(geojson)
  if mask.crs is None:
    raise RuntimeError(f"GeoJSON has no CRS: {geojson}")
  mask = mask.to_crs("EPSG:4326")

  # Deterministic file list (and fail early if nothing was downloaded)
  nc_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.nc")))
  if not nc_files:
    raise RuntimeError(f"No NetCDF files found in cache: {CACHE_DIR}")

  rasters: list[xr.Dataset] = []

  for f in nc_files:
    try:
      clipped = open_and_clip_nc(
        f, mask
      )  # your helper that opens with netcdf4/h5netcdf, normalizes lon/lat, sets CRS & clips
      # Optional: chunk for nicer Zarr writes (tune as needed)
      if "time" in clipped.dims:
        clipped = clipped.chunk({"time": 30})
      rasters.append(clipped)
    except Exception as exc:
      print(f"Error opening {f}: {exc}\n Trying to continue...")
      continue

  if not rasters:
    raise RuntimeError("All NetCDFs failed to open/clip. Check engines or file formats.")

  # Merge safely and silence future warnings:
  weather_dataset = xr.merge(
    rasters,
    join="outer",  # future default will be 'exact' -> be explicit
    compat="override",  # future default will change -> be explicit
    combine_attrs="drop_conflicts",
  )

  # Cleanups
  weather_dataset = weather_dataset.drop_vars("spatial_ref", errors="ignore")
  if "time" in weather_dataset.coords:
    weather_dataset["time"] = weather_dataset.time.dt.floor("D")

  # Write Zarr (prefer v2 for broad compatibility; avoid “consolidated v3” warning)
  weather_dataset.to_zarr(
    output_file,
    mode="w",
    consolidated=False,  # avoid v3 consolidated warning in your environment
    zarr_version=2,  # most tools expect v2 today
  )

  print(f"Wrote: {output_file}")
  return weather_dataset


def parseParameters(paramString: str) -> list[str]:
  if not paramString:
    return mapper.DEFAULT_VARIABLES

  cleaned = []
  for var in paramString.split(","):
    if var in mapper.ALLOWED_VARIABLES:
      cleaned.append(var)
    else:
      print(
        f"Variable {var} not found in allowed variables list. Removing it from the download list."
      )

  return cleaned or mapper.DEFAULT_VARIABLES


def clean_up_files(files: list) -> None:
  for file_path in files:
    pathlib.Path(file_path).unlink(missing_ok=True)


def collect_future_results(futures: list) -> list[str]:
  results = []
  for future in futures:
    try:
      results.append(future.result())
    except Exception as exc:
      print(f"Error downloading file: {future}, exception: {exc}")
  return results


if __name__ == "__main__":
  # Get Arguments - model, variables, product, date range, and geo_json
  args = setupArgs()
  configure_spatial_env(args.projLib, args.gdalData)
  parameters = parseParameters(args.parameters)
  output_dir = args.outputDir.rstrip("/")

  files = mapper.generate_file_names(
    args.reference,
    args.hydroModel,
    parameters,
    args.startYear,
    args.endYear,
    args.gcm,
    args.climateScenario,
    args.downscalingMethod,
  )
  start_time = dt.now()
  with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(pull_from_globus, file) for file in files]
    wait(futures)

  downloaded_files = collect_future_results(futures)
  end_time = dt.now()

  print(
    "Time to download {} files: {} seconds".format(
      len(downloaded_files), (end_time - start_time).seconds
    )
  )

  nc_files = list(glob.iglob(os.path.join(CACHE_DIR, "*.nc")))
  if not nc_files:
    raise RuntimeError(
      "No NetCDF files were downloaded successfully; try re-running or check network."
    )

  create_ornl_dataset(
    args.startYear,
    args.endYear,
    output_dir,
    args.geojson,
    args.reference,
    args.gcm,
    args.climateScenario,
    args.downscalingMethod,
  )

  clean_up_files(list(glob.iglob(os.path.join(CACHE_DIR, "*.nc"))))
