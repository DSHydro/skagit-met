import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import regionmask
import xarray as xr
from dask.diagnostics import ProgressBar
from shapely.geometry import LinearRing, Polygon, shape

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_GEOJSON = PROJECT_ROOT / "data/GIS/SkagitBoundary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/CONUS"

OSN_ENDPOINT = "https://usgs.osn.mghpcc.org"
STORE_URL = "s3://hytest/conus404/conus404_{}.zarr"
DATASETS = {
  "daily",
  "hourly",
  "monthly",
}
DEFAULT_VARS = [
  "T2",
  "Q2",
  "U10",
  "V10",
  "PSFC",
  "SWDNB",
  "LWDNB",
  "RAINC",
  "RAINNC",
  "SNOWNC",
  "SMOIS",
  "TSLB",
  "HFX",
  "LH",
  "HGT",
]


def setupArgs() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Download CONUS404 data from the USGS OSN public S3 endpoint, clip to a geographic region, "
      "compute derived variables, and save as Zarr. "
    )
  )
  parser.add_argument(
    "--datasetKind",
    default="daily",
    type=str,
    choices=list(DATASETS),
    help="Temporal resolution of the CONUS404 dataset (daily or hourly). Default: daily.",
  )
  parser.add_argument(
    "--startDate",
    default=None,
    type=str,
    help="Start date (inclusive) to subset, e.g. 2014-10-01. Defaults to full dataset range.",
  )
  parser.add_argument(
    "--endDate",
    default=None,
    type=str,
    help="End date (inclusive) to subset, e.g. 2023-12-31. Defaults to full dataset range.",
  )
  parser.add_argument(
    "--parameters",
    type=str,
    default=",".join(DEFAULT_VARS),
    help=(
      f"Comma-separated list of CONUS404 variables to keep. Defaults to: {','.join(DEFAULT_VARS)}"
    ),
  )
  parser.add_argument(
    "--noDerivedVars",
    dest="derivedVars",
    action="store_false",
    help="Disable computation of derived variables.",
  )
  parser.add_argument(
    "--geojson",
    default=str(DEFAULT_GEOJSON),
    type=str,
    help=f"Path to a GeoJSON file used to spatially mask the data. Default: {DEFAULT_GEOJSON}",
  )
  parser.add_argument(
    "--outputDir",
    default=str(DEFAULT_OUTPUT_DIR),
    type=str,
    help=f"Output path for the Zarr store. Default: {DEFAULT_OUTPUT_DIR}",
  )
  return parser.parse_args()


def load_boundary(json_path: str) -> gpd.GeoDataFrame:
  """Load a boundary GeoJSON into a GeoDataFrame."""
  with open(json_path) as f:
    obj = json.load(f)

  basin_geom = None
  if isinstance(obj, dict) and "type" in obj:
    try:
      if obj["type"] == "FeatureCollection":
        geom = shape(obj["features"][0]["geometry"])
      elif obj["type"] == "Feature":
        geom = shape(obj["geometry"])
      else:
        geom = shape(obj)
      basin_geom = geom
    except Exception:
      pass

  # Fallback: {"lon": [...], "lat": [...]}
  if basin_geom is None and isinstance(obj, dict) and "lon" in obj and "lat" in obj:
    coords = list(zip(obj["lon"], obj["lat"]))
    # Closes the polygon ring
    if coords[0] != coords[-1]:
      coords.append(coords[0])
    basin_geom = Polygon(LinearRing(coords))

  if basin_geom is None:
    raise ValueError(
      'Invalid boundary file format. Must be GeoJSON or {"lon": [...], "lat": [...]}.'
    )

  return gpd.GeoDataFrame({"name": ["boundary"]}, geometry=[basin_geom], crs="EPSG:4326")


def open_conus404(dataset_kind: str) -> xr.Dataset:
  """Open the CONUS404 Zarr store from the USGS OSN public endpoint."""
  store_url = STORE_URL.format(dataset_kind)
  return xr.open_zarr(
    store=store_url,
    storage_options={"anon": True, "client_kwargs": {"endpoint_url": OSN_ENDPOINT}},
    consolidated=True,
  )


def subset_dataset(
  ds: xr.Dataset, start_date: str | None, end_date: str | None, variables: list[str]
) -> xr.Dataset:
  """Slice time and filter to requested variables."""
  if start_date or end_date:
    ds = ds.sel(time=slice(start_date, end_date))

  available = set(ds.data_vars)

  if missing := [v for v in variables if v not in available]:
    print(f"Warning: variables not found in dataset and will be ignored: {missing}")

  keep = [v for v in variables if v in available]
  if not keep:
    raise ValueError(
      f"None of the requested variables {variables} are available. Available variables: {sorted(available)[:25]}"
    )

  return ds[keep]


def get_spatial_dims(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, tuple[str, str]]:
  """Locate 2D lat/lon coordinate arrays and return them together with their dimension names."""
  for lat_name, lon_name in [("lat", "lon"), ("XLAT", "XLONG"), ("latitude", "longitude")]:
    if lat_name in ds and lon_name in ds:
      da_lat = ds[lat_name]
      da_lon = ds[lon_name]
      return da_lat, da_lon, tuple(da_lat.dims)
  raise KeyError(
    "Could not find lat/lon coordinate variables in dataset. "
    f"Available variables: {list(ds.coords)}"
  )


def apply_spatial_mask(
  ds: xr.Dataset, gdf: gpd.GeoDataFrame, da_lon: xr.DataArray, da_lat: xr.DataArray
) -> xr.Dataset:
  """Mask the dataset to the boundary polygon using regionmask."""
  gdf_exploded = gdf.explode(ignore_index=True)

  if hasattr(regionmask.Regions, "from_geopandas"):
    regions = regionmask.Regions.from_geopandas(gdf_exploded, names="name")
  else:
    outlines = list(filter(None, gdf_exploded.geometry))
    regions = regionmask.Regions(
      outlines=outlines,
      names=["boundary"] * len(outlines),
      numbers=list(range(len(outlines))),
      name="boundary",
    )

  mask = regions.mask(da_lon, da_lat)
  return ds.where(mask.notnull())


def compute_derived_vars(ds: xr.Dataset) -> xr.Dataset:
  """Add derived variables (WS10, PRECIP_TOT, RH, VPD) when source variables are available."""
  # Compute WS10
  if {"U10", "V10"} <= set(ds.data_vars):
    ds["WS10"] = (ds["U10"] ** 2 + ds["V10"] ** 2) ** 0.5
    ds["WS10"].attrs.update(units="m s-1", long_name="10 m wind speed")

  # Compute PRECIP_TOT
  has_rain = {"RAINC", "RAINNC"} <= set(ds.data_vars)
  has_snow = "SNOWNC" in ds.data_vars
  if has_rain or has_snow:
    pieces = []
    if has_rain:
      pieces.append(ds["RAINC"] + ds["RAINNC"])
    if has_snow:
      pieces.append(ds["SNOWNC"])
    ds["PRECIP_TOT"] = sum(pieces)
    ds["PRECIP_TOT"].attrs.update(units="mm", long_name="Total precipitation (rain+snow)")

  # Compute RH and VPD
  if {"T2", "Q2", "PSFC"} <= set(ds.data_vars):
    t_c = ds["T2"] - 273.15
    p_kpa = ds["PSFC"] / 1000.0
    q = ds["Q2"]
    es_kpa = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    e_kpa = (q * p_kpa) / (0.622 + 0.378 * q)
    ds["RH"] = (e_kpa / es_kpa).clip(0, 1) * 100.0
    ds["VPD"] = (es_kpa - e_kpa).clip(min=0)
    ds["RH"].attrs.update(units="%", long_name="Relative Humidity")
    ds["VPD"].attrs.update(units="kPa", long_name="Vapor Pressure Deficit")

  return ds


def rechunk_dataset(ds: xr.Dataset, spatial_dims: tuple[str, str], dataset_kind: str) -> xr.Dataset:
  """Apply appropriate chunking and strip stale encoding chunks."""
  time_chunk = 24 * 7 if dataset_kind == "hourly" else 30
  chunks = {"time": time_chunk, spatial_dims[0]: 300, spatial_dims[1]: 300}
  ds = ds.chunk(chunks)

  # Remove bad encodings (prevents overlapping chunk error)
  for v in ds.variables:
    ds[v].encoding.pop("chunks", None)

  # Re-chunk the coordinate arrays
  coord_chunks = {spatial_dims[0]: 300, spatial_dims[1]: 300}
  for coord in ("lat", "lon", "XLAT", "XLONG", "latitude", "longitude"):
    if coord in ds:
      ds[coord] = ds[coord].chunk(coord_chunks)

  return ds


def write_to_zarr(ds: xr.Dataset, output_path: str) -> None:
  """Write the dataset to a Zarr v2 store."""
  Path(output_path).parent.mkdir(parents=True, exist_ok=True)
  print(f"Writing to {output_path} ...")
  with ProgressBar():
    ds.to_zarr(output_path, mode="w", consolidated=True, zarr_version=2)
  print(f"Wrote: {output_path}")


if __name__ == "__main__":
  args = setupArgs()
  variables = [v.strip() for v in args.parameters.split(",") if v.strip()]

  output_dir = args.outputDir.rstrip("/")
  os.makedirs(output_dir, exist_ok=True)

  start = args.startDate.replace("-", "") if args.startDate else None
  end = args.endDate.replace("-", "") if args.endDate else None
  out_zarr = os.path.join(output_dir, f"conus404_{start}-{end}_{args.datasetKind}.zarr")

  print(f"Opening CONUS404 {args.datasetKind} dataset from OSN...")
  ds = open_conus404(args.datasetKind)

  print("Subsetting time range and variables...")
  ds = subset_dataset(ds, args.startDate, args.endDate, variables)

  print("Locating spatial coordinate arrays...")
  da_lat, da_lon, spatial_dims = get_spatial_dims(ds)

  print(f"Loading boundary from {args.geojson}...")
  gdf = load_boundary(args.geojson)

  print("Applying spatial mask...")
  ds = apply_spatial_mask(ds, gdf, da_lon, da_lat)

  if args.derivedVars:
    print("Computing derived variables...")
    ds = compute_derived_vars(ds)

  print("Rechunking dataset...")
  ds = rechunk_dataset(ds, spatial_dims, args.datasetKind)

  write_to_zarr(ds, out_zarr)
