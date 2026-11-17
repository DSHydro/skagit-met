"""
Build the CONUS404 daily precipitation Zarr used by the Skagit analysis.

Source:
    CONUS404 daily (USGS HyTEST), PREC_ACC_NC = precipitation accumulated over each day (mm),
    read lazily over HTTPS from the public USGS Open Storage Network copy:
        https://usgs.osn.mghpcc.org/hytest/conus404/conus404_daily.zarr
    (the zarr behind HyTEST intake catalog entry conus404-catalog:conus404-daily-osn).
    No login or extra packages (intake / s3fs) are needed.

Output (Zarr v2):  <VAULT_DIR>/conus404_daily/conus404_skagit_precip_daily_full.zarr
    precip_daily        (time, y, x)  daily precipitation on the full native CONUS404 grid, mm
    precip_basin_daily  (time)        mean over the SkagitBoundary.json basin, mm
    basin_mask          (y, x)        1 inside SkagitBoundary.json, 0 outside
    lat, lon            (y, x)        grid cell centers

    The analysis (analysis/3_bulk_bias_new_version.py and the plotting scripts) reads
    precip_daily and averages it over its own HUC8 subbasin mask.

Usage:
    python conus404_daily_downloader.py [--startYear 1981] [--endYear 2024] [--output PATH] [--overwrite]

    Years are calendar years (Jan 1 - Dec 31, or the end of the data; CONUS404 daily
    currently ends 2024-09-30). The full 1981-2024 run writes ~89 GB. If the output already
    exists the script stops without changing anything; pass --overwrite to replace it.

Note on earlier versions of this file:
    The conus404_skagit_precip_daily_full.zarr produced before this script (commit 598e77a)
    only had data in the 36 x 33 grid-cell rectangle around SkagitBoundary.json
    (y 807-842, x 240-272), with NaN elsewhere. That rectangle misses 8 of the 517 cells
    of the HUC8 subbasin mask used by the analysis, so CONUS404 basin means computed from
    it used 509 cells. This script writes the full grid, so all 517 cells are used.
    Where both versions have data, the values are identical (checked for 2021).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------- Config (edit as needed) ----------------
# CONUS404 daily on the USGS Open Storage Network (public, no login). This is the zarr behind
# the HyTEST intake catalog entry conus404-catalog:conus404-daily-osn
# (s3://hytest/conus404/conus404_daily.zarr, endpoint https://usgs.osn.mghpcc.org/).
SOURCE_URL = "https://usgs.osn.mghpcc.org/hytest/conus404/conus404_daily.zarr"
SOURCE_LABEL = "intake:conus404-catalog:conus404-daily-osn"  # same label as the existing output's attrs
PRECIP_VAR = "PREC_ACC_NC"    # daily (24-hour) accumulated precipitation, mm
BASE_DIR = Path(__file__).resolve().parents[1]  # skagit_met_for_publication/
DEFAULT_BOUNDARY = BASE_DIR / "data" / "GIS" / "SkagitBoundary.json"
VAULT_DIR = Path("/data0/skagit_met/data_transfer/data")  # shared folder with all downloaded products
DEFAULT_OUTPUT = VAULT_DIR / "conus404_daily" / "conus404_skagit_precip_daily_full.zarr"
DEFAULT_START = 1981
DEFAULT_END = 2024  # CONUS404 daily currently ends 2024-09-30


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Extract Skagit basin daily precipitation from CONUS404 daily (USGS HyTEST, OSN)."
  )
  parser.add_argument(
    "--boundary",
    type=Path,
    default=DEFAULT_BOUNDARY,
    help="Path to the basin GeoJSON boundary (default: <publication>/data/GIS/SkagitBoundary.json).",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=DEFAULT_OUTPUT,
    help="Output Zarr path (default: <VAULT_DIR>/conus404_daily/conus404_skagit_precip_daily_full.zarr).",
  )
  parser.add_argument(
    "--startYear",
    type=int,
    default=DEFAULT_START,
    help=f"First calendar year to include, from Jan 1 (default: {DEFAULT_START}).",
  )
  parser.add_argument(
    "--endYear",
    type=int,
    default=DEFAULT_END,
    help=f"Last calendar year to include, through Dec 31 or the end of the data (default: {DEFAULT_END}).",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace the output Zarr if it already exists (default: refuse and exit).",
  )
  parser.add_argument(
    "--source",
    type=str,
    default=SOURCE_URL,
    help="URL of the CONUS404 daily zarr (default: USGS OSN copy).",
  )
  return parser.parse_args()


# ---------------- Helpers ----------------
def read_geom(path: Path):
  obj = json.load(open(path))
  if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
    return obj["features"][0]["geometry"]
  if isinstance(obj, dict) and obj.get("type") == "Feature":
    return obj["geometry"]
  return obj  # assume bare geometry


def find_lon_lat(ds):
  pairs = [
    ("lon", "lat"),
    ("longitude", "latitude"),
    ("XLONG_M", "XLAT_M"),
    ("XLONG", "XLAT"),
  ]
  for lonn, latn in pairs:
    if (lonn in ds) and (latn in ds):
      lon = ds[lonn]
      lat = ds[latn]
      if "time" in lon.dims:
        lon = lon.isel(time=0, drop=True)
      if "time" in lat.dims:
        lat = lat.isel(time=0, drop=True)
      rename = {}
      if "south_north" in lat.dims:
        rename["south_north"] = "y"
      if "west_east" in lat.dims:
        rename["west_east"] = "x"
      if rename:
        lat = lat.rename(rename)
        lon = lon.rename(rename)
      return lon, lat
  raise KeyError("Could not locate lon/lat in dataset.")


def check_daily_mm(da):
  """
  Confirm the precipitation is already a daily total in mm, and return it unchanged.

  In CONUS404 daily, PREC_ACC_NC is the precipitation accumulated over each day
  (integration_length = "24-hour accumulation"), not a running total, so it must
  NOT be differenced between days.
  """
  units = str(da.attrs.get("units", "")).strip().lower()
  integration = str(da.attrs.get("integration_length", "")).lower()
  if units != "mm":
    raise ValueError(f"Expected {da.name} in mm, got units={units!r}")
  if "24-hour" not in integration:
    raise ValueError(f"Expected a 24-hour accumulation, got integration_length={integration!r}")
  return da


def build_mask(lon, lat, geom):
  import geopandas as gpd
  import regionmask
  from shapely.geometry import shape

  poly = shape(geom)
  gdf = gpd.GeoDataFrame({"name": ["Skagit"]}, geometry=[poly], crs="EPSG:4326")
  try:
    regs = regionmask.Regions.from_geopandas(gdf, names="name", name="basin")
    m = regs.mask(lon, lat)  # NaN outside, 0 inside
    mask = ~m.isnull()
  except AttributeError:
    m = regionmask.mask_geopandas(gdf, lon, lat)
    mask = (
      (~m.isnull())
      if isinstance(m, xr.DataArray)
      else xr.DataArray(np.where(np.isnan(m), False, True), dims=lat.dims, coords=lat.coords)
    )
  if mask.sum() == 0:
    raise RuntimeError("Polygon produced empty mask on this grid.")
  return mask


# ---------- Open the CONUS404 daily dataset ----------
def open_conus404_daily(source_url: str = SOURCE_URL):
  """Open the CONUS404 daily zarr lazily over HTTPS (only metadata is read here)."""
  ds = xr.open_dataset(source_url, engine="zarr", consolidated=True, chunks={})
  if PRECIP_VAR not in ds:
    raise KeyError(f"{PRECIP_VAR} not found in {source_url}")
  print(f"[source] {source_url} (precip var: {PRECIP_VAR})")
  return ds, PRECIP_VAR, SOURCE_LABEL


# --- chunk helper: handle both tuple-of-tuples (old) and dict (new) styles
def first_chunk(da: xr.DataArray, dim: str) -> int:
  """
  Return the first chunk size along `dim`, robust to xarray/dask versions.
  """
  # Newer: dict-like .chunksizes or .chunks
  if hasattr(da, "chunksizes") and isinstance(da.chunksizes, dict) and dim in da.chunksizes:
    return int(da.chunksizes[dim][0])
  if hasattr(da, "chunks") and isinstance(da.chunks, dict) and dim in da.chunks:
    return int(da.chunks[dim][0])

  # Older: tuple-of-tuples ordered by axis
  axis = da.get_axis_num(dim)
  ch = getattr(da, "chunks", None)
  if ch is None:
    # fallback: unchunked, use size
    return int(da.sizes[dim])
  return int(ch[axis][0])


def generate_conus_precip(
  boundary_geo: Path,
  out_zarr: Path,
  start_year: int,
  end_year: int,
  source_url: str = SOURCE_URL,
  overwrite: bool = False,
) -> None:
  if start_year > end_year:
    raise ValueError("startYear must be <= endYear")

  boundary_geo = boundary_geo.expanduser()
  out_zarr = out_zarr.expanduser()
  # Check before any (slow) processing: the existing output is used by the analysis
  if out_zarr.exists() and not overwrite:
    raise SystemExit(f"{out_zarr} already exists. Re-run with --overwrite to replace it.")
  out_zarr.parent.mkdir(parents=True, exist_ok=True)

  geom = read_geom(boundary_geo)
  ds, precip_var, src_label = open_conus404_daily(source_url)
  print(f"Opened source: {src_label} | precip var: {precip_var}")

  p_all = ds[precip_var]

  t0, t1 = f"{start_year}-01-01", f"{end_year}-12-31"
  if "time" not in p_all.dims:
    raise RuntimeError("Dataset has no 'time' dimension.")
  p = p_all.sel(time=slice(t0, t1))

  inc = check_daily_mm(p)  # already mm/day; used as is

  lon, lat = find_lon_lat(ds)
  mask = build_mask(lon, lat, geom)
  weights = mask.astype("float32")
  weights = weights / weights.sum()

  daily_grid = inc.resample(time="1D").sum()
  daily_basin = (inc * weights).sum(["y", "x"], skipna=True).resample(time="1D").sum()

  ann_mean = float(daily_basin.resample(time="YE").sum().mean().compute())
  print(
    f"[check] Skagit mean annual (CONUS404) ≈ {ann_mean:.1f} mm/yr over {start_year}-{end_year}"
  )

  daily_basin = daily_basin.chunk({"time": 180})
  daily_grid = daily_grid.chunk({"time": 90, "y": 256, "x": 256})
  mask_c = mask.chunk({"y": 256, "x": 256})
  lat_c = lat.chunk({"y": 256, "x": 256})
  lon_c = lon.chunk({"y": 256, "x": 256})

  encoding_series = {
    "precip_basin_daily": {
      "chunks": (first_chunk(daily_basin, "time"),),
      "compressor": None,
      "dtype": "float32",
    }
  }

  if out_zarr.exists():
    shutil.rmtree(out_zarr)

  out1 = xr.Dataset(
    {"precip_basin_daily": daily_basin.astype("float32")},
    coords={"time": daily_basin["time"]},
    attrs={
      "source": src_label,
      "precip_var": precip_var,
      "years": f"{start_year}-{end_year}",
      "note": "precip_basin_daily: Skagit basin-weighted daily total (mm)",
    },
  )
  out1.to_zarr(out_zarr, mode="w", consolidated=True, zarr_format=2, encoding=encoding_series)
  print(f"[write] basin series → {out_zarr}")

  encoding_grid = {
    "precip_daily": {
      "chunks": (
        first_chunk(daily_grid, "time"),
        first_chunk(daily_grid, "y"),
        first_chunk(daily_grid, "x"),
      ),
      "compressor": None,
      "dtype": "float32",
    },
    "basin_mask": {
      "chunks": (first_chunk(mask_c, "y"), first_chunk(mask_c, "x")),
      "compressor": None,
      "dtype": "float32",
    },
    "lat": {
      "chunks": (first_chunk(lat_c, "y"), first_chunk(lat_c, "x")),
      "compressor": None,
      "dtype": "float32",
    },
    "lon": {
      "chunks": (first_chunk(lon_c, "y"), first_chunk(lon_c, "x")),
      "compressor": None,
      "dtype": "float32",
    },
  }

  out2 = xr.Dataset(
    data_vars={
      "precip_daily": daily_grid.astype("float32"),
      "basin_mask": mask_c.astype("float32"),
    },
    coords={
      "time": daily_grid["time"],
      "y": lat_c["y"],
      "x": lat_c["x"],
      "lat": (("y", "x"), lat_c.data.astype("float32")),
      "lon": (("y", "x"), lon_c.data.astype("float32")),
    },
    attrs={
      "source": src_label,
      "precip_var": precip_var,
      "years": f"{start_year}-{end_year}",
      "note": "precip_daily: daily sum on native grid (mm); basin_mask: True in Skagit",
    },
  )
  out2.to_zarr(out_zarr, mode="a", consolidated=True, zarr_format=2, encoding=encoding_grid)
  print(f"[write] grid + mask appended → {out_zarr}")
  print("[done] All writes completed.")


if __name__ == "__main__":
  cli_args = parse_args()
  generate_conus_precip(
    boundary_geo=cli_args.boundary,
    out_zarr=cli_args.output,
    start_year=cli_args.startYear,
    end_year=cli_args.endYear,
    source_url=cli_args.source,
    overwrite=cli_args.overwrite,
  )
