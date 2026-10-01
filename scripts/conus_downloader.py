# ========================= CONUS404 → Skagit daily precip (Zarr v2) =========================
# Defensive Intake walk (skip parquet/missing plugins), increments→mm, basin+grid,
# Zarr v2 with uniform chunking, two-phase write (series first, then grid+mask)
# ============================================================================================

import argparse
import json
import shutil
from pathlib import Path

import intake
import numpy as np
import xarray as xr

# ---------------- Config (edit as needed) ----------------
CAT_URL = "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml"
DEFAULT_BOUNDARY = Path("data/GIS/SkagitBoundary.json")
DEFAULT_OUTPUT = Path("data/weather_data/conus404_skagit_precip_daily.zarr")
DEFAULT_START = 2014
DEFAULT_END = 2020


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Extract Skagit basin daily precipitation from CONUS404 via the HyTEST Intake catalog."
  )
  parser.add_argument(
    "--boundary",
    type=Path,
    default=DEFAULT_BOUNDARY,
    help="Path to the basin GeoJSON boundary (default: data/GIS/SkagitBoundary.json).",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=DEFAULT_OUTPUT,
    help="Output Zarr path (default: data/weather_data/conus404_skagit_precip_daily.zarr).",
  )
  parser.add_argument(
    "--startYear",
    type=int,
    default=DEFAULT_START,
    help="First water year to include (default: 2014).",
  )
  parser.add_argument(
    "--endYear",
    type=int,
    default=DEFAULT_END,
    help="Last water year to include (default: 2020).",
  )
  parser.add_argument(
    "--catalog",
    type=str,
    default=CAT_URL,
    help="HyTEST Intake catalog URL (default: CONUS404 entry in main catalog).",
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


def guess_precip_var(ds):
  cand = set(ds.data_vars)
  for k in ["PREC_ACC_NC", "APCP", "TP", "TOT_PREC", "pr", "precip"]:
    if k in cand:
      return k
  if {"RAINNC", "RAINC"} <= cand:
    return "RAINNC+RAINC"
  for k in ["RAINNC", "RAINC", "PREC_ACC_C"]:
    if k in cand:
      return k
  raise KeyError("No recognizable precip variable found.")


def to_increments_mm(da, time_dim="time"):
  """
  Make per-step increments and ensure units are mm.
  Handles cumulative-with-resets and common unit conventions (kg m^-2, meters).
  """
  units = str(da.attrs.get("units", "")).lower()

  # kg m^-2 == mm
  if ("kg" in units and "m-2" in units) or ("kg m" in units and "-2" in units):
    da = da.copy()
    da.attrs["units"] = "mm"

  # meters -> mm (common for WRF accum)
  if units.strip() in {"m", "meter", "meters"}:
    da = da * 1000.0
    da = da.copy()
    da.attrs["units"] = "mm"

  # Decide cumulative vs already-incremental from a tiny slice
  probe = da
  for d in da.dims:
    if d != time_dim:
      probe = probe.isel({d: 0})
  if probe.sizes.get(time_dim, 0) > 3:
    s = np.asarray(probe.values).squeeze()
    dif = np.diff(s)
    is_accum = (np.count_nonzero(dif < -1e-6) / max(1, dif.size)) < 0.05
  else:
    is_accum = True  # conservative

  if is_accum or "acc" in (da.name or "").lower():
    da = da.sortby(time_dim)
    inc = da.diff(time_dim, label="upper")
    inc = inc.where(inc >= 0, 0).fillna(0)
    inc.attrs["units"] = "mm"
    return inc

  # already per-step
  da = da.copy()
  if da.attrs.get("units", "").lower() not in {"mm", "millimeter", "millimeters"}:
    # last resort: assume mm-like if in kg m^-2 or already handled above
    da.attrs["units"] = "mm"
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


# ---------- Defensive Intake walk (skip parquet / missing plugins) ----------
def _safe_list(cat):
  try:
    return list(cat)
  except Exception:
    return []


def iter_entries_safe(cat, prefix=""):
  """
  Yield (path_string, entry) for nested sub-catalogs without forcing plugin loads.
  Skips entries whose driver clearly needs a missing plugin (e.g., parquet).
  """
  for key in _safe_list(cat):
    try:
      entry = cat[key]
    except Exception:
      continue

    path = f"{prefix}{key}"
    drv = (getattr(entry, "_driver", "") or "").lower()
    if "parquet" in drv:
      continue

    try:
      if entry.container == "catalog":
        subcat = entry
        for subk in _safe_list(subcat):
          try:
            subentry = subcat[subk]
          except Exception:
            continue
          subpath = f"{path}:{subk}"
          subdrv = (getattr(subentry, "_driver", "") or "").lower()
          if "parquet" in subdrv:
            continue
          if getattr(subentry, "container", None) == "catalog":
            try:
              subsub = subentry
              for subsubk in _safe_list(subsub):
                try:
                  subsubentry = subsub[subsubk]
                except Exception:
                  continue
                subsubpath = f"{subpath}:{subsubk}"
                subsubdrv = (getattr(subsubentry, "_driver", "") or "").lower()
                if "parquet" in subsubdrv:
                  continue
                yield subsubpath, subsubentry
            except Exception:
              continue
          else:
            yield subpath, subentry
      else:
        yield path, entry
    except Exception:
      continue


def open_conus_source_intake_only(catalog_url: str = CAT_URL):
  """Find a CONUS404 precipitation-capable entry via Intake; open with adapter defaults."""
  cat = intake.open_catalog(catalog_url)
  tried = []
  for path, entry in iter_entries_safe(cat):
    if "conus404" not in path.lower():
      continue

    drv = (getattr(entry, "_driver", "") or "").lower()
    if "parquet" in drv:
      continue

    # IMPORTANT: do NOT pass storage_options here (this adapter rejects it)
    try:
      ds = entry.to_dask()
    except Exception as e:
      tried.append((path, f"open failed: {type(e).__name__}: {e}"))
      continue

    if not hasattr(ds, "data_vars") or "time" not in ds.dims:
      tried.append((path, "opened but not xarray/time"))
      continue

    try:
      pv = guess_precip_var(ds)
      print(f"[source] using intake entry: {path} (precip var: {pv})")
      return ds, pv, f"intake:{path}"
    except Exception:
      tried.append((path, "no precip var"))
      continue

  msg = "No precip-capable CONUS404 source found via intake."
  if tried:
    preview = "\n".join(f" - {p}: {why}" for p, why in tried[:15])
    if len(tried) > 15:
      preview += f"\n ... and {len(tried) - 15} more."
    msg += "\nTried:\n" + preview
  raise RuntimeError(msg)


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
  catalog_url: str = CAT_URL,
) -> None:
  if start_year > end_year:
    raise ValueError("startYear must be <= endYear")

  boundary_geo = boundary_geo.expanduser()
  out_zarr = out_zarr.expanduser()
  out_zarr.parent.mkdir(parents=True, exist_ok=True)

  geom = read_geom(boundary_geo)
  ds, precip_var, src_label = open_conus_source_intake_only(catalog_url)
  print(f"Opened source: {src_label} | precip var: {precip_var}")

  if precip_var == "RAINNC+RAINC":
    p_all = (ds["RAINNC"] + ds["RAINC"]).rename("PREC_TOTAL")
  else:
    p_all = ds[precip_var]

  t0, t1 = f"{start_year}-01-01", f"{end_year}-12-31"
  if "time" not in p_all.dims:
    raise RuntimeError("Dataset has no 'time' dimension.")
  p = p_all.sel(time=slice(t0, t1))

  inc = to_increments_mm(p, "time")

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
    catalog_url=cli_args.catalog,
  )
