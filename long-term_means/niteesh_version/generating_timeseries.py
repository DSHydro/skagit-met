# ==========================================================
# FULL NOTEBOOK CELL: Water-Year Total Precipitation (mm) by 4 regions
#
# OUTPUTS:
#   - Water-year total precipitation CSV by dataset x region x year
#   - 4 figures total:
#       1) Upper Skagit
#       2) Sauk
#       3) Lower Skagit
#       4) Full Skagit
#
# WATER YEAR:
#   - Labeled by ending year
#   - WY 1983 = Oct 1982 .. Sep 1983
#
# REGIONS:
#   - Upper Skagit
#   - Sauk
#   - Lower Skagit
#   - Full Skagit
#
# IMPORTANT:
#   - Full continuous timeline
#   - No seasonal panels
#   - No MAE plots
#   - Uses water-year TOTAL precipitation (mm), not mean
#   - Daymet-PC and CONUS404 are disabled by default to avoid the
#     timeout/auth issues from the previous run
# ==========================================================

import re
import sys
import time
import zipfile
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- Config ----------------
BASE = Path("/data0/skagit_met/data_transfer/data")
BOUNDARY_GEO = Path("../data/GIS/SkagitBoundary.json")
HUC8_GEO     = Path("../data/GIS/SkagitSubBasin_HUC8.geojson")

YEAR_MIN, YEAR_MAX = 1983, 2024  # water-year labels

# dataset roots
P_PRISM_DIR  = BASE / "prism_ppt"
PNNL_HIST    = BASE / "PNNL" / "historical"

OUT = Path("/data0/hernanqd/plots_code/skagit-met/long-term_means") / "derived"
OUT.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUT / f"wy_total_precip_skagit_regions_{YEAR_MIN}_{YEAR_MAX}_ALL.csv"

REGION_NAMES = ["Upper Skagit", "Sauk", "Lower Skagit", "Full Skagit"]

# Water year Y = Oct(Y-1) .. Sep(Y)
GLOBAL_START = f"{YEAR_MIN-1}-10-01"
GLOBAL_END   = f"{YEAR_MAX}-09-30"

# Remote-heavy datasets: disabled by default for stability
USE_DAYMET_PC = True
USE_CONUS404 = True

# ==========================================================
# CANONICAL COLORS / ORDER
# ==========================================================
DATASET_ORDER = [
    "PRISM 4km",
    "PRISM 800m",
    "Daymet-PC",
    "UCLA ERA5 d02",
    "PNNL hist",
    "CONUS404",
    "SNOTEL (mean stations)",
]

COLORS = {
    "PRISM 4km":     "#1B9E77",
    "PRISM 800m":    "#66C2A5",
    "Daymet-PC":     "#7570B3",
    "UCLA ERA5 d02": "#E6AB02",
    "PNNL hist":     "#D95F02",
    "CONUS404":      "#666666",
    "SNOTEL (mean stations)": "#000000",
}

def canonical_label(dsname: str) -> str:
    s = str(dsname)

    if s == "PRISM":
        return "PRISM 4km"

    if "Daymet v4" in s or "Planetary Computer" in s or "Daymet" in s:
        return "Daymet-PC"


    if "UCLA" in s and "d02" in s:
        return "UCLA ERA5 d02"

    if s.startswith("PNNL"):
        return "PNNL hist"

    if s.startswith("CONUS404"):
        return "CONUS404"

    if s.startswith("SNOTEL"):
        return "SNOTEL (mean stations)"

    return s

def color_for_dataset(dsname: str) -> str:
    return COLORS.get(canonical_label(dsname), "#000000")

# ==========================================================
# Water-year helpers
# ==========================================================
def years_for_water_year():
    return range(YEAR_MIN, YEAR_MAX + 1)

def water_year_time_window(year: int):
    return f"{year-1}-10-01", f"{year}-09-30"

def water_year_index(time_da):
    y = time_da.dt.year
    m = time_da.dt.month
    wy = xr.where(m >= 10, y + 1, y)
    return wy.rename("year")

def ensure_time_sorted_unique(da):
    if "time" not in da.dims:
        return da
    da = da.sortby("time")
    try:
        t = pd.to_datetime(da["time"].values)
        _, idx = np.unique(t, return_index=True)
        da = da.isel(time=np.sort(idx))
    except Exception:
        pass
    return da

def water_year_sum_mm(da):
    """
    da dims: ('time','region') or ('time',)
    returns dims: ('year','region') or ('year',)
    """
    if "time" not in da.dims:
        return None

    da = ensure_time_sorted_unique(da)
    wy = water_year_index(da["time"])
    out = da.groupby(wy).sum("time", skipna=True)

    if "year" not in out.dims:
        candidates = [d for d in out.dims if d not in ("time", "region")]
        if candidates:
            out = out.rename({candidates[0]: "year"})

    out.name = "mm"
    return out

# ==========================================================
# Zarr / grid helpers
# ==========================================================
def safe_open_zarr(p: Path):
    try:
        return xr.open_zarr(p, consolidated=True)
    except Exception:
        return xr.open_zarr(p, consolidated=False)

def find_lat_lon_names(ds):
    lat_name = next((n for n in ["lat", "latitude", "XLAT", "XLAT_M", "lat2d"] if n in ds), None)
    lon_name = next((n for n in ["lon", "longitude", "XLONG", "XLONG_M", "lon2d"] if n in ds), None)
    return lat_name, lon_name

def _to_2d(arr):
    if "time" in arr.dims:
        arr = arr.isel(time=0)
    for d in list(arr.dims):
        if arr.sizes[d] == 1:
            arr = arr.isel({d: 0})
    while arr.ndim > 2:
        extra = [d for d in arr.dims if d not in ("x", "y", "lon", "lat")]
        if not extra:
            break
        arr = arr.isel({extra[0]: 0})
    return arr

# ==========================================================
# Regions
# ==========================================================
def load_regions_gdf():
    """
    Load 3 HUC8 polygons clipped to Skagit boundary + Full Skagit polygon.
    Returns GeoDataFrame with 'Name' and 'geometry' in EPSG:4326.
    """
    import geopandas as gpd

    skagit_gdf = gpd.read_file(BOUNDARY_GEO).to_crs("EPSG:4326")
    huc8 = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")

    sub3 = ["Upper Skagit", "Sauk", "Lower Skagit"]
    huc8_3 = huc8[huc8["Name"].isin(sub3)].copy()
    if huc8_3.empty:
        raise ValueError(f"No matching HUC8 regions found for {sub3}. Check 'Name' field.")

    huc8_3 = gpd.overlay(huc8_3, skagit_gdf, how="intersection")
    regions_3 = huc8_3[["Name", "geometry"]].reset_index(drop=True)

    full_row = gpd.GeoDataFrame(
        {"Name": ["Full Skagit"], "geometry": [skagit_gdf.geometry.iloc[0]]},
        crs="EPSG:4326",
    )

    regions_gdf = pd.concat([regions_3, full_row], ignore_index=True)
    regions_gdf["Name"] = pd.Categorical(regions_gdf["Name"], categories=REGION_NAMES, ordered=True)
    regions_gdf = regions_gdf.sort_values("Name").reset_index(drop=True)
    return regions_gdf

def region_mean(da, grid_ds=None, regions_gdf=None):
    """
    Mean(da) inside each region polygon.
    Uses regionmask with overlap=True so Full Skagit can overlap others.
    Returns DataArray with a 'region' dimension.
    """
    if regions_gdf is None:
        raise ValueError("regions_gdf is required")

    import regionmask

    time_dim = "time" if "time" in da.dims else None
    spatial = [d for d in da.dims if d != time_dim]
    if not spatial:
        return da.expand_dims(region=list(regions_gdf["Name"].values))

    host = grid_ds if grid_ds is not None else da.to_dataset(name="_tmp")
    lat_name, lon_name = find_lat_lon_names(host)
    if not (lat_name and lon_name):
        out = da.mean(spatial, skipna=True)
        return out.expand_dims(region=list(regions_gdf["Name"].values))

    lon = _to_2d(host[lon_name])
    lat = _to_2d(host[lat_name])

    regs = regionmask.Regions(
        outlines=list(regions_gdf.geometry.values),
        names=list(regions_gdf["Name"].astype(str).values),
        numbers=list(range(len(regions_gdf))),
        name="Skagit_regions",
        overlap=True,
    )

    m3 = regs.mask_3D(lon, lat)

    spatial_dims = [d for d in da.dims if d != time_dim]
    mask_spatial_dims = [d for d in m3.dims if d != "region"]

    if set(mask_spatial_dims) != set(spatial_dims):
        rename_map = {}
        for md in mask_spatial_dims:
            for pdim in spatial_dims:
                if m3.sizes.get(md) == da.sizes.get(pdim):
                    rename_map[md] = pdim
                    break
        if rename_map:
            m3 = m3.rename(rename_map)

    out_list = []
    for i, _rname in enumerate(regs.names):
        m = m3.isel(region=i)
        w = xr.where(m, 1.0, np.nan)
        out_list.append((da * w).mean(dim=spatial_dims, skipna=True))

    out = xr.concat(
        out_list,
        dim="region",
        coords="minimal",
        compat="override",
    ).assign_coords(region=regs.names)
    return out

# ==========================================================
# Plot helpers
# ==========================================================
def _ordered_datasets_present(df_sub: pd.DataFrame):
    present = list(df_sub["dataset"].unique())
    ordered = []
    for key in DATASET_ORDER:
        ordered += [d for d in present if canonical_label(d) == key]
    ordered += [d for d in present if d not in ordered]
    return ordered

def _legend_datasets_present(df_sub: pd.DataFrame, value_col: str):
    ds_order = _ordered_datasets_present(df_sub)
    out = []
    for d in ds_order:
        sub = df_sub[df_sub["dataset"] == d]
        if sub[value_col].notna().any():
            out.append(d)
    return out

def _ylim_for_series(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)

    ymax = float(np.nanmax(v))
    if ymax <= 0:
        return (0.0, 1.0)

    return (0.0, ymax * 1.05)

def _build_legend_handles(dataset_names):
    handles = []
    for dsname in dataset_names:
        handles.append(
            Line2D(
                [0], [0],
                color=color_for_dataset(dsname),
                marker="o",
                linewidth=2.0,
                label=dsname,
            )
        )
    return handles

def plot_region_water_year_full_timeline(
    df_region: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title_prefix: str,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(24, 8))

    ds_order = _ordered_datasets_present(df_region)
    ylims = _ylim_for_series(df_region[value_col].values)

    for name in ds_order:
        g = df_region[df_region["dataset"] == name].sort_values("year")
        g = g[np.isfinite(g[value_col])]
        if g.empty:
            continue

        ax.plot(
            g["year"],
            g[value_col],
            marker="o",
            color=color_for_dataset(name),
            linewidth=2.0,
            markersize=4,
        )

    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(*ylims)
    ax.set_xlabel("Water Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    xticks = np.arange(YEAR_MIN, YEAR_MAX + 1, 2)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", rotation=45)

    legend_names = _legend_datasets_present(df_region, value_col)
    if legend_names:
        handles = _build_legend_handles(legend_names)
        fig.legend(
            handles=handles,
            labels=legend_names,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(5, len(legend_names)),
            fontsize=10,
            frameon=False,
        )

    fig.suptitle(title_prefix, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print("Saved plot →", out_path)
    plt.show()

# ==========================================================
# Collectors
# ==========================================================
regions_gdf = load_regions_gdf()
rows = []

# ==========================================================
# Daymet v4 (Planetary Computer) — WY totals
# ==========================================================
if USE_DAYMET_PC:
    def ensure_daymet_pkgs():
        needed = ["pystac-client", "planetary-computer", "pyproj", "fsspec", "zarr"]
        import importlib
        import subprocess

        missing = []
        for p in needed:
            mod = p.replace("-", "_")
            try:
                importlib.import_module(mod)
            except Exception:
                missing.append(p)

        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

    ensure_daymet_pkgs()

    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import CRS, Transformer
    from fsspec.implementations.http import HTTPFileSystem
    from shapely.ops import transform as shp_transform
    from urllib.parse import urlparse, urlunparse

    class SASAppendingHTTPFileSystem(HTTPFileSystem):
        def __init__(self, query: str, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fixed_query = query.lstrip("?")

        def _with_query(self, url: str) -> str:
            u = urlparse(url)
            q = u.query or self._fixed_query
            return urlunparse((u.scheme, u.netloc, u.path, u.params, q, u.fragment))

        def _open(self, path, mode="rb", block_size=None, **kwargs):
            return super()._open(self._with_query(path), mode=mode, block_size=block_size, **kwargs)

        async def _cat_file(self, url, start=None, end=None, **kwargs):
            return await super()._cat_file(self._with_query(url), start=start, end=end, **kwargs)

    def get_daymet_mapper():
        stac = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        col = stac.get_collection("daymet-daily-na")
        asset_key = "zarr-https" if "zarr-https" in col.assets else "zarr-abfs"
        signed = pc.sign(col.assets[asset_key]).href

        u = urlparse(signed)
        base = f"{u.scheme}://{u.netloc}{u.path}"
        query = u.query
        fs = SASAppendingHTTPFileSystem(query)
        return fs.get_mapper(base)

    def daymet_lcc_crs():
        return CRS.from_proj4(
            "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )

    _DAYMET_TFM_LL_TO_XY = Transformer.from_crs("EPSG:4326", daymet_lcc_crs(), always_xy=True)

    def lonlat_to_daymet_xy_bounds(geom_wgs84, pad_m=5000.0):
        minx, miny, maxx, maxy = geom_wgs84.bounds
        corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
        xs, ys = zip(*[_DAYMET_TFM_LL_TO_XY.transform(lon, lat) for lon, lat in corners])
        return (
            float(min(xs) - pad_m),
            float(max(xs) + pad_m),
            float(min(ys) - pad_m),
            float(max(ys) + pad_m),
        )

    def _daymet_project_geom_to_xy(geom_ll):
        def _proj(x, y, z=None):
            return _DAYMET_TFM_LL_TO_XY.transform(x, y)
        return shp_transform(_proj, geom_ll)

    _DAYMET_MASK_CACHE = {}

    def daymet_window_region_total(ds_daymet, t0: str, t1: str, region_name: str, geom_ll):
        xmin, xmax, ymin, ymax = lonlat_to_daymet_xy_bounds(geom_ll, pad_m=5000.0)

        y0 = float(ds_daymet["y"].values[0])
        y_slice = slice(ymax, ymin) if y0 > float(ds_daymet["y"].values[-1]) else slice(ymin, ymax)

        pr = ds_daymet["prcp"].sel(
            time=slice(t0, t1),
            x=slice(xmin, xmax),
            y=y_slice,
        ).chunk({"time": 366, "y": 256, "x": 256})

        key = (
            float(pr["x"].values[0]),
            float(pr["x"].values[-1]),
            float(pr["y"].values[0]),
            float(pr["y"].values[-1]),
            pr.sizes["x"],
            pr.sizes["y"],
            region_name,
        )

        mask = _DAYMET_MASK_CACHE.get(key)
        if mask is None:
            xx, yy = xr.broadcast(pr["x"], pr["y"])
            geom_xy = _daymet_project_geom_to_xy(geom_ll)
            from shapely import contains_xy
            mask_np = contains_xy(geom_xy, np.asarray(xx.values), np.asarray(yy.values))
            mask = xr.DataArray(mask_np, dims=xx.dims, coords=xx.coords)
            _DAYMET_MASK_CACHE[key] = mask

        region_daily = pr.where(mask).mean(dim=("y", "x"), skipna=True)
        return float(region_daily.sum(dim="time", skipna=True).compute().values)

    DAYMET_NAME = "Daymet v4 (Planetary Computer)"
    regions_ll = regions_gdf.to_crs("EPSG:4326")

    for yr in years_for_water_year():
        t0, t1 = water_year_time_window(yr)
        totals = {r: np.nan for r in REGION_NAMES}

        for attempt in range(1, 4):
            ds_daymet = None
            try:
                mapper = get_daymet_mapper()
                ds_daymet = xr.open_zarr(mapper, consolidated=True)
                if "prcp" not in ds_daymet:
                    raise RuntimeError("Daymet dataset missing 'prcp'")

                for rname in REGION_NAMES:
                    geom = regions_ll.loc[regions_ll["Name"] == rname, "geometry"].iloc[0]
                    totals[rname] = daymet_window_region_total(ds_daymet, t0, t1, rname, geom)

                break
            except Exception as e:
                print(f"[DAYMET-PC] WY {yr} attempt {attempt}/3 failed ({type(e).__name__}): {e}")
                time.sleep(5 * attempt)
            finally:
                if ds_daymet is not None:
                    try:
                        ds_daymet.close()
                    except Exception:
                        pass

        for rname, val in totals.items():
            if np.isfinite(val):
                rows.append({
                    "dataset": DAYMET_NAME,
                    "region": str(rname),
                    "year": int(yr),
                    "mm": float(val),
                })
else:
    print("[DAYMET-PC] Skipped (USE_DAYMET_PC=False)")

# ==========================================================
# ==========================================================

def hrrr_f06_daily_precip_mm(ds: xr.Dataset, var="tp") -> xr.DataArray:
    """
      - ds.time is INIT time
      - keep init hours 00/06/12/18
      - shift init -> valid time by +6h
      - daily totals = sum of 4x 6h accumulations per day
    """
    if var not in ds:
        raise ValueError(f"Variable '{var}' not found in dataset")

    da = ensure_time_sorted_unique(ds[var])

    t = pd.to_datetime(da["time"].values)
    keep = np.isin(t.hour, [0, 6, 12, 18])
    da = da.sel(time=da["time"].values[keep])

    da = da.assign_coords(time=(pd.to_datetime(da["time"].values) + pd.Timedelta(hours=6)))
    daily = da.resample(time="1D").sum()
    return daily

def _find_hrrr_tp_var(ds: xr.Dataset):
    if "tp" in ds.data_vars:
        return "tp"
    if "apcp" in ds.data_vars:
        return "apcp"
    for v in ds.data_vars:
        lv = v.lower()
        if lv == "tp" or "apcp" in lv or "prec" in lv:
            return v
    return None

def hrrr_month_prefixes_for_water_year(year: int):
    prev = year - 1
    return [
        f"{prev}-10", f"{prev}-11", f"{prev}-12",
        f"{year}-01", f"{year}-02", f"{year}-03",
        f"{year}-04", f"{year}-05", f"{year}-06",
        f"{year}-07", f"{year}-08", f"{year}-09",
    ]

by_prefix = {p.name[:7]: p for p in monthly}
available = set(by_prefix.keys())

if not monthly:
    pass
else:

    for yr in years_for_water_year():
            need = hrrr_month_prefixes_for_water_year(yr)
            missing = sorted([m for m in need if m not in available])
            if missing:
                continue

            t0, t1 = water_year_time_window(yr)
            wy_totals = {r: 0.0 for r in REGION_NAMES}
            any_ok = False

            ds0 = None
            try:
                ds0 = safe_open_zarr(by_prefix[need[0]])
                var0 = _find_hrrr_tp_var(ds0)
                if var0 is None:
                    continue

                for m in need:
                    p = by_prefix[m]
                    ds = None
                    try:
                        ds = safe_open_zarr(p)
                        var = var0 if var0 in ds.data_vars else _find_hrrr_tp_var(ds)
                        if var is None:
                            continue

                        daily = hrrr_f06_daily_precip_mm(ds, var=var)
                        if daily.sizes.get("time", 0) == 0:
                            continue

                        rm_daily = region_mean(daily, grid_ds=ds0, regions_gdf=regions_gdf)
                        rm_daily = ensure_time_sorted_unique(rm_daily)
                        rm_daily = rm_daily.sel(time=slice(t0, t1))
                        if rm_daily.sizes.get("time", 0) == 0:
                            continue

                        month_tot = rm_daily.sum("time", skipna=True)
                        for rname in month_tot["region"].values:
                            v = float(month_tot.sel(region=rname).values)
                            if np.isfinite(v):
                                wy_totals[str(rname)] += v
                                any_ok = True

                    except Exception as e:
                        pass
                    finally:
                        if ds is not None:
                            try:
                                ds.close()
                            except Exception:
                                pass

                if not any_ok:
                    continue

                for rname, val in wy_totals.items():
                    if np.isfinite(val):
                        rows.append({
                            "region": str(rname),
                            "year": int(yr),
                            "mm": float(val),
                        })


            finally:
                if ds0 is not None:
                    try:
                        ds0.close()
                    except Exception:
                        pass

# ==========================================================
# UCLA ERA5 WRF d02 (daily NetCDF) — WY totals
# ==========================================================
UCLA_DATASET_NAME = "UCLA ERA5 WRF d02 (daily NetCDF)"
UCLA_PREC_DIR      = Path("/data0/skagit_met/data_transfer/data/ucla_era5_d02_daily/prec")
UCLA_PREC_TEMPLATE = "prec.daily.era5.d02.{year}.nc"
UCLA_COORD_FILE    = Path("/data0/skagit_met/data_transfer/data/ucla_era5_d02_daily/static/wrfinput_d02_coord.nc")

def pick_time_dim(da):
    if "time" in da.dims:
        return "time"
    if "day" in da.dims:
        return "day"
    for d in da.dims:
        if "time" in d.lower() or "day" in d.lower():
            return d
    raise ValueError(f"[UCLA] No time-like dim found. dims={da.dims}")

def ensure_sorted_unique_time(da, tdim="time"):
    da = da.sortby(tdim)
    try:
        t = pd.to_datetime(da[tdim].values)
        _, idx = np.unique(t, return_index=True)
        da = da.isel({tdim: np.sort(idx)})
    except Exception:
        pass
    return da

def ucla_file_label_for_date(dt: pd.Timestamp) -> int:
    return dt.year if dt.month >= 9 else (dt.year - 1)

def open_ucla_prec_for_window(t0: str, t1: str):
    t0_ts = pd.Timestamp(t0)
    t1_ts = pd.Timestamp(t1)
    labels_needed = sorted({ucla_file_label_for_date(t0_ts), ucla_file_label_for_date(t1_ts)})

    files = []
    for lab in labels_needed:
        f = UCLA_PREC_DIR / UCLA_PREC_TEMPLATE.format(year=lab)
        if f.exists():
            files.append(str(f))

    if not files:
        return None, []

    dsets, pr_list = [], []
    for f in files:
        ds = xr.open_dataset(f, engine="netcdf4", decode_cf=True, mask_and_scale=True)
        if "prec" not in ds:
            ds.close()
            continue

        pr = ds["prec"]
        tdim = pick_time_dim(pr)
        if tdim != "time":
            pr = pr.rename({tdim: "time"})

        pr_list.append(pr)
        dsets.append(ds)

    if not pr_list:
        return None, []

    pr_all = xr.concat(pr_list, dim="time", join="outer", coords="minimal", compat="override")
    pr_all = ensure_sorted_unique_time(pr_all, "time")
    pr_all = pr_all.sel(time=slice(t0, t1))

    if pr_all.sizes.get("time", 0) == 0:
        for ds in dsets:
            try:
                ds.close()
            except Exception:
                pass
        return None, []

    return pr_all, dsets

def build_ucla_region_masks(coord_file: Path, regions_gdf):
    import shapely
    try:
        from shapely import contains_xy
        has_contains_xy = True
    except Exception:
        has_contains_xy = False
        from shapely.prepared import prep

    g = xr.open_dataset(coord_file, engine="netcdf4")
    lat = g["lat2d"].squeeze(drop=True).astype("float64")
    lon = g["lon2d"].squeeze(drop=True).astype("float64")
    g.close()

    lonv = lon.values.ravel()
    latv = lat.values.ravel()
    ny, nx = lon.shape

    masks = {}
    for _, row in regions_gdf.iterrows():
        name = str(row["Name"])
        geom = row["geometry"]

        if has_contains_xy:
            m_flat = contains_xy(geom, lonv, latv)
        else:
            pg = prep(geom)
            m_flat = np.array([pg.contains(shapely.Point(x, y)) for x, y in zip(lonv, latv)], dtype=bool)

        mask2d = m_flat.reshape(ny, nx).astype("float32")
        masks[name] = xr.DataArray(mask2d, dims=lon.dims, coords=lon.coords)

    return masks

def masked_mean_timeseries(pr: xr.DataArray, mask2d: xr.DataArray) -> xr.DataArray:
    tdim = "time"
    spatial_dims = [d for d in pr.dims if d != tdim]

    mask = mask2d
    if set(mask.dims) != set(spatial_dims):
        rename_map = {}
        for md in mask.dims:
            for pdim in spatial_dims:
                if mask.sizes[md] == pr.sizes[pdim]:
                    rename_map[md] = pdim
                    break
        mask = mask.rename(rename_map)

    denom = mask.sum(dim=spatial_dims, skipna=True)
    if float(denom.values) == 0.0:
        return xr.full_like(pr.isel({tdim: 0}), np.nan).expand_dims({tdim: pr[tdim]})

    pr = pr.chunk({tdim: min(366, pr.sizes[tdim])})
    num = (pr * mask).sum(dim=spatial_dims, skipna=True)
    return num / denom

if UCLA_PREC_DIR.exists() and UCLA_COORD_FILE.exists():
    UCLA_REGION_MASKS = build_ucla_region_masks(UCLA_COORD_FILE, regions_gdf)

    for yr in years_for_water_year():
        t0, t1 = water_year_time_window(yr)
        pr, parents = open_ucla_prec_for_window(t0, t1)
        if pr is None:
            continue

        try:
            for rname in REGION_NAMES:
                mask = UCLA_REGION_MASKS.get(rname)
                if mask is None:
                    continue

                ts = masked_mean_timeseries(pr, mask)
                val = float(ts.sum(dim="time", skipna=True).compute().values)
                if np.isfinite(val):
                    rows.append({
                        "dataset": UCLA_DATASET_NAME,
                        "region": str(rname),
                        "year": int(yr),
                        "mm": float(val),
                    })
        finally:
            for ds in parents:
                try:
                    ds.close()
                except Exception:
                    pass
else:
    print("[UCLA] Missing UCLA precip dir or coord file -> skipping UCLA.")

# ==========================================================
# SNOTEL — WY totals
# ==========================================================
try:
    import geopandas as gpd
    import shapely.geometry as sgeom
    import requests
except Exception as e:
    print(f"[SNOTEL] Missing dependency ({e}), skipping SNOTEL.")
else:
    print("[SNOTEL] Loading stations metadata from egagli/snotel_ccss_stations...")

    stations_url = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/all_stations.geojson"
    try:
        r = requests.get(stations_url, timeout=30)
        r.raise_for_status()
        gj = r.json()
    except Exception as e:
        print(f"[SNOTEL] Failed to download stations GeoJSON: {e}")
        gj = None

    if gj:
        records, geoms = [], []
        for feat in gj.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry")
            if geom is None:
                continue
            records.append(props)
            geoms.append(sgeom.shape(geom))

        stations = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326") if records else None
        if stations is None or stations.empty:
            print("[SNOTEL] No stations loaded, skipping.")
        else:
            skagit_geom = gpd.read_file(BOUNDARY_GEO).to_crs("EPSG:4326").geometry.iloc[0]
            inside = stations.geometry.within(skagit_geom)
            skagit_stations = stations[inside].copy()

            desired_names = {"Beaver Pass", "Brown Top", "Marten Ridge", "Rainy Pass", "Swamp Creek", "Thunder Basin"}
            if not skagit_stations.empty and "name" in skagit_stations.columns:
                skagit_stations = skagit_stations[skagit_stations["name"].isin(desired_names)]

            if skagit_stations.empty:
                print("[SNOTEL] No Skagit stations found after filtering, skipping.")
            else:
                base_csv_url = "https://raw.githubusercontent.com/egagli/snotel_ccss_stations/main/data"
                station_wy_totals = {}

                for _, st in skagit_stations.iterrows():
                    code = st.get("code")
                    if pd.isna(code):
                        continue

                    csv_url = f"{base_csv_url}/{code}.csv"
                    try:
                        df_s = pd.read_csv(csv_url, index_col="datetime", parse_dates=True)
                    except Exception as e:
                        print(f"[SNOTEL] Failed to read {code}: {e}")
                        continue

                    if "PRCPSA" not in df_s.columns:
                        continue

                    precip_mm = (df_s["PRCPSA"] * 1000.0).loc[GLOBAL_START:GLOBAL_END]
                    idx = precip_mm.index
                    wy = np.where(idx.month >= 10, idx.year + 1, idx.year)

                    tmp = pd.DataFrame({"mm": precip_mm.values, "year": wy}, index=idx)
                    tmp = tmp[tmp["year"].between(YEAR_MIN, YEAR_MAX)]

                    station_wy_totals[code] = tmp.groupby("year")["mm"].sum()

                if station_wy_totals:
                    regs3 = regions_gdf[regions_gdf["Name"].isin(["Upper Skagit", "Sauk", "Lower Skagit"])].copy().to_crs("EPSG:4326")
                    skagit_stations = skagit_stations.to_crs("EPSG:4326")

                    try:
                        joined = gpd.sjoin(skagit_stations, regs3.rename(columns={"Name": "region"}), predicate="within", how="left")
                    except TypeError:
                        joined = gpd.sjoin(skagit_stations, regs3.rename(columns={"Name": "region"}), op="within", how="left")

                    sub_joined = joined.dropna(subset=["region"]).copy()

                    if not sub_joined.empty:
                        regional_values = {}
                        for _, row in sub_joined.iterrows():
                            code = row.get("code")
                            region = row["region"]
                            if pd.isna(code) or code not in station_wy_totals:
                                continue
                            agg = station_wy_totals[code]
                            for yy, val in agg.items():
                                regional_values.setdefault((region, int(yy)), []).append(float(val))

                        for (region, yy), vals in regional_values.items():
                            rows.append({
                                "dataset": "SNOTEL (mean stations)",
                                "region": str(region),
                                "year": int(yy),
                                "mm": float(np.nanmean(vals)),
                            })

                    full_values = {}
                    for code, agg in station_wy_totals.items():
                        for yy, val in agg.items():
                            full_values.setdefault(int(yy), []).append(float(val))

                    for yy, vals in full_values.items():
                        rows.append({
                            "dataset": "SNOTEL (mean stations)",
                            "region": "Full Skagit",
                            "year": int(yy),
                            "mm": float(np.nanmean(vals)),
                        })

# ==========================================================
# PNNL (historical) — WY totals
# NOTE: your mask file is 3-region only; Full Skagit may not exist here.
# ==========================================================
PNNL_REGION_MASK = Path("/home/balaji24/skagit-met/analysis/skagit_huc8_3mask_pnnl_450x450.nc")

def months_present_in_selection(time_values):
    t = pd.to_datetime(time_values)
    return set(pd.Index(t.month).unique().tolist())

def years_needed_for_water_year(yr: int):
    return {yr - 1, yr}

if PNNL_HIST.exists() and PNNL_REGION_MASK.exists():
    mask_ds = xr.open_dataset(PNNL_REGION_MASK)
    if "mask" not in mask_ds:
        raise ValueError("[PNNL] Expected variable 'mask' in PNNL_REGION_MASK")
    if "region" not in mask_ds["mask"].dims:
        raise ValueError("[PNNL] Expected 'mask' to have a 'region' dimension")

    region_labels = [str(r) for r in mask_ds["region"].values]
    print("[PNNL] Starting water-year regional ingest...")

    need_months = set(range(1, 13))

    for yr in years_for_water_year():
        t0, t1 = water_year_time_window(yr)

        files = []
        for yy in sorted(years_needed_for_water_year(yr)):
            files += sorted(glob(str(PNNL_HIST / f"{yy}" / "*PREC_ACC_NC*.nc")))
        files = sorted(files)

        if not files:
            continue

        def _preprocess(ds):
            keep = [v for v in ds.data_vars if "PREC_ACC_NC" in v]
            return ds[keep] if keep else ds

        dsp = xr.open_mfdataset(
            files,
            combine="by_coords",
            engine="netcdf4",
            parallel=False,
            preprocess=_preprocess,
            chunks={"time": 24 * 7},
        )

        v = next((vv for vv in dsp.data_vars if "PREC_ACC_NC" in vv), None)
        if v is None:
            dsp.close()
            continue

        p = ensure_time_sorted_unique(dsp[v]).sel(time=slice(t0, t1))
        if p.sizes.get("time", 0) == 0:
            dsp.close()
            continue

        seen_months = months_present_in_selection(p["time"].values)
        missing_months = sorted(list(need_months - set(seen_months)))
        if missing_months:
            print(f"[PNNL] SKIP WY {yr}: missing months {missing_months}")
            dsp.close()
            continue

        P = p.sum(dim="time", skipna=True)

        spatial_dims = [d for d in P.dims if d != "time"]
        if len(spatial_dims) != 2:
            dsp.close()
            raise ValueError(f"[PNNL] Unexpected spatial dims for {v}: {P.dims}")

        for rname in region_labels:
            mr = mask_ds["mask"].sel(region=rname)

            if set(mr.dims) != set(spatial_dims):
                mr = mr.rename({mr.dims[0]: spatial_dims[0], mr.dims[1]: spatial_dims[1]})

            val = P.where(mr).mean(dim=spatial_dims, skipna=True).compute()
            mm = float(val.values)

            if np.isfinite(mm):
                rows.append({
                    "dataset": "PNNL (historical)",
                    "region": str(rname),
                    "year": int(yr),
                    "mm": float(mm),
                })

        dsp.close()

    mask_ds.close()
else:
    if PNNL_HIST.exists():
        print("[PNNL] Skipping regional PNNL: missing region mask file:", PNNL_REGION_MASK)

# ==========================================================
# CONUS404 — WY totals
# ==========================================================
if USE_CONUS404:
    try:
        import intake
        import shapely
    except ImportError as e:
        print(f"[CONUS404] Missing dependency ({e}), skipping CONUS404.")
    else:
        print("[CONUS404] Loading from HyTEST (conus404-daily-osn)...")
        url = "https://raw.githubusercontent.com/hytest-org/hytest/main/dataset_catalog/hytest_intake_catalog.yml"
        cat = intake.open_catalog(url)
        ds = cat["conus404-catalog"]["conus404-daily-osn"].to_dask()

        var = "PREC_ACC_NC"
        if var in ds.data_vars:
            precip = ds[var].sel(time=slice(GLOBAL_START, GLOBAL_END))

            lat = ds["lat"]
            lon = ds["lon"]

            regs = regions_gdf.to_crs("EPSG:4326")

            for region in REGION_NAMES:
                geom = regs.loc[regs["Name"] == region, "geometry"].iloc[0]
                mask_np = shapely.contains_xy(geom, lon.values, lat.values)
                mask_da = xr.DataArray(mask_np, dims=("y", "x"))

                region_daily = precip.where(mask_da).mean(dim=("y", "x"))
                ann = water_year_sum_mm(region_daily)

                if ann is None or "year" not in ann.dims:
                    continue

                for yr in years_for_water_year():
                    if yr not in ann["year"].values:
                        continue
                    val = float(ann.sel(year=yr).values)
                    if np.isfinite(val):
                        rows.append({
                            "dataset": "CONUS404 (daily-osn)",
                            "region": region,
                            "year": int(yr),
                            "mm": float(val),
                        })
else:
    print("[CONUS404] Skipped (USE_CONUS404=False)")

# ==========================================================
# PRISM (local zarr) — WY totals
# ==========================================================
PRISM_DATASET_NAME = "PRISM"

def _detect_prism_var(ds):
    candidates = ["ppt", "precip", "precipitation", "pr", "tp"]
    for v in candidates:
        if v in ds.data_vars:
            return v
    for v in ds.data_vars:
        if ("ppt" in v.lower()) or ("prec" in v.lower()):
            return v
    return None

prism_zarrs = []
if P_PRISM_DIR.exists():
    prism_zarrs = sorted([p for p in P_PRISM_DIR.glob("*.zarr") if p.is_dir()])

if not prism_zarrs:
    print(f"[PRISM] No PRISM zarrs found under {P_PRISM_DIR} -> PRISM skipped.")
else:
    print(f"[PRISM] Found {len(prism_zarrs)} zarr(s). Opening and ingesting...")

    prism_dsets = []
    for p in prism_zarrs:
        try:
            prism_dsets.append(safe_open_zarr(p))
        except Exception as e:
            print(f"[PRISM] Failed to open {p.name}: {e}")

    if not prism_dsets:
        print("[PRISM] No PRISM datasets could be opened -> PRISM skipped.")
    else:
        v = _detect_prism_var(prism_dsets[0])
        if v is None:
            for ds_try in prism_dsets[1:]:
                v = _detect_prism_var(ds_try)
                if v is not None:
                    break

        if v is None:
            print(f"[PRISM] Could not detect precip variable. data_vars={list(prism_dsets[0].data_vars)}")
        else:
            da_list = []
            for ds_pr in prism_dsets:
                if v not in ds_pr:
                    continue
                da = ds_pr[v]
                if "time" not in da.dims:
                    tdim = next((d for d in da.dims if ("time" in d.lower()) or ("day" in d.lower())), None)
                    if tdim:
                        da = da.rename({tdim: "time"})
                if "time" in da.dims:
                    da_list.append(da)

            if not da_list:
                print("[PRISM] No usable PRISM precip arrays (missing time?) -> PRISM skipped.")
            else:
                prism = xr.concat(da_list, dim="time", join="outer", coords="minimal", compat="override")
                prism = ensure_time_sorted_unique(prism)
                prism = prism.sel(time=slice(GLOBAL_START, GLOBAL_END))

                prism_daily = region_mean(prism, grid_ds=prism.to_dataset(name="ppt"), regions_gdf=regions_gdf)
                prism_daily = ensure_time_sorted_unique(prism_daily)

                ann = water_year_sum_mm(prism_daily)
                if ann is not None and "year" in ann.dims:
                    for yr in years_for_water_year():
                        if yr not in ann["year"].values:
                            continue
                        for rname in ann["region"].values:
                            val = float(ann.sel(year=yr, region=rname).values)
                            if np.isfinite(val):
                                rows.append({
                                    "dataset": PRISM_DATASET_NAME,
                                    "region": str(rname),
                                    "year": int(yr),
                                    "mm": float(val),
                                })

                print("[PRISM] Ingest complete.")

    for ds_pr in prism_dsets:
        try:
            ds_pr.close()
        except Exception:
            pass

# ==========================================================
# RESULTS
# ==========================================================
df = (
    pd.DataFrame(rows)
      .groupby(["dataset", "region", "year"], as_index=False)
      .agg(mm=("mm", "sum"))
      .sort_values(["region", "dataset", "year"])
      .reset_index(drop=True)
)

df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
df.to_csv(CSV_OUT, index=False)
print("Saved CSV →", CSV_OUT)

# ==========================================================
# PLOT EXCLUSIONS (plot-only; data stays in CSV)
# ==========================================================
PLOT_EXCLUSIONS = {
    "CONUS404 (daily-osn)": {2022},
    "Daymet v4 (Planetary Computer)": {2021},
    "PRISM": {1982},
}

df_plot = df.copy()

def _is_excluded(row) -> bool:
    ds = str(row["dataset"])
    yr = int(row["year"])
    return yr in PLOT_EXCLUSIONS.get(ds, set())

df_plot.loc[df_plot.apply(_is_excluded, axis=1), "mm"] = np.nan
df_plot.loc[df_plot["mm"] == 0, "mm"] = np.nan

# ==========================================================
# PLOTS: one figure per region
# ==========================================================
for region in REGION_NAMES:
    df_r = df_plot[df_plot["region"] == region].copy()
    if df_r.empty:
        continue

    out_png = OUT / f"wy_total_precip_{region.replace(' ','_')}_{YEAR_MIN}_{YEAR_MAX}_FULL_TIMELINE.png"
    plot_region_water_year_full_timeline(
        df_region=df_r,
        value_col="mm",
        ylabel="Precipitation (mm)",
        title_prefix=f"{region} — Water-Year Total Precipitation ({YEAR_MIN}–{YEAR_MAX})",
        out_path=out_png,
    )