"""Daily precipitation of every product on the PRISM day (shared by analysis and plots).

The PRISM day D runs from 12 UTC on D-1 to 12 UTC on D (04 PST to 04 PST) and is
labelled D. The products label their days differently, so matching them by date label
alone puts them up to ~20 h out of step with PRISM (GitHub issue #61). This module
returns each product's daily totals already on the PRISM day, so that the bias table
(analysis/3_bulk_bias_new_version.py) and the figures use the same days.

How each product is put on the PRISM day (align=True):
    prism     as is.
    daymet    local midnight-to-midnight day d overlaps PRISM day d+1 for 20 h:
              relabelled d+1.
    gridmet   07-07 UTC day d overlaps PRISM day d+1 for 19 h: relabelled d+1.
    pnnl      hourly PREC_ACC_NC, assumed UTC and holding the hour ending at the stamp
              (the files do not say): day D = stamps D-1 13:00 .. D 12:00.
    conus     CONUS404 hourly PREC_ACC_NC ("accumulated over prior 60 minutes", UTC),
              clipped files from downloaders/conus404_hourly_downloader.py:
              day D = stamps D-1 13:00 .. D 12:00.
    ucla      RAINC + RAINNC running totals at 12 UTC from
              downloaders/ucla_era5_12utc_accum_download.py: day D = total(D 12 UTC) -
              total(D-1 12 UTC) within one water-year run; the day labelled Sep 1 spans
              two runs (old run Sep 1 00 - Aug 31 12, plus new run Sep 1 12 - Sep 1 00).
With align=False the products' own date labels are used (daily files for CONUS404 and
UCLA, PNNL stamps 00..23 of D), to reproduce the earlier, unaligned results.

Missing data are never filled: a day that is not complete in the source (missing file,
missing hour, NaN snapshot) is left out of the result.

Main functions
    load_daily(product, dates, align=True, crop="basin")
        -> DataArray (time, y, x) of daily totals [mm] for the requested PRISM-day labels
           that are available, float64, with coordinates lon/lat (1D for the rectilinear
           PRISM and gridMET grids, 2D otherwise).
    basin_mask(da, regions) HUC8 union of REGION_NAMES (or of the given sub-basins) on
                            the grid of da (bool, y x).
    boundary_mask(da)       cells inside the watershed outline SkagitBoundary.json (maps,
                            hourly plots).
    basin_mean(da, mask)    unweighted mean over the mask cells -> pandas Series by time.
    load_hourly(product, start, end)
        -> hourly precipitation [mm] of pnnl / conus / ucla, each value being the hour
           ending at its UTC stamp (used by the hourly storm plots).

Grid orientation note: the PNNL files are (time, x, y) with x = south_north and
y = west_east, the opposite of what the names suggest; here they are returned as (y, x)
like every other product.
"""

import os
import tempfile
import warnings
import zipfile
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import rioxarray  # noqa: F401  (registers the rasterio backend used for PRISM)
import xarray as xr

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # skagit_met_for_publication/
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")

VAULT_DIR = "/data0/skagit_met/data_transfer/data"
PRISM_DIR = os.path.join(VAULT_DIR, "prism_new_hq")                    # <decade>/prism_ppt_us_25m_YYYYMMDD.zip
DAYMET_DIR = os.path.join(VAULT_DIR, "daymet_new_hq")                  # prcp_YYYY_subset.nc
GRIDMET_DIR = os.path.join(VAULT_DIR, "gridmet")                       # YYYY_daily_4km_gridMET_data.zarr
PNNL_DIR = os.path.join(VAULT_DIR, "PNNL/historical")                  # YYYY/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.YYYY.nc
PNNL_GEO_FILE = os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc")
CONUS404_HOURLY_DIR = os.path.join(VAULT_DIR, "conus404_hourly")      # conus404_skagit_prec_hourly_YYYY.nc
CONUS404_DAILY_ZARR = os.path.join(VAULT_DIR, "conus404_daily/conus404_skagit_precip_daily_full.zarr")
UCLA_12UTC_DIR = os.path.join(VAULT_DIR, "ucla_era5_d02_12utc_accum")  # prec_accum_12utc.era5.d02.<WY>.nc
UCLA_DAILY_DIR = os.path.join(VAULT_DIR, "ucla_era5_d02_daily")       # prec/prec.daily.era5.d02.<WY>.nc
UCLA_GEO_FILE = os.path.join(UCLA_DAILY_DIR, "static/wrfinput_d02_coord.nc")
# Full-domain UCLA hourly files, only for the hourly-plot event windows (auxhist_d01_<YYYY-MM-DD_HH>:00:00.nc)
UCLA_HOURLY_EVENTS_DIR = os.path.join(VAULT_DIR, "ucla_era5_d02_hourly/hourly_ar_non_ar_events")
BOUNDARY_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitBoundary.json")  # watershed outline (maps, hourly plots)

# ============================================================
# SETTINGS
# ============================================================
PRODUCTS = ("prism", "daymet", "gridmet", "pnnl", "conus", "ucla")
REGION_NAMES = ["Upper Skagit", "Sauk", "Lower Skagit"]  # HUC8 sub-basins forming the basin
CROP_BUFFER = 2          # grid cells kept around the basin when crop="basin"
PNNL_LAST_YEAR = 2020    # PNNL historical run ends in 2020
ONE_DAY = pd.Timedelta(days=1)


# ============================================================
# Masks and basin means
# ============================================================
@lru_cache(maxsize=None)
def _basin_regions(regions=tuple(REGION_NAMES)):
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    idx = [gdf[gdf["Name"].str.contains(r, case=False)].index[0] for r in regions]
    return gdf.loc[idx]


def _mask_from_lonlat(lon, lat, regions=tuple(REGION_NAMES)):
    """Boolean (y, x) mask of the union of ``regions`` (HUC8 names) for 1D or 2D lon/lat."""
    m = regionmask.mask_3D_geopandas(_basin_regions(tuple(regions)), lon, lat).any(dim="region")
    return np.asarray(m.values, dtype=bool)


def basin_mask(da, regions=REGION_NAMES):
    """Mask of HUC8 sub-basins on the grid of ``da`` (as returned by load_daily).

    ``regions`` defaults to the union of REGION_NAMES (the whole basin); pass e.g.
    ["Sauk"] for a single sub-basin.
    """
    return _mask_from_lonlat(da["lon"].values, da["lat"].values, tuple(regions))


@lru_cache(maxsize=1)
def _boundary_polygon():
    return gpd.read_file(BOUNDARY_GEO).to_crs("EPSG:4326").union_all()


def boundary_mask(da):
    """Cells of ``da``'s grid whose centre is in the watershed outline (SkagitBoundary.json).

    This is the mask of the maps and hourly plots; the bias table uses basin_mask
    (HUC8 sub-basins) instead.
    """
    lon, lat = da["lon"].values, da["lat"].values
    if np.ndim(lon) == 1:
        lon, lat = np.meshgrid(lon, lat)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon.ravel(), lat.ravel()), crs="EPSG:4326")
    return pts.intersects(_boundary_polygon()).values.reshape(lon.shape)


def basin_mean(da, mask):
    """Unweighted mean over the cells where ``mask`` is True, NaNs ignored.

    Returns a pandas Series indexed by time (NaN where no cell has data).
    """
    vals = da.values.reshape(da.sizes["time"], -1)[:, np.asarray(mask).ravel()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN day -> NaN
        means = np.nanmean(vals, axis=1) if vals.shape[1] else np.full(len(vals), np.nan)
    return pd.Series(means, index=pd.DatetimeIndex(da["time"].values))


def _box(mask, buffer=CROP_BUFFER):
    """(row slice, col slice) of the mask's bounding box plus ``buffer`` cells."""
    rows, cols = np.where(mask)
    ny, nx = mask.shape
    return (slice(max(rows.min() - buffer, 0), min(rows.max() + buffer + 1, ny)),
            slice(max(cols.min() - buffer, 0), min(cols.max() + buffer + 1, nx)))


def _crop_box(lon, lat, crop):
    """(row slice, col slice) selecting the requested crop on a 1D or 2D lon/lat grid.

    crop = "basin"                      basin bounding box + CROP_BUFFER cells
           None                         full grid
           (lon_min, lon_max, lat_min, lat_max)
                                        rows/cols of all cells whose centre is in the box
    """
    if crop is None:
        return slice(None), slice(None)
    if isinstance(crop, str):
        if crop != "basin":
            raise ValueError(f"unknown crop {crop!r}")
        return _box(_mask_from_lonlat(lon, lat))
    lon2d, lat2d = np.meshgrid(lon, lat) if np.ndim(lon) == 1 else (lon, lat)
    inside = (lon2d >= crop[0]) & (lon2d <= crop[1]) & (lat2d >= crop[2]) & (lat2d <= crop[3])
    return _box(inside, buffer=0)


def _daily_array(values, times, lon, lat):
    """Standard output: DataArray (time, y, x) float64 with lon/lat coords (1D or 2D)."""
    values = np.asarray(values, dtype="float64")
    coords = {"time": pd.DatetimeIndex(times)}
    if np.ndim(lon) == 1:
        coords["lon"] = ("x", np.asarray(lon))
        coords["lat"] = ("y", np.asarray(lat))
    else:
        coords["lon"] = (("y", "x"), np.asarray(lon))
        coords["lat"] = (("y", "x"), np.asarray(lat))
    return xr.DataArray(values, dims=("time", "y", "x"), coords=coords, name="precip")


def _norm_dates(dates):
    return pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates).normalize())))


# ============================================================
# PRISM
# ============================================================
def get_decade_folder(year: int) -> str:
    """PRISM sub-folder holding a given year (e.g. 1981-1990, 2021-2024)."""
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


def load_prism_day(date):
    """One day of PRISM (zipped GeoTIFF) as a 2D DataArray (y, x) with 1D x/y coords.

    Negative values (no data, -9999) are set to NaN. Returns None if missing/unreadable.
    """
    date = pd.Timestamp(date)
    zip_path = os.path.join(PRISM_DIR, get_decade_folder(date.year), f"prism_ppt_us_25m_{date:%Y%m%d}.zip")
    if not os.path.exists(zip_path):
        return None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmpdir)
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".tif"):
                        da = rioxarray.open_rasterio(os.path.join(root, f)).squeeze().load()
                        da = da.astype("float64")
                        return da.where(da >= 0)
    except Exception as e:
        print(f"  Error loading PRISM for {date:%Y%m%d}: {e}")
    return None


def _load_prism(dates, crop):
    box, out, lon, lat = None, [], None, None
    for d in dates:
        da = load_prism_day(d)
        if da is None:
            continue
        if box is None:
            box = _crop_box(da.x.values, da.y.values, crop)
            lon, lat = da.x.values[box[1]], da.y.values[box[0]]
        out.append((d, da.values[box]))
    if not out:
        return None
    return _daily_array(np.stack([v for _, v in out]), [d for d, _ in out], lon, lat)


# ============================================================
# Daymet and gridMET (daily files, relabelled d -> d+1 when aligning)
# ============================================================
def _load_daily_files(dates, align, open_year, var_of, lonlat_of, crop):
    """Shared loader for products with one daily file per calendar year."""
    shift = ONE_DAY if align else pd.Timedelta(0)
    src_dates = dates - shift
    pieces, box, lon, lat = [], None, None, None
    for year in sorted(set(src_dates.year)):
        ds = open_year(year)
        if ds is None:
            continue
        with ds:
            da = var_of(ds)
            if box is None:
                lo, la = lonlat_of(ds)
                box = _crop_box(lo, la, crop)
                lon = lo[box[1]] if np.ndim(lo) == 1 else lo[box]
                lat = la[box[0]] if np.ndim(la) == 1 else la[box]
            t = pd.DatetimeIndex(da["time"].values).normalize()
            keep = ~t.duplicated() & t.isin(src_dates)
            if keep.any():
                sub = da.isel(time=np.where(keep)[0])
                vals = sub.values[:, box[0], box[1]]
                pieces.append((t[keep], vals))
    if not pieces:
        return None
    times = np.concatenate([p[0] for p in pieces])
    vals = np.concatenate([p[1] for p in pieces])
    order = np.argsort(times)
    return _daily_array(vals[order], pd.DatetimeIndex(times[order]) + shift, lon, lat)


def _open_daymet(year):
    p = os.path.join(DAYMET_DIR, f"prcp_{year}_subset.nc")
    return xr.open_dataset(p) if os.path.exists(p) else None


def _open_gridmet(year):
    p = os.path.join(GRIDMET_DIR, f"{year}_daily_4km_gridMET_data.zarr")
    if not os.path.exists(p):
        return None
    ds = xr.open_zarr(p)
    return ds.rename({"day": "time"}) if "day" in ds.dims else ds


def _load_daymet(dates, align, crop):
    return _load_daily_files(dates, align, _open_daymet, lambda ds: ds["prcp"],
                             lambda ds: (ds["lon"].values, ds["lat"].values), crop)


def _load_gridmet(dates, align, crop):
    var = lambda ds: ds["prcp"] if "prcp" in ds.data_vars else ds["precipitation_amount"]
    return _load_daily_files(dates, align, _open_gridmet, var,
                             lambda ds: (ds["lon"].values, ds["lat"].values), crop)


# ============================================================
# Hourly products summed to days (PNNL, CONUS404)
# ============================================================
def _hour_stamps(dates, align):
    """Hourly stamps making up each requested day, and the day label of each stamp."""
    first = -11 if align else 0   # aligned: D-1 13:00 .. D 12:00 ; else D 00:00 .. D 23:00
    offs = pd.to_timedelta(np.arange(first, first + 24), unit="h")
    stamps = pd.DatetimeIndex([d + o for d in dates for o in offs])
    return stamps


def _label_of(stamps, align):
    if align:
        return (stamps - pd.Timedelta(hours=13)).normalize() + ONE_DAY
    return stamps.normalize()


def _sum_hours(hourly_vals, stamps, align):
    """Sum hourly grids into days; keep only days with all 24 hours."""
    labels = _label_of(stamps, align)
    days, inv, counts = np.unique(labels.values, return_inverse=True, return_counts=True)
    out = np.zeros((len(days),) + hourly_vals.shape[1:], dtype="float64")
    np.add.at(out, inv, hourly_vals.astype("float64"))
    nan_hour = np.isnan(hourly_vals).all(axis=(1, 2))
    bad = np.zeros(len(days), dtype=bool)
    np.logical_or.at(bad, inv, nan_hour)
    full = (counts == 24) & ~bad
    return pd.DatetimeIndex(days[full]), out[full]


@lru_cache(maxsize=1)
def _pnnl_grid():
    with xr.open_dataset(PNNL_GEO_FILE) as g:
        return g["XLONG_M"].values[0], g["XLAT_M"].values[0]


def _load_pnnl(dates, align, crop):
    lon2d, lat2d = _pnnl_grid()
    box = _crop_box(lon2d, lat2d, crop)
    want = _hour_stamps(dates, align)
    vals, stamps = [], []
    for year in sorted(set(want.year)):
        if year > PNNL_LAST_YEAR:
            continue
        path = os.path.join(PNNL_DIR, str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path) as ds:
            t = pd.DatetimeIndex(ds["time"].values)
            keep = t.isin(want) & ~t.duplicated()
            if keep.any():
                # file dims (time, x, y): x = south_north (rows), y = west_east (cols)
                sub = ds["PREC_ACC_NC"].isel(time=np.where(keep)[0], x=box[0], y=box[1])
                vals.append(sub.values)
                stamps.append(t[keep])
    if not vals:
        return None
    days, daily = _sum_hours(np.concatenate(vals), pd.DatetimeIndex(np.concatenate(stamps)), align)
    return _daily_array(daily, days, lon2d[box], lat2d[box])


def _load_conus(dates, align, crop):
    if not align:
        return _load_conus_daily_file(dates, crop)
    want = _hour_stamps(dates, align)
    vals, stamps, lon, lat, box = [], [], None, None, None
    for year in sorted(set(want.year)):
        path = os.path.join(CONUS404_HOURLY_DIR, f"conus404_skagit_prec_hourly_{year}.nc")
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path) as ds:
            if box is None:
                lo, la = ds["lon"].values, ds["lat"].values
                box = _crop_box(lo, la, crop)
                lon, lat = lo[box], la[box]
            t = pd.DatetimeIndex(ds["time"].values)
            keep = t.isin(want) & ~t.duplicated()
            if keep.any():
                vals.append(ds["PREC_ACC_NC"].isel(time=np.where(keep)[0], y=box[0], x=box[1]).values)
                stamps.append(t[keep])
    if not vals:
        return None
    days, daily = _sum_hours(np.concatenate(vals), pd.DatetimeIndex(np.concatenate(stamps)), align)
    return _daily_array(daily, days, lon, lat)


def _load_conus_daily_file(dates, crop):
    """CONUS404 daily product (UTC days), date labels as stored."""
    if not os.path.exists(CONUS404_DAILY_ZARR):
        return None
    with xr.open_zarr(CONUS404_DAILY_ZARR) as ds:
        lo, la = ds["lon"].values, ds["lat"].values
        box = _crop_box(lo, la, crop)
        t = pd.DatetimeIndex(ds["time"].values).normalize()
        keep = t.isin(dates) & ~t.duplicated()
        if not keep.any():
            return None
        vals = ds["precip_daily"].isel(time=np.where(keep)[0], y=box[0], x=box[1]).values
        return _daily_array(vals, t[keep], lo[box], la[box])


# ============================================================
# UCLA
# ============================================================
def _load_ucla(dates, align, crop):
    return _load_ucla_12utc(dates, crop) if align else _load_ucla_daily_file(dates, crop)


def _water_year_runs(dates):
    """Water-year runs (Sep 1 WY .. Sep 1 WY+1) holding the 12 UTC ends of the requested days."""
    runs = set()
    for d in dates:
        for t in (d, d - ONE_DAY):          # D 12 UTC and D-1 12 UTC
            runs.add(t.year if t.month >= 9 else t.year - 1)
    return sorted(runs)


def _load_ucla_12utc(dates, crop):
    within, first_part, last_part = {}, {}, {}
    lon = lat = box = None
    for wy in _water_year_runs(dates):
        path = os.path.join(UCLA_12UTC_DIR, f"prec_accum_12utc.era5.d02.{wy}.nc")
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path) as ds:
            if box is None:
                lo, la = ds["lon2d"].values, ds["lat2d"].values
                box = _crop_box(lo, la, crop)
                lon, lat = lo[box], la[box]
            acc = (ds["RAINC"].astype("float64") + ds["RAINNC"].astype("float64")).values[:, box[0], box[1]]
            t = pd.DatetimeIndex(ds["time"].values)
        start, end = pd.Timestamp(f"{wy}-09-01"), pd.Timestamp(f"{wy + 1}-09-01")
        at = {tt: acc[i] for i, tt in enumerate(t)}
        noon = sorted(tt for tt in t if tt.hour == 12)
        for a, b in zip(noon[:-1], noon[1:]):          # consecutive 12 UTC values
            if b - a == ONE_DAY:
                within[b.normalize()] = at[b] - at[a]
        if start + pd.Timedelta(hours=12) in at and start in at:
            first_part[start] = at[start + pd.Timedelta(hours=12)] - at[start]
        if end in at and end - pd.Timedelta(hours=12) in at:
            last_part[end] = at[end] - at[end - pd.Timedelta(hours=12)]
    for d in set(first_part) & set(last_part):          # day labelled Sep 1 spans two runs
        within[d] = last_part[d] + first_part[d]
    days = [d for d in dates if d in within and not np.isnan(within[d]).all()]
    if not days:
        return None
    return _daily_array(np.stack([within[d] for d in days]), days, lon, lat)


def _load_ucla_daily_file(dates, crop):
    """UCLA daily product (water-year files, UTC days), date labels as stored."""
    with xr.open_dataset(UCLA_GEO_FILE) as g:
        lo, la = g["lon2d"].values, g["lat2d"].values
    box = _crop_box(lo, la, crop)
    pieces = []
    for wy in sorted({d.year if d.month >= 9 else d.year - 1 for d in dates}):
        path = os.path.join(UCLA_DAILY_DIR, "prec", f"prec.daily.era5.d02.{wy}.nc")
        if not os.path.exists(path):
            continue
        with xr.open_dataset(path) as ds:
            if "day" in ds.dims:
                ds = ds.rename({"day": "time"})
            var = "prec" if "prec" in ds.data_vars else "pr"
            t = pd.DatetimeIndex(ds["time"].values).normalize()
            keep = t.isin(dates) & ~t.duplicated()
            if keep.any():
                da = ds[var].isel(time=np.where(keep)[0])
                pieces.append((t[keep], da.values[:, box[0], box[1]]))
    if not pieces:
        return None
    times = np.concatenate([p[0] for p in pieces])
    vals = np.concatenate([p[1] for p in pieces])
    order = np.argsort(times)
    times, idx = np.unique(times[order], return_index=True)
    return _daily_array(vals[order][idx], pd.DatetimeIndex(times), lo[box], la[box])


# ============================================================
# Hourly data (hourly plots)
# ============================================================
def _hourly_array(vals, stamps, lon, lat):
    return _daily_array(vals, stamps, lon, lat).rename("precip_hourly")


def load_hourly(product, start, end, crop="basin"):
    """Hourly precipitation [mm] of ``product`` for stamps ``start`` .. ``end`` (UTC, inclusive).

    Each value is the precipitation of the hour ENDING at its stamp, for every product:
        pnnl   PREC_ACC_NC as stored (assumed UTC, end of hour)
        conus  CONUS404 PREC_ACC_NC from the clipped hourly files ("prior 60 minutes")
        ucla   change of RAINC + RAINNC since the previous hour, from the full-domain hourly
               files in UCLA_HOURLY_EVENTS_DIR (only the hourly-plot event windows exist there;
               the file one hour before ``start`` is needed too)
    Missing hours are left out. Returns DataArray (time, y, x) float64 with lon/lat
    coordinates, or None if no hour is available.
    """
    stamps = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h")
    if product == "ucla":
        return _load_ucla_hourly(stamps, crop)
    if product not in ("pnnl", "conus"):
        raise ValueError(f"no hourly data for {product!r}; expected 'pnnl', 'conus' or 'ucla'")
    vals, got, lon, lat = [], [], None, None
    for year in sorted(set(stamps.year)):
        if product == "pnnl":
            path = os.path.join(PNNL_DIR, str(year), f"PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
        else:
            path = os.path.join(CONUS404_HOURLY_DIR, f"conus404_skagit_prec_hourly_{year}.nc")
        if not os.path.exists(path) or (product == "pnnl" and year > PNNL_LAST_YEAR):
            continue
        with xr.open_dataset(path) as ds:
            if product == "pnnl":
                lo, la = _pnnl_grid()
            else:
                lo, la = ds["lon"].values, ds["lat"].values
            box = _crop_box(lo, la, crop)
            lon, lat = lo[box], la[box]
            t = pd.DatetimeIndex(ds["time"].values)
            keep = t.isin(stamps) & ~t.duplicated()
            if not keep.any():
                continue
            if product == "pnnl":   # file dims (time, x, y): x = south_north (rows), y = west_east (cols)
                sub = ds["PREC_ACC_NC"].isel(time=np.where(keep)[0], x=box[0], y=box[1])
            else:
                sub = ds["PREC_ACC_NC"].isel(time=np.where(keep)[0], y=box[0], x=box[1])
            vals.append(sub.values)
            got.append(t[keep])
    if not vals:
        return None
    return _hourly_array(np.concatenate(vals), np.concatenate(got), lon, lat)


def _load_ucla_hourly(stamps, crop):
    with xr.open_dataset(UCLA_GEO_FILE) as g:
        lo, la = g["lon2d"].values, g["lat2d"].values
    box = _crop_box(lo, la, crop)
    acc = {}
    for t in stamps.insert(0, stamps[0] - pd.Timedelta(hours=1)):
        path = os.path.join(UCLA_HOURLY_EVENTS_DIR, f"auxhist_d01_{t:%Y-%m-%d_%H}:00:00.nc")
        if os.path.exists(path):
            with xr.open_dataset(path) as ds:
                a = ds["RAINC"].isel(Time=0).astype("float64") + ds["RAINNC"].isel(Time=0).astype("float64")
                acc[t] = a.values[box]
    got = [t for t in stamps if t in acc and t - pd.Timedelta(hours=1) in acc]
    if not got:
        return None
    vals = np.stack([acc[t] - acc[t - pd.Timedelta(hours=1)] for t in got])
    return _hourly_array(vals, got, lo[box], la[box])


# ============================================================
# Public entry point
# ============================================================
_LOADERS = {
    "prism": lambda dates, align, crop: _load_prism(dates, crop),
    "daymet": _load_daymet,
    "gridmet": _load_gridmet,
    "pnnl": _load_pnnl,
    "conus": _load_conus,
    "ucla": _load_ucla,
}


def load_daily(product, dates, align=True, crop="basin"):
    """Daily precipitation [mm] of ``product`` for the requested day labels.

    Parameters
    ----------
    product : one of PRODUCTS ("prism", "daymet", "gridmet", "pnnl", "conus", "ucla").
    dates   : iterable of dates (day labels D; time of day is ignored).
    align   : True  -> PRISM day (12 UTC on D-1 to 12 UTC on D), see module docstring.
              False -> the product's own date labels (earlier, unaligned behaviour).
    crop    : "basin" -> bounding box of the REGION_NAMES basin + CROP_BUFFER cells;
              None    -> the full grid of the (possibly pre-clipped) source;
              (lon_min, lon_max, lat_min, lat_max) -> cells whose centre is in that box
              (e.g. a map's domain; limited to what pre-clipped sources contain).

    Returns
    -------
    xarray.DataArray (time, y, x), float64, only for the requested days that are
    complete in the source (sorted by time), or None if none are.
    """
    if product not in _LOADERS:
        raise ValueError(f"unknown product {product!r}; expected one of {PRODUCTS}")
    dates = _norm_dates(dates)
    if len(dates) == 0:
        return None
    da = _LOADERS[product](dates, align, crop)
    if da is None or da.sizes["time"] == 0:
        return None
    return da.sortby("time")
