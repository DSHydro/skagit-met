#!/usr/bin/env python3
"""Download UCLA ERA5-WRF d02 accumulated precipitation at 12 UTC each day, clipped to the Skagit.

Purpose: daily totals on the PRISM day (12 UTC on D-1 to 12 UTC on D, labelled D;
GitHub issue #61). The UCLA *daily* files use a different day boundary and cannot
simply be shifted, so daily totals are rebuilt from the hourly WRF output.

Why only 12 UTC: the hourly files store RAINC and RAINNC as running totals since the
start of each run (not hourly amounts). The precipitation of PRISM day D is therefore
    (RAINC + RAINNC)(D 12 UTC) - (RAINC + RAINNC)(D-1 12 UTC)
so one file per day gives exact 12-12 UTC totals; the other 23 hours add nothing.

Runs: hourly/<YYYY>/ holds one WRF run per water year, Sep 1 YYYY 00:00 to Sep 1 YYYY+1
00:00 (the boundary hour is in both runs). Each run's counters start at its own spin-up
start, so differences must stay within one run. For each run this script keeps every
12:00 UTC file plus its first (Sep 1 00:00) and last (Sep 1 00:00 next year) file, so
the day labelled Sep 1 can be built as
    old run: Sep 1 00 - Aug 31 12   +   new run: Sep 1 12 - Sep 1 00.

Data source (public, no login):
    https://wrf-cmip6-noversioning.s3.amazonaws.com/index.html#downscaled_products/reanalysis/era5/hourly/
    file: hourly/<YYYY>/d02/auxhist_d01_<YYYY-MM-DD>_<HH>:00:00  (~4 MB, full d02 domain, UTC)
Reference: https://dept.atmos.ucla.edu/alexhall/downscaling-cmip6

Clipping: bounding box of the HUC8 sub-basins in REGION_NAMES (Upper Skagit, Sauk,
Lower Skagit) on the d02 grid, plus BUFFER cells on each side. The analysis applies
the sub-basin mask itself. Each raw file is deleted right after clipping.

Output (one netCDF per run / water-year folder):
    OUTPUT_DIR/prec_accum_12utc.era5.d02.<YYYY>.nc
        RAINC, RAINNC  (time, south_north, west_east)  running totals since run start, mm
        lat2d, lon2d   (south_north, west_east)
        time = nominal file time (UTC). Attribute 'missing_times' lists files not found
        on the server (stored as NaN).

Usage:
    python ucla_era5_12utc_accum_download.py [--start 1979] [--end 2024] [--workers 8]

    --start/--end are water-year folders. 1979 (Sep 1979 - Aug 1980) is needed for the
    day labelled 1980-01-01. Existing outputs are skipped; each output is written to a
    .part file and renamed when complete, so re-running resumes.
"""
import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import requests
import xarray as xr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # skagit_met_for_publication/
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
REGION_NAMES = ["Upper Skagit", "Sauk", "Lower Skagit"]
BUFFER = 2  # grid cells added on each side of the sub-basin bounding box

VAULT_DIR = "/data0/skagit_met/data_transfer/data"
OUTPUT_DIR = os.path.join(VAULT_DIR, "ucla_era5_d02_12utc_accum")
COORD_FILE = os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc")  # from ucla_era5_daily_download.py

HOURLY_URL = "https://wrf-cmip6-noversioning.s3.amazonaws.com/downscaled_products/reanalysis/era5/hourly"
VARS = ["RAINC", "RAINNC"]


def basin_box(lon2d, lat2d):
    """(y slice, x slice) of the REGION_NAMES bounding box on this grid, plus BUFFER cells."""
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    idx = [gdf[gdf["Name"].str.contains(r, case=False)].index[0] for r in REGION_NAMES]
    mask = regionmask.mask_3D_geopandas(gdf.loc[idx], lon2d, lat2d).any(dim="region").values
    iy, ix = np.where(mask)
    ny, nx = mask.shape
    ys = slice(max(iy.min() - BUFFER, 0), min(iy.max() + BUFFER + 1, ny))
    xs = slice(max(ix.min() - BUFFER, 0), min(ix.max() + BUFFER + 1, nx))
    print(f"Basin mask: {mask.sum()} cells; clip box y {ys.start}-{ys.stop - 1}, x {xs.start}-{xs.stop - 1}")
    return ys, xs


def run_times(wy):
    """Nominal UTC times kept for water-year run wy: Sep 1 00:00, every 12:00, next Sep 1 00:00."""
    start = pd.Timestamp(f"{wy}-09-01 00:00")
    end = pd.Timestamp(f"{wy + 1}-09-01 00:00")
    noons = pd.date_range(start + pd.Timedelta(hours=12), end - pd.Timedelta(hours=12), freq="D")
    return pd.DatetimeIndex([start, *noons, end])


def fetch(session, wy, t, tmpdir):
    """Download one hourly file to tmpdir; return its path, or None if not on the server.

    Runs in worker threads: only does HTTP and file writing (netCDF/HDF5 reading is not
    thread-safe, so files are opened in the main thread by clip()).
    """
    url = f"{HOURLY_URL}/{wy}/d02/auxhist_d01_{t:%Y-%m-%d_%H}:00:00"
    path = os.path.join(tmpdir, f"{wy}_{t:%Y%m%d%H}.nc")
    for attempt in range(4):
        try:
            with session.get(url, timeout=120) as r:
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                with open(path, "wb") as f:
                    f.write(r.content)
            return path
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"{url}: {e}") from e


def clip(path, ys, xs):
    """Read the clipped RAINC/RAINNC from a downloaded hourly file, then delete the file."""
    try:
        with xr.open_dataset(path) as ds:
            return {v: ds[v].isel(Time=0, south_north=ys, west_east=xs).values for v in VARS}
    finally:
        os.remove(path)


def process_run(wy, ys, xs, lat2d, lon2d, workers):
    out = os.path.join(OUTPUT_DIR, f"prec_accum_12utc.era5.d02.{wy}.nc")
    if os.path.exists(out):
        print(f"[{wy}] exists, skipping")
        return
    times = run_times(wy)
    print(f"[{wy}] downloading {len(times)} files ({times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d})", flush=True)
    shape = (len(times), ys.stop - ys.start, xs.stop - xs.start)
    data = {v: np.full(shape, np.nan, dtype="float32") for v in VARS}
    missing = []
    session = requests.Session()
    with tempfile.TemporaryDirectory() as tmpdir, ThreadPoolExecutor(workers) as pool:
        # downloads run in threads; results arrive in order and are clipped here, one at a time
        for i, (t, path) in enumerate(zip(times, pool.map(lambda t: fetch(session, wy, t, tmpdir), times))):
            if path is None:
                missing.append(f"{t:%Y-%m-%d_%H}")
                continue
            res = clip(path, ys, xs)
            for v in VARS:
                data[v][i] = res[v]

    dims = ("time", "south_north", "west_east")
    ds = xr.Dataset(
        {v: (dims, data[v], {"units": "mm", "description": f"{v}: accumulated since start of run (WRF)"}) for v in VARS},
        coords={"time": times,
                "lat2d": (dims[1:], lat2d[ys, xs]), "lon2d": (dims[1:], lon2d[ys, xs])},
        attrs={"source": f"{HOURLY_URL}/{wy}/d02/",
               "run": f"water-year run {wy}: {times[0]} to {times[-1]} UTC",
               "note": "PRISM-day precip for day D = (RAINC+RAINNC)(D 12UTC) - (D-1 12UTC), within one run",
               "clip": f"d02 rows {ys.start}:{ys.stop}, cols {xs.start}:{xs.stop} (HUC8 {REGION_NAMES} box + {BUFFER} cells)",
               "missing_times": ", ".join(missing)},
    )
    enc = {v: {"zlib": True, "complevel": 4} for v in VARS}
    ds.to_netcdf(out + ".part", encoding=enc)
    os.replace(out + ".part", out)
    print(f"[{wy}] saved {out}" + (f"  (missing on server: {len(missing)})" if missing else ""), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=int, default=1979, help="first water-year folder (default 1979)")
    p.add_argument("--end", type=int, default=2024, help="last water-year folder (default 2024)")
    p.add_argument("--workers", type=int, default=8, help="parallel downloads (default 8)")
    a = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with xr.open_dataset(COORD_FILE) as c:
        lat2d, lon2d = c["lat2d"].values, c["lon2d"].values
    ys, xs = basin_box(lon2d, lat2d)
    failed = []
    for wy in range(a.start, a.end + 1):
        try:
            process_run(wy, ys, xs, lat2d, lon2d, a.workers)
        except Exception as e:
            print(f"[{wy}] FAILED: {e}  (re-run to retry)", flush=True)
            failed.append(wy)
    print("Done." + (f" Failed runs: {failed}" if failed else " All runs complete."))


if __name__ == "__main__":
    main()
