#!/usr/bin/env python3
"""Download CONUS404 hourly precipitation, clipped to the Skagit, one netCDF per calendar year.

Purpose: daily totals on the PRISM day (12 UTC on D-1 to 12 UTC on D, labelled D;
GitHub issue #61). The CONUS404 daily product uses a different day boundary and cannot
simply be shifted, so daily totals are rebuilt from the hourly data:
    PRISM day D = sum of hourly PREC_ACC_NC stamped D-1 13:00 ... D 12:00 UTC.

Source (public, no login): CONUS404 hourly, USGS HyTEST, on the Open Storage Network
    https://usgs.osn.mghpcc.org/hytest/conus404/conus404_hourly.zarr
    PREC_ACC_NC: "accumulated over prior 60 minutes" (mm), i.e. each value is the
    hour ending at its time stamp. Times are UTC (WRF model time). Data end 2024-09-30 23:00.
    The zarr is chunked (144 h, 175, 175); the Skagit sub-basins fall inside a single
    spatial chunk, so only that chunk column is read.

Clipping: bounding box of the HUC8 sub-basins in REGION_NAMES (Upper Skagit, Sauk,
Lower Skagit) on the CONUS404 grid, plus BUFFER cells on each side. The analysis applies
the sub-basin mask itself.

Output:
    OUTPUT_DIR/conus404_skagit_prec_hourly_<YYYY>.nc
        PREC_ACC_NC (time, y, x)  precipitation in the hour ending at time (mm), UTC
        lat, lon    (y, x)

Usage:
    python conus404_hourly_downloader.py [--start 1979] [--end 2024]

    1979 holds only 1979-10-01 onward (start of CONUS404); its Dec 31 hours are needed
    for the day labelled 1980-01-01. Existing years are skipped; each year is written to
    a .part file and renamed when complete, so re-running resumes.
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import regionmask
import xarray as xr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # skagit_met_for_publication/
HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
REGION_NAMES = ["Upper Skagit", "Sauk", "Lower Skagit"]
BUFFER = 2  # grid cells added on each side of the sub-basin bounding box

SOURCE_URL = "https://usgs.osn.mghpcc.org/hytest/conus404/conus404_hourly.zarr"
PRECIP_VAR = "PREC_ACC_NC"
VAULT_DIR = "/data0/skagit_met/data_transfer/data"
OUTPUT_DIR = os.path.join(VAULT_DIR, "conus404_hourly")


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


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=int, default=1979, help="first calendar year (default 1979)")
    p.add_argument("--end", type=int, default=2024, help="last calendar year (default 2024)")
    a = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds = xr.open_dataset(SOURCE_URL, engine="zarr", consolidated=True, chunks={})
    integ = ds[PRECIP_VAR].attrs.get("integration_length", "")
    if "prior 60 minutes" not in integ:
        raise ValueError(f"Unexpected {PRECIP_VAR} integration_length: {integ!r}")
    lon2d, lat2d = ds["lon"].values, ds["lat"].values
    ys, xs = basin_box(lon2d, lat2d)
    clip = ds[PRECIP_VAR].isel(y=ys, x=xs)

    failed = []
    for year in range(a.start, a.end + 1):
        out = os.path.join(OUTPUT_DIR, f"conus404_skagit_prec_hourly_{year}.nc")
        if os.path.exists(out):
            print(f"[{year}] exists, skipping")
            continue
        try:
            da = clip.sel(time=slice(f"{year}-01-01", f"{year}-12-31T23")).load()
            if da.sizes["time"] == 0:
                print(f"[{year}] no data in source, skipping")
                continue
            hours = da.sizes["time"]
            out_ds = xr.Dataset(
                {PRECIP_VAR: da.astype("float32").drop_vars(["lat", "lon"], errors="ignore")},
                coords={"lat": (("y", "x"), lat2d[ys, xs]), "lon": (("y", "x"), lon2d[ys, xs])},
                attrs={"source": SOURCE_URL,
                       "note": "PREC_ACC_NC = precipitation in the hour ending at time (UTC), mm. "
                               "PRISM day D = sum of D-1 13:00 .. D 12:00.",
                       "clip": f"rows {ys.start}:{ys.stop}, cols {xs.start}:{xs.stop} "
                               f"(HUC8 {REGION_NAMES} box + {BUFFER} cells)"},
            )
            out_ds[PRECIP_VAR].attrs = {k: v for k, v in ds[PRECIP_VAR].attrs.items() if k != "grid_mapping"}
            out_ds.to_netcdf(out + ".part", encoding={PRECIP_VAR: {"zlib": True, "complevel": 4}})
            os.replace(out + ".part", out)
            print(f"[{year}] saved {hours} h, NaN hours: {int(np.isnan(da.values).all(axis=(1, 2)).sum())}", flush=True)
        except Exception as e:
            print(f"[{year}] FAILED: {e}  (re-run to retry)", flush=True)
            failed.append(year)
            if os.path.exists(out + ".part"):
                os.remove(out + ".part")
    print("Done." + (f" Failed years: {failed}" if failed else " All years complete."))


if __name__ == "__main__":
    main()
