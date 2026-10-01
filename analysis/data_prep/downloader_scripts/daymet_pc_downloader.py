"""
This script downloads one year of a selected Daymet variable from the
Planetary Computer, subsets it to a lon/lat bounding box, and writes the
result out as a local NetCDF file.

The main reason this exists is to pull smaller Daymet chunks without having to
work directly against the full remote Zarr every time. It uses a signed SAS
URL and appends the query string as-is, which avoids some of the access issues
I was running into with remote reads.

Output files are written as:
`{out_dir}/daymet_{var}_{year}.nc`

This is meant to be run as a standalone Python script, not from inside a
notebook cell.
"""

import argparse, sys, time
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from urllib.parse import urlparse, urlunparse

def ensure_pkgs():
    needed = ["pystac-client", "planetary-computer", "pyproj", "fsspec", "zarr", "netcdf4", "aiohttp", "dask"]
    import importlib, subprocess
    missing = []
    for p in needed:
        mod = p.replace("-", "_")
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(p)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

ensure_pkgs()

from pystac_client import Client
import planetary_computer as pc
from pyproj import CRS, Transformer
from fsspec.implementations.http import HTTPFileSystem
import aiohttp
import dask
import zarr

# safer for remote reads
dask.config.set(scheduler="single-threaded")
try:
    zarr.config.set({"async.concurrency": 4})
    zarr.config.set({"async.timeout": 1800})
    zarr.config.set({"codec_pipeline.batch_size": 4})
except Exception:
    pass

HTTP_TOTAL_TIMEOUT_S = 1800
HTTP_SOCK_READ_S     = 1800


class SASAppendingHTTPFileSystem(HTTPFileSystem):
    """
    Appends the original SAS query string verbatim to every request.
    Avoids Azure SAS signature failures caused by param re-encoding/re-ordering.
    """
    def __init__(self, query: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_query = query.lstrip("?")

        # Only set timeouts here. Do NOT create TCPConnector in a non-async process.
        timeout = aiohttp.ClientTimeout(
            total=HTTP_TOTAL_TIMEOUT_S,
            sock_connect=HTTP_TOTAL_TIMEOUT_S,
            sock_read=HTTP_SOCK_READ_S,
        )

        client_kwargs = kwargs.get("client_kwargs", {}) or {}
        client_kwargs.update({"timeout": timeout})
        self.client_kwargs = client_kwargs

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
    mapper = fs.get_mapper(base)
    return fs, mapper

def open_daymet_dataset():
    fs, mapper = get_daymet_mapper()
    ds = xr.open_zarr(mapper, consolidated=True)
    return ds, fs

def close_fs(fs):
    try:
        fs.close_session()
    except Exception:
        pass
    try:
        if getattr(fs, "session", None) is not None:
            # best-effort
            import asyncio
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(fs.session.close())
    except Exception:
        pass

def daymet_lcc_crs():
    return CRS.from_proj4(
        "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )

_TFM_LL_TO_XY = Transformer.from_crs("EPSG:4326", daymet_lcc_crs(), always_xy=True)

def bbox_lonlat_to_daymet_xy(bbox, pad_m=0.0):
    min_lon, min_lat, max_lon, max_lat = bbox
    corners = [(min_lon, min_lat), (min_lon, max_lat), (max_lon, min_lat), (max_lon, max_lat)]
    xs, ys = zip(*[_TFM_LL_TO_XY.transform(lon, lat) for lon, lat in corners])
    xmin, xmax = float(min(xs) - pad_m), float(max(xs) + pad_m)
    ymin, ymax = float(min(ys) - pad_m), float(max(ys) + pad_m)
    return xmin, xmax, ymin, ymax

def subset_daymet(ds, var, time_slice, bbox, pad_m=0.0):
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found. Example vars: {list(ds.data_vars)[:30]}")

    xmin, xmax, ymin, ymax = bbox_lonlat_to_daymet_xy(bbox, pad_m=pad_m)
    y0 = float(ds["y"].values[0])
    y1 = float(ds["y"].values[-1])
    y_slice = slice(ymax, ymin) if y0 > y1 else slice(ymin, ymax)

    sub = ds[[var]].sel(time=time_slice, x=slice(xmin, xmax), y=y_slice).sortby("time")

    # drop duplicate times (rare)
    try:
        tt = pd.to_datetime(sub["time"].values)
        _, idx = np.unique(tt, return_index=True)
        sub = sub.isel(time=np.sort(idx))
    except Exception:
        pass

    return sub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True, help="Daymet variable (e.g., tmin, tmax, prcp)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MIN_LON","MIN_LAT","MAX_LON","MAX_LAT"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pad_m", type=float, default=0.0)
    ap.add_argument("--rechunk_time", type=int, default=31)
    ap.add_argument("--sleep_s", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / f"daymet_{args.var}_{args.year}.nc"
    time_slice = slice(f"{args.year}-01-01", f"{args.year}-12-31")

    ds = None
    fs = None
    try:
        print(f"\n=== {args.year} ===")
        print("[open] opening Daymet Zarr (fresh signed SAS, verbatim append)...")
        ds, fs = open_daymet_dataset()

        print(f"[subset] VAR={args.var}, TIME={time_slice.start}..{time_slice.stop}, BBOX={tuple(args.bbox)}")
        sub = subset_daymet(ds, args.var, time_slice, tuple(args.bbox), pad_m=args.pad_m)

        print("[info] subset sizes:", dict(sub.sizes))
        print("[info] units:", sub[args.var].attrs.get("units", "(missing)"))

        sub = sub.chunk({"time": args.rechunk_time})

        print("[load] downloading subset into memory ...")
        sub_local = sub.load()

        print("[write] writing:", out_nc)
        sub_local.to_netcdf(out_nc, engine="netcdf4")
        print("[done] wrote:", out_nc)

    finally:
        if ds is not None:
            try: ds.close()
            except Exception: pass
        if fs is not None:
            close_fs(fs)
        time.sleep(args.sleep_s)

if __name__ == "__main__":
    main()
