import argparse
import os
from pathlib import Path

import elevation
import geopandas as gpd
import rioxarray  # noqa: F401
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data/GIS/SkagitRiver_90mDEM.tif"
DEFAULT_GEOJSON = PROJECT_ROOT / "data/GIS/SkagitBoundary.json"


def setupArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downloads an SRTM 90 m Digital Elevation Model (DEM) for a bounding box, "
            "clips it to a watershed boundary, and saves the result as a GeoTIFF."
        )
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
        help=(
            "Bounding box for DEM download as four floats: minLon minLat maxLon maxLat. "
            "If omitted, bounds are computed from --geojson."
        ),
    )
    parser.add_argument(
        "--outputFile",
        default=DEFAULT_OUTPUT,
        type=str,
        help=f"Output path for the raw (unclipped) DEM GeoTIFF. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--clippedOutputFile",
        default=None,
        type=str,
        help=(
            "Output path for the clipped DEM GeoTIFF. Defaults to <outputFile stem>_clipped.tif next to the outputFile."
        ),
    )
    parser.add_argument(
        "--geojson",
        default=DEFAULT_GEOJSON,
        type=str,
        help=(
            "GeoJSON file defining the watershed boundary used to clip the DEM. "
            f"Default: {DEFAULT_GEOJSON}"
        ),
    )
    parser.add_argument(
        "--skipDownload",
        action="store_true",
        default=False,
        help="Skip downloading a new DEM if the output file already exists.",
    )
    return parser.parse_args()


def bounds_from_geojson(geojson_path: str, buffer_deg: float = 0.0) -> tuple[float, float, float, float]:
    gdf = gpd.read_file(geojson_path)

    # Ensure lon/lat degree bounds are interpreted consistently
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = gdf.total_bounds
    return (minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg)


def download_dem(bounds: tuple[float, float, float, float], output_file: str) -> None:
    """Download SRTM 90 m tiles for the given bounding box and merge into a single GeoTIFF."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Downloading SRTM 90 m DEM for bounds {bounds} -> {output_file}")
    elevation.clip(bounds=bounds, output=output_file)
    print("Download complete.")


def clip_dem_to_boundary(dem_path: str, geojson_path: str, clipped_output: str) -> xr.DataArray:
    """Open the DEM with rioxarray and clip it to the watershed polygon."""
    boundary = gpd.read_file(geojson_path)
    dem = rioxarray.open_rasterio(dem_path, masked=True)

    # Ensure the boundary CRS matches the DEM CRS before clipping
    if boundary.crs is None:
        print("Warning: boundary GeoJSON has no CRS; assuming EPSG:4326.")
        boundary = boundary.set_crs("EPSG:4326")
    if dem.rio.crs is not None and boundary.crs != dem.rio.crs:
        boundary = boundary.to_crs(dem.rio.crs)

    clipped = dem.rio.clip(boundary.geometry)

    os.makedirs(os.path.dirname(os.path.abspath(clipped_output)), exist_ok=True)
    clipped.rio.to_raster(clipped_output)
    print(f"Clipped DEM written to: {clipped_output}")
    return clipped


def get_clipped_path(output_file: str) -> str:
    """Derive a default clipped output path from the raw DEM path."""
    base, ext = os.path.splitext(output_file)
    return base + "_clipped" + ext


if __name__ == "__main__":
    args = setupArgs()

    output_file = os.path.abspath(args.outputFile)
    geojson_file = os.path.abspath(args.geojson)
    clipped_output = os.path.abspath(args.clippedOutputFile or get_clipped_path(output_file))

    if args.bounds is None:
        print("Bounding box automatically computed from GeoJSON.")
        bounds = bounds_from_geojson(geojson_file)
    else:
        bounds = tuple(args.bounds)

    # Download raw DEM
    if args.skipDownload and os.path.exists(output_file):
        print(f"Raw DEM already exists, skipping download: {output_file}")
    else:
        download_dem(bounds, output_file)

    if not os.path.exists(output_file):
        raise RuntimeError(f"Expected DEM file not found after download: {output_file}")

    # Clip to watershed boundary
    print(f"Clipping DEM to boundary: {geojson_file}")
    clip_dem_to_boundary(output_file, geojson_file, clipped_output)
