#!/usr/bin/env python3
"""
Download DAYMET precipitation spatial subsets via THREDDS NCSS Web Service

No authentication required! 

Usage:
    python daymet_ncss_download.py --years 2010 2020 \
      --bbox 49.313400 47.957521 -120.654609 -122.538683 \
      --inspect
"""

import requests
import xarray as xr
from pathlib import Path
from typing import List
import argparse


# THREDDS NCSS Configuration - Using official ORNL DAAC format
THREDDS_BASE = "https://thredds.daac.ornl.gov/thredds/ncss/grid/ornldaac/2129"


def build_ncss_url(
    year: int,
    north: float,
    south: float,
    east: float,
    west: float,
) -> str:
    """Build THREDDS NCSS request URL for DAYMET precipitation subset."""
    
    filename = f"daymet_v4_daily_na_prcp_{year}.nc"
    file_url = f"{THREDDS_BASE}/{filename}"
    
    # Handle leap years - Daymet uses 12-30 for leap years, 12-31 for others
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    end_day = "12-30" if is_leap_year else "12-31"
    
    # Build NCSS parameters (matching ORNL DAAC official format)
    url = f"{file_url}?"
    url += f"var=lat&var=lon&var=prcp"
    url += f"&north={north}&west={west}&east={east}&south={south}"
    url += f"&horizStride=1"
    url += f"&time_start={year}-01-01T12:00:00Z"
    url += f"&time_end={year}-{end_day}T12:00:00Z"
    url += f"&timeStride=1&accept=netcdf"
    
    return url


def download_daymet_subset(
    year: int,
    north: float,
    south: float,
    east: float,
    west: float,
    output_dir: Path,
) -> Path:
    """Download a spatial subset of DAYMET precipitation for one year."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build URL
    url = build_ncss_url(year, north, south, east, west)
    
    # Output filename
    output_file = output_dir / f"daymet_prcp_{year}.nc"
    
    print(f"  Year {year}...", end='', flush=True)
    
    try:
        response = requests.get(url, timeout=600, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size_mb = output_file.stat().st_size / 1e6
        print(f" ✓ ({file_size_mb:.1f} MB)")
        
        return output_file
    
    except requests.exceptions.RequestException as e:
        print(f" ✗ Error: {e}")
        if output_file.exists():
            output_file.unlink()
        return None


def download_daymet_range(
    years: List[int],
    north: float,
    south: float,
    east: float,
    west: float,
    output_dir: Path,
) -> List[Path]:
    """Download DAYMET for multiple years."""
    
    files = []
    for year in sorted(years):
        file_path = download_daymet_subset(year, north, south, east, west, output_dir)
        if file_path:
            files.append(file_path)
    
    return files


def load_and_inspect(files: List[Path]) -> xr.Dataset:
    """Load downloaded files and show summary."""
    
    if not files:
        print("No files to load")
        return None
    
    print(f"\nLoading {len(files)} file(s)...")
    
    try:
        ds = xr.open_mfdataset(files, combine='by_coords')
        
        print("\n" + "="*60)
        print("✓ Data Loaded Successfully")
        print("="*60)
        print(f"\nDimensions:")
        print(f"  Time steps: {ds.dims['time']}")
        print(f"  Lat points: {ds.dims['lat']}")
        print(f"  Lon points: {ds.dims['lon']}")
        
        print(f"\nDate range:")
        print(f"  Start: {ds['time'].values[0]}")
        print(f"  End: {ds['time'].values[-1]}")
        
        print(f"\nPrecipitation (prcp):")
        prcp = ds['prcp']
        print(f"  Data type: {prcp.dtype}")
        print(f"  Units: mm/day")
        print(f"  Min: {float(prcp.min()):.2f} mm/day")
        print(f"  Max: {float(prcp.max()):.2f} mm/day")
        print(f"  Mean: {float(prcp.mean()):.2f} mm/day")
        print(f"  Std Dev: {float(prcp.std()):.2f} mm/day")
        
        print(f"\nData ready for analysis!")
        print(f"Location: {files[0].parent}")
        print("="*60 + "\n")
        
        return ds
    
    except Exception as e:
        print(f"✗ Error loading files: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Download DAYMET precipitation spatial subsets via THREDDS NCSS',
    )
    
    parser.add_argument('--years', type=int, nargs=2, required=True,
                       metavar=('START', 'END'),
                       help='Year range (e.g., 2010 2020)')
    parser.add_argument('--bbox', type=float, nargs=4, required=True,
                       metavar=('NORTH', 'SOUTH', 'EAST', 'WEST'),
                       help='Bounding box in decimal degrees')
    parser.add_argument('--output-dir', type=Path, default=Path('./daymet_data'),
                       help='Output directory (default: ./daymet_data)')
    parser.add_argument('--inspect', action='store_true',
                       help='Load and inspect data after download')
    
    args = parser.parse_args()
    
    year_start, year_end = args.years
    years = list(range(year_start, year_end + 1))
    north, south, east, west = args.bbox
    
    # Print banner
    print("\n" + "="*60)
    print("DAYMET Precipitation Download (THREDDS NCSS)")
    print("="*60)
    print(f"Region: N={north}°, S={south}°, E={east}°, W={west}°")
    print(f"Years: {year_start}-{year_end} ({len(years)} years)")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    print(f"Downloading {len(years)} files...\n")
    
    # Download files
    files = download_daymet_range(
        years, north, south, east, west,
        args.output_dir
    )
    
    if not files:
        print("\n✗ No files downloaded successfully")
        return 1
    
    print(f"\n✓ Downloaded {len(files)} file(s)")
    
    # Optionally inspect
    if args.inspect:
        ds = load_and_inspect(files)
        if ds:
            ds.close()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
