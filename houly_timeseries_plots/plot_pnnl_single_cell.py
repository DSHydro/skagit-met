"""
Extract and plot hourly PNNL precipitation at a single grid cell location.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit-met"
pnnl_grid_file = Path("/data0/skagit_met/data_transfer/data/PNNL/historical/SERDP6km.geo_em.d01.nc")

OUTPUT_DIR = Path(BASE_DIR) / "houly_timeseries_plots/pnnl_single_cell_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Event date ranges to process (date_start, date_end, event_date, ar_category)
DATE_RANGES = [
    (datetime(1990, 11, 21, 0, 0, 0), datetime(1990, 11, 29, 23, 0, 0), "1990-11-24", "5.0"),
]

# Target location (lat, lon)
TARGET_LAT = 48.51865 #48.5
TARGET_LON = -120.7358 #-122.0


def get_pnnl_data_path(year):
    """Get PNNL data path for a given year."""
    pnnl_path = Path(f"/data0/skagit_met/data_transfer/data/PNNL/historical/{year}/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
    if pnnl_path.exists():
        return pnnl_path
    else:
        return None


def find_nearest_grid_point(lat, lon, lat2d, lon2d):
    """Find nearest grid point index using scipy's cdist."""
    distances = cdist([(lat, lon)], list(zip(lat2d.flat, lon2d.flat)))[0]
    nearest_idx = np.argmin(distances)
    x, y = np.unravel_index(nearest_idx, lat2d.shape)
    return x, y, lat2d[x, y], lon2d[x, y]


def extract_pnnl_at_point(date_start, date_end, lat, lon):
    """Extract hourly PNNL precipitation at nearest grid point."""
    year = date_start.year
    pnnl_data_path = get_pnnl_data_path(year)

    if pnnl_data_path is None:
        print(f"  [WARN] PNNL data not found for year {year}")
        return None, None, None, None

    try:
        # Load PNNL coordinates
        ds_coords = xr.open_dataset(pnnl_grid_file)
        pnnl_lon = ds_coords['XLONG_M'].values[0]
        pnnl_lat = ds_coords['XLAT_M'].values[0]
        ds_coords.close()

        # Find nearest grid point
        x, y, grid_lat, grid_lon = find_nearest_grid_point(lat, lon, pnnl_lat, pnnl_lon)
        print(f"  Nearest PNNL grid point: ({grid_lat:.4f}, {grid_lon:.4f})")

        # Load PNNL data
        ds_pnnl = xr.open_dataset(pnnl_data_path)
        date_start_pd = pd.Timestamp(date_start)
        date_end_pd = pd.Timestamp(date_end)
        ds_pnnl_subset = ds_pnnl.sel(time=slice(date_start_pd, date_end_pd))

        if len(ds_pnnl_subset.time) == 0:
            print(f"  [WARN] No PNNL data in time range")
            ds_pnnl.close()
            return None, None, None, None

        # Extract at nearest grid point
        precip_values = ds_pnnl_subset['PREC_ACC_NC'].isel(x=x, y=y).values
        times = pd.to_datetime(ds_pnnl_subset.time.values)

        ds_pnnl.close()

        return precip_values, times, grid_lat, grid_lon

    except Exception as e:
        print(f"  [ERROR] Error processing PNNL data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def plot_pnnl_hourly(precip, times, lat, lon, grid_lat, grid_lon, event_date, ar_category):
    """Create plot of hourly PNNL precipitation at a single point."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot hourly precipitation
    if precip is not None and len(precip) > 0:
        ax.plot(times, precip, marker='o', linewidth=2, markersize=4,
                color='#2ca02c', label='PNNL', alpha=0.8)

    # Formatting
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hourly Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'PNNL Hourly Precipitation - Single Grid Cell\nTarget: ({lat:.4f}, {lon:.4f}) | Grid: ({grid_lat:.4f}, {grid_lon:.4f})\n{event_date} (AR={ar_category})',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot
    output_file = OUTPUT_DIR / f"pnnl_single_cell_hourly_{event_date}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Hourly plot saved: {output_file}")
    plt.close()


def plot_pnnl_cumulative(precip, times, lat, lon, grid_lat, grid_lon, event_date, ar_category):
    """Create cumulative plot of PNNL precipitation at a single point."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Compute cumulative precipitation
    if precip is not None and len(precip) > 0:
        cumul_precip = np.cumsum(precip)
        ax.plot(times, cumul_precip, marker='o', linewidth=2, markersize=4,
                color='#2ca02c', label='PNNL (cumulative)', alpha=0.8)

    # Formatting
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=12, fontweight='bold')
    ax.set_title(f'PNNL Cumulative Precipitation - Single Grid Cell\nTarget: ({lat:.4f}, {lon:.4f}) | Grid: ({grid_lat:.4f}, {grid_lon:.4f})\n{event_date} (AR={ar_category})',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot
    output_file = OUTPUT_DIR / f"pnnl_single_cell_cumulative_{event_date}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Cumulative plot saved: {output_file}")
    plt.close()


def main():
    """Main processing loop."""
    print("=" * 80)
    print("PNNL Single Grid Cell Hourly Precipitation")
    print("=" * 80)

    # Process each event
    for idx, (start_date, end_date, event_date, ar_category) in enumerate(DATE_RANGES, 1):
        print(f"\nEvent {idx}/{len(DATE_RANGES)}: {event_date} (AR={ar_category})")
        print(f"  Period: {start_date.date()} to {end_date.date()}")
        print(f"  Target location: ({TARGET_LAT:.4f}, {TARGET_LON:.4f})")

        # Extract PNNL data at point
        precip, times, grid_lat, grid_lon = extract_pnnl_at_point(start_date, end_date, TARGET_LAT, TARGET_LON)

        if precip is not None:
            print(f"  PNNL: {len(precip)} time steps extracted")
            # Create hourly plot
            plot_pnnl_hourly(precip, times, TARGET_LAT, TARGET_LON, grid_lat, grid_lon, event_date, ar_category)
            # Create cumulative plot
            plot_pnnl_cumulative(precip, times, TARGET_LAT, TARGET_LON, grid_lat, grid_lon, event_date, ar_category)
        else:
            print(f"  Skipping event (no PNNL data)")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
