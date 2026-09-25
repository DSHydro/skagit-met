"""
Extract hourly precipitation at SNOTEL station locations from CONUS404, UCLA, PNNL.
Compare SNOTEL accumulated precipitation with hourly timeseries from each product.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
import os

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
SNOTEL_DATA_BASE = Path(BASE_DIR) / "data/hourly_snotel"
ELEVATION_SHAPEFILE = Path(BASE_DIR) / "data/GIS/skagit_elevation_3520ft.shp"

# Data paths
conus_data_path = Path("/data0/hernanqd/instance_2021_data/preparing_datasets/CONUS404/hourly_ar_non_ar_events")
ucla_data_path = Path("/data0/hernanqd/instance_2021_data/hourly_ar_non_ar_events")
ucla_coords_file = Path("/data0/hernanqd/instance_2021_data/preparing_datasets/UCLA/wrfinput_d02_coord.nc")
pnnl_grid_file = Path("/data0/skagit_met/data_transfer/data/PNNL/historical/SERDP6km.geo_em.d01.nc")

OUTPUT_DIR = Path(BASE_DIR) / "houly_timeseries_plots/comparing_snotel_sations/plots_snotel_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load elevation shapefile for masking
import geopandas as gpd
gdf_elevation = gpd.read_file(str(ELEVATION_SHAPEFILE))

# Event date ranges to process (date_start_ucla, date_start, date_end, event_date, ar_category)

# AR events
# DATE_RANGES = [
    # (datetime(1990, 11, 20, 23, 0, 0), datetime(1990, 11, 21, 0, 0, 0), datetime(1990, 11, 29, 23, 0, 0), "1990-11-24", "5.0"),
    # (datetime(1995, 11, 25, 23, 0, 0), datetime(1995, 11, 26, 0, 0, 0), datetime(1995, 12, 4, 23, 0, 0), "1995-11-29", "4.0"),
    # (datetime(1990, 11, 6, 23, 0, 0), datetime(1990, 11, 7, 0, 0, 0), datetime(1990, 11, 15, 23, 0, 0), "1990-11-10", "4.0"),
    # (datetime(2006, 11, 3, 23, 0, 0), datetime(2006, 11, 4, 0, 0, 0), datetime(2006, 11, 12, 23, 0, 0), "2006-11-07", "5.0"),
    # (datetime(2003, 10, 13, 23, 0, 0), datetime(2003, 10, 14, 0, 0, 0), datetime(2003, 10, 28, 23, 0, 0), "2003-10-21", "5.0"),
    # (datetime(2021, 11, 12, 23, 0, 0), datetime(2021, 11, 13, 0, 0, 0), datetime(2021, 11, 20, 23, 0, 0), "2021-11-15", "4.0"),
# ]

# non-AR events
DATE_RANGES = [
    (datetime(1995, 11, 28, 23, 0, 0), datetime(1995, 11, 29, 0, 0, 0), datetime(1995, 12, 7, 23, 0, 0), "1995-12-02", "0.0"),
    (datetime(2011, 1, 14, 23, 0, 0), datetime(2011, 1, 15, 0, 0, 0), datetime(2011, 1, 23, 23, 0, 0), "2011-01-18", "0.0"),
    (datetime(2015, 11, 10, 23, 0, 0), datetime(2015, 11, 11, 0, 0, 0), datetime(2015, 11, 19, 23, 0, 0), "2015-11-14", "0.0"),
    (datetime(2007, 3, 9, 23, 0, 0), datetime(2007, 3, 10, 0, 0, 0), datetime(2007, 3, 19, 22, 0, 0), "2007-03-13", "0.0"),
    # (datetime(2021, 11, 29, 23, 0, 0), datetime(2021, 11, 30, 0, 0, 0), datetime(2021, 12, 7, 23, 0, 0), "2021-12-03", "0.0"),
    (datetime(2010, 12, 10, 23, 0, 0), datetime(2010, 12, 11, 0, 0, 0), datetime(2010, 12, 19, 23, 0, 0), "2010-12-14", "0.0"),
    (datetime(2009, 11, 23, 23, 0, 0), datetime(2009, 11, 24, 0, 0, 0), datetime(2009, 12, 2, 23, 0, 0), "2009-11-27", "0.0"),
]

def get_pnnl_data_path(year):
    """Get PNNL data path for a given year."""
    pnnl_path = Path(f"/data0/skagit_met/data_transfer/data/PNNL/historical/{year}/PNNL_WRF.HIST.CTRL.hourly.PREC_ACC_NC.{year}.nc")
    if pnnl_path.exists():
        return pnnl_path
    else:
        return None


def quality_check_accumulated_precip(precip_data, site_id):
    """
    Quality check for accumulated precipitation.
    Detects and removes peaks (points higher than both neighbors).
    Returns cleaned data and list of issues found.
    """
    cleaned = np.copy(precip_data).astype(float)
    issues = []
    bad_indices = set()

    # Detect PEAKS - points that are higher than both neighbors
    peaks = []
    for i in range(1, len(cleaned)-1):
        if not np.isnan(cleaned[i-1]) and not np.isnan(cleaned[i]) and not np.isnan(cleaned[i+1]):
            if cleaned[i] > cleaned[i-1] and cleaned[i] > cleaned[i+1]:
                peaks.append(i)
                bad_indices.add(i)

    if len(peaks) > 0:
        issues.append(f"    Peaks (local maxima): {len(peaks)} points removed")

    # Mark all bad indices as NaN
    for idx in bad_indices:
        cleaned[idx] = np.nan

    return cleaned, issues


def load_snotel_data_with_coords(start_date, end_date):
    """Load SNOTEL data and extract station coordinates."""
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Find SNOTEL file matching the event period
    zarr_files = list(SNOTEL_DATA_BASE.glob('*_SNOTEL_hourly_data.zarr'))
    zarr_path = None

    for f in zarr_files:
        fname = f.stem.replace('_SNOTEL_hourly_data', '')
        date_parts = fname.split('_')
        if len(date_parts) >= 2:
            try:
                file_start = pd.Timestamp(date_parts[0])
                file_end = pd.Timestamp(date_parts[1])
                if file_start <= end_ts and start_ts <= file_end:
                    zarr_path = f
                    break
            except:
                continue

    if zarr_path is None:
        print(f"  [WARN] SNOTEL data not found for period {start_ts.date()} to {end_ts.date()}")
        return None

    try:
        ds_snotel = xr.open_zarr(str(zarr_path))
        times = pd.to_datetime(ds_snotel.time.values)

        # Filter times to event period
        mask = (times >= pd.Timestamp(start_date)) & (times <= pd.Timestamp(end_date))
        filtered_times = times[mask]
        filtered_time_indices = np.where(mask)[0]

        # Get station information
        sites = ds_snotel.site.values
        lats = ds_snotel.lat.values
        lons = ds_snotel.lon.values
        site_names = ds_snotel.site_name.values
        elevations = ds_snotel.elevation_ft.values

        # Get precipitation data for filtered times
        precip_data = ds_snotel['ACCUMULATED PRECIPITATION'].isel(time=filtered_time_indices).values

        # Convert from inches to mm (SNOTEL is in inches, multiply by 25.4)
        precip_mm = precip_data * 25.4

        ds_snotel.close()

        station_info = []
        for i, site in enumerate(sites):
            station_info.append({
                'site_id': site,
                'site_name': site_names[i] if hasattr(site_names[i], 'item') else site_names[i],
                'lat': float(lats[i]),
                'lon': float(lons[i]),
                'elevation_ft': float(elevations[i])
            })

        snotel_data = {
            'times': filtered_times,
            'precip': precip_mm,  # (time, site)
            'stations': station_info,
            'n_stations': len(sites)
        }
        print(f"  SNOTEL loaded: {snotel_data['n_stations']} stations, {len(filtered_times)} time steps")
        return snotel_data
    except Exception as e:
        print(f"  [ERROR] SNOTEL loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_nearest_grid_point(lat, lon, lat2d, lon2d):
    """Find nearest grid point index using scipy's cdist. Returns (i, j) for CONUS/UCLA."""
    distances = cdist([(lat, lon)], list(zip(lat2d.flat, lon2d.flat)))[0]
    nearest_idx = np.argmin(distances)
    j, i = np.unravel_index(nearest_idx, lat2d.shape)
    return i, j, lat2d[j, i], lon2d[j, i]


def find_nearest_grid_point_pnnl(lat, lon, lat2d, lon2d):
    """Find nearest grid point index for PNNL. Returns (x, y) for xarray isel."""
    distances = cdist([(lat, lon)], list(zip(lat2d.flat, lon2d.flat)))[0]
    nearest_idx = np.argmin(distances)
    x, y = np.unravel_index(nearest_idx, lat2d.shape)
    return x, y, lat2d[x, y], lon2d[x, y]


def extract_conus_at_point(date_start, date_end, lat, lon):
    """Extract hourly precipitation at nearest grid point from CONUS404 files."""
    conus_files_list = sorted(conus_data_path.glob('*.PREC_ACC_NC.wrf2d_d01_*.nc'))

    if not conus_files_list:
        return None, None

    # Find files in date range
    conus_files_in_range = []
    for f in conus_files_list:
        try:
            ds_temp = xr.open_dataset(f)
            time = ds_temp.Time.values[0]
            if pd.Timestamp(date_start) <= pd.Timestamp(time) <= pd.Timestamp(date_end):
                conus_files_in_range.append(f)
            ds_temp.close()
        except:
            pass

    if not conus_files_in_range:
        return None, None

    try:
        # Concatenate all files in range along Time dimension
        ds_conus_list = [xr.open_dataset(f) for f in conus_files_in_range]
        ds_conus = xr.concat(ds_conus_list, dim='Time')

        # Find nearest grid point from concatenated dataset
        lat2d_conus = ds_conus['XLAT'].values
        lon2d_conus = ds_conus['XLONG'].values
        i, j, _, _ = find_nearest_grid_point(lat, lon, lat2d_conus, lon2d_conus)

        # Extract at point
        precip_values = ds_conus['PREC_ACC_NC'].isel(south_north=j, west_east=i).values

        times = pd.to_datetime(ds_conus.Time.values)

        ds_conus.close()
        return np.array(precip_values), np.array(times)
    except Exception as e:
        print(f"    [WARN] Error processing CONUS data: {e}")
        import traceback
        traceback.print_exc()

    return None, None


def extract_pnnl_at_point(date_start, date_end, lat, lon):
    """Extract hourly PNNL precipitation at nearest grid point."""
    year = date_start.year
    pnnl_data_path = get_pnnl_data_path(year)

    if pnnl_data_path is None:
        print(f"      [WARN] PNNL data not found for year {year}")
        return None, None

    try:
        # Load PNNL coordinates
        ds_coords = xr.open_dataset(pnnl_grid_file)
        pnnl_lon = ds_coords['XLONG_M'].values[0]
        pnnl_lat = ds_coords['XLAT_M'].values[0]
        ds_coords.close()

        # Find nearest grid point
        x, y, grid_lat, grid_lon = find_nearest_grid_point_pnnl(lat, lon, pnnl_lat, pnnl_lon)
        print(f"      Nearest PNNL grid point: ({grid_lat:.4f}, {grid_lon:.4f})")

        # Load PNNL data
        ds_pnnl = xr.open_dataset(pnnl_data_path)
        date_start_pd = pd.Timestamp(date_start)
        date_end_pd = pd.Timestamp(date_end)
        ds_pnnl_subset = ds_pnnl.sel(time=slice(date_start_pd, date_end_pd))

        if len(ds_pnnl_subset.time) == 0:
            print(f"      [WARN] No PNNL data in time range")
            ds_pnnl.close()
            return None, None

        # Extract at nearest grid point
        precip_values = ds_pnnl_subset['PREC_ACC_NC'].isel(x=x, y=y).values
        times = pd.to_datetime(ds_pnnl_subset.time.values)

        ds_pnnl.close()

        return precip_values, times

    except Exception as e:
        print(f"      [ERROR] Error processing PNNL data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_ucla_at_point(date_start, date_end, lat, lon):
    """Extract hourly precipitation increments at nearest grid point from UCLA files."""
    date_start_ucla = date_start - timedelta(hours=1)
    current = date_start_ucla
    ucla_filenames = []

    while current <= date_end:
        fn = f"auxhist_d01_{current.strftime('%Y-%m-%d_%H:%M:%S')}.nc"
        ucla_filenames.append(ucla_data_path / fn)
        current += timedelta(hours=1)

    ucla_files_exist = [f for f in ucla_filenames if f.exists()]

    if not ucla_files_exist:
        return None, None

    try:
        # Load UCLA coordinates from separate file
        ds_coords = xr.open_dataset(ucla_coords_file)
        lat2d_ucla = ds_coords['lat2d'].values
        lon2d_ucla = ds_coords['lon2d'].values
        ds_coords.close()

        # Find nearest grid point
        i, j, _, _ = find_nearest_grid_point(lat, lon, lat2d_ucla, lon2d_ucla)
    except Exception as e:
        print(f"    [WARN] Could not find UCLA grid point: {e}")
        return None, None

    precip_values = []
    times = []

    try:
        ds_ucla = xr.open_mfdataset(ucla_files_exist, concat_dim='Time', combine='nested')

        # Get total rain (cumulative convective + non-convective)
        total_rain = ds_ucla['RAINC'] + ds_ucla['RAINNC']

        # Extract at point and compute hourly increments
        point_data = total_rain.isel(south_north=j, west_east=i)
        increments = point_data.diff(dim='Time').values

        # Get times (skip first time since diff creates NaN)
        times_raw = ds_ucla['Times'].values[1:]
        times_dt = []
        for t in times_raw:
            if isinstance(t, bytes):
                t_str = t.decode().replace('_', ' ')
            else:
                t_str = str(t).replace('_', ' ')
            times_dt.append(pd.Timestamp(t_str))

        precip_values = increments
        times = np.array(times_dt)
        ds_ucla.close()
    except Exception as e:
        print(f"    [WARN] Error reading UCLA data: {e}")

    if len(precip_values) > 0:
        return np.array(precip_values), np.array(times)
    return None, None


def plot_snotel_mean(snotel_times, snotel_precip_mm, stations, event_date, ar_category):
    """Create plot of SNOTEL stations and their mean (all valid stations required)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Filter out stations with insufficient data for this event
    # Require at least 14 measurements (roughly 2 per day for a week-long event)
    min_measurements = 14
    valid_stations = []
    valid_precip = []
    for i, station in enumerate(stations):
        n_valid = (~np.isnan(snotel_precip_mm[:, i])).sum()
        if n_valid >= min_measurements:
            valid_stations.append(station)
            valid_precip.append(snotel_precip_mm[:, i])
            print(f"  [KEEP] {station['site_id']}: {n_valid} valid measurements")
        else:
            print(f"  [SKIP] {station['site_id']}: only {n_valid} measurements (need {min_measurements})")

    if len(valid_stations) == 0:
        print(f"  [WARN] No valid stations with data for this event")
        plt.close()
        return

    valid_precip = np.array(valid_precip).T  # (time, site)

    # Convert to xarray for easier manipulation
    ds = xr.Dataset(
        {'precip': (['time', 'site'], valid_precip)},
        coords={'time': snotel_times, 'site': [s['site_id'] for s in valid_stations]}
    )

    # Normalize: subtract first VALID value for each station (not just the first time)
    # This handles cases where the first timestep is NaN
    first_valid = ds['precip'].where(ds['precip'].notnull()).min(dim='time')
    storm_precip = ds['precip'] - first_valid

    # Find timesteps where ALL valid stations have data (no NaN in any station)
    all_stations_valid = storm_precip.notnull().all(dim='site')

    # Calculate mean only at timesteps where all valid stations are present
    # Timesteps with any NaN are excluded from the mean
    mean = storm_precip.where(all_stations_valid).mean(dim='site')

    # Plot mean (mean of all valid stations at each time)
    mean_elev = np.mean([s['elevation_ft'] for s in valid_stations])
    ax.plot(snotel_times, mean.values, marker='o', linewidth=2, markersize=8,
            label=f'Mean ({mean_elev:.0f} ft)', color='black', alpha=0.8, zorder=10)

    # Formatting
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=11, fontweight='bold')
    ax.set_title(f'SNOTEL Accumulated Precipitation - Mean (All Stations Required)\n{event_date} (AR={ar_category})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot
    output_file = OUTPUT_DIR / f"snotel_mean_{event_date}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    """Main processing loop."""
    print("=" * 80)
    print("SNOTEL Mean Precipitation (Common Observations)")
    print("=" * 80)

    # Process each event
    for idx, (date_start_ucla, start_date, end_date, event_date, ar_category) in enumerate(DATE_RANGES, 1):
        print(f"\n{'=' * 80}")
        print(f"Event {idx}/{len(DATE_RANGES)}: {event_date} (AR={ar_category}) ({start_date.date()} to {end_date.date()})")
        print(f"{'=' * 80}")

        # Load SNOTEL data
        print("Loading SNOTEL data...")
        snotel_data = load_snotel_data_with_coords(start_date, end_date)

        if snotel_data is None:
            print("  Skipping this event (no SNOTEL data)")
            continue

        snotel_times = snotel_data['times']
        snotel_precip_mm = snotel_data['precip']  # (time, site) in mm
        stations = snotel_data['stations']

        print(f"  Loaded {len(stations)} SNOTEL stations: {', '.join([s['site_id'] for s in stations])}")

        # Quality check: detect and mask bad data for each station
        print(f"  Quality check and cleaning:")
        snotel_precip_cleaned = np.copy(snotel_precip_mm)
        for i, station in enumerate(stations):
            cleaned, issues = quality_check_accumulated_precip(snotel_precip_mm[:, i], station['site_id'])
            snotel_precip_cleaned[:, i] = cleaned  # Use cleaned data
            if issues:
                print(f"    {station['site_id']}:")
                for issue in issues:
                    print(issue)
            else:
                print(f"    {station['site_id']}: OK")

        # Create mean plot across all stations (using cleaned data)
        print(f"  Creating mean plot...")
        plot_snotel_mean(snotel_times, snotel_precip_cleaned, stations, event_date, ar_category)

        # # Optional: Uncomment below to also extract other datasets for comparison
        # # For now, focusing on SNOTEL only
        # for station_idx, station_info in enumerate(stations):
        #     print(f"\n  Station {station_idx + 1}/{len(stations)}: {station_info['site_name']} ({station_info['site_id']})")
        #     lat = station_info['lat']
        #     lon = station_info['lon']
        #     print(f"    Extracting hourly precipitation at station location ({lat:.4f}, {lon:.4f})...")
        #     conus_precip, conus_times = extract_conus_at_point(start_date, end_date, lat, lon)
        #     ucla_precip, ucla_times = extract_ucla_at_point(start_date, end_date, lat, lon)
        #     pnnl_precip, pnnl_times = extract_pnnl_at_point(start_date, end_date, lat, lon)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
