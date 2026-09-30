"""
Extract spatial mean hourly precipitation from CONUS404, UCLA, PNNL over elevation shapefile.
Compare with SNOTEL mean accumulated precipitation.
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import os
import geopandas as gpd

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
SNOTEL_DATA_BASE = Path(BASE_DIR) / "data/hourly_snotel"
ELEVATION_SHAPEFILE = Path(BASE_DIR) / "data/GIS/skagit_elevation_3520ft.shp"

# Data paths
conus_data_path = Path("/data0/skagit_met/data_transfer/data/CONUS_hourly/hourly_ar_non_ar_events")
ucla_data_path = Path("/data0/skagit_met/data_transfer/data/ucla_era5_d02_hourly/hourly_ar_non_ar_events")
ucla_coords_file = Path("/data0/hernanqd/instance_2021_data/preparing_datasets/UCLA/wrfinput_d02_coord.nc")
pnnl_grid_file = Path("/data0/skagit_met/data_transfer/data/PNNL/historical/SERDP6km.geo_em.d01.nc")

OUTPUT_DIR = Path(BASE_DIR) / "houly_timeseries_plots/comparing_snotel_sations/plots_snotel_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def get_elevation_mask(lon2d, lat2d):
    """Create 2D mask for elevation shapefile boundary."""
    boundary_gdf = gpd.read_file(str(ELEVATION_SHAPEFILE)).to_crs("EPSG:4326")
    if hasattr(boundary_gdf, 'union_all'):
        boundary_poly = boundary_gdf.union_all()
    else:
        boundary_poly = boundary_gdf.unary_union

    df_pts = pd.DataFrame({'lon': lon2d.flatten(), 'lat': lat2d.flatten()})
    gdf_pts = gpd.GeoDataFrame(
        df_pts,
        geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
        crs="EPSG:4326"
    )
    inside = gdf_pts.intersects(boundary_poly).values
    return inside.reshape(lon2d.shape)


def calculate_basin_mean(da, mask_2d):
    """Calculate basin-mean precipitation with proper masking."""
    data = da.values
    mask = mask_2d.values if hasattr(mask_2d, 'values') else mask_2d

    if data.ndim == 3:
        time_steps = data.shape[0]
        data_flat = data.reshape(time_steps, -1)
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            mean_vals = np.nanmean(data_flat[:, mask_idx], axis=1)
        else:
            mean_vals = np.full(time_steps, np.nan)

        # Get time coordinate - try multiple approaches
        time_coord = None

        try:
            first_dim = da.dims[0]
            if first_dim in da.coords:
                time_coord_candidate = da.coords[first_dim].values
                if len(time_coord_candidate) > 0:
                    time_coord = time_coord_candidate
        except Exception as e:
            pass

        if time_coord is None:
            for name in ['time', 'Time', 'TIME']:
                if name in da.coords:
                    time_coord = da.coords[name].values
                    break

        if time_coord is None:
            time_coord = range(time_steps)

        return pd.Series(mean_vals, index=pd.to_datetime(time_coord))
    elif data.ndim == 2:
        data = data.astype(float)
        data[data < 0] = np.nan
        data_flat = data.flatten()
        mask_flat = mask.flatten()
        mask_idx = mask_flat > 0
        if mask_idx.any():
            mean_val = np.nanmean(data_flat[mask_idx])
        else:
            mean_val = np.nan
        return mean_val
    else:
        raise ValueError(f"Unexpected data dimensions: {data.ndim}")


def quality_check_accumulated_precip(precip_data, site_id):
    """Quality check for accumulated precipitation. Detects and removes peaks."""
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


def extract_conus_spatial_mean(date_start, date_end):
    """Extract spatial mean hourly precipitation from CONUS404 files over elevation mask."""
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
        ds_conus_list = [xr.open_dataset(f) for f in conus_files_in_range]
        ds_conus = xr.concat(ds_conus_list, dim='Time')

        lat2d_conus = ds_conus['XLAT'].values
        lon2d_conus = ds_conus['XLONG'].values
        mask = get_elevation_mask(lon2d_conus, lat2d_conus)

        series_conus = calculate_basin_mean(ds_conus['PREC_ACC_NC'], mask)
        times_conus = pd.to_datetime(ds_conus.Time.values)

        ds_conus.close()
        return series_conus, times_conus
    except Exception as e:
        print(f"    [WARN] Error processing CONUS data: {e}")
        import traceback
        traceback.print_exc()

    return None, None


def extract_pnnl_spatial_mean(date_start, date_end):
    """Extract spatial mean hourly PNNL precipitation over elevation mask."""
    year = date_start.year
    pnnl_data_path = get_pnnl_data_path(year)

    if pnnl_data_path is None:
        print(f"    [WARN] PNNL data not found for year {year}")
        return None, None

    try:
        # Load PNNL coordinates
        ds_coords = xr.open_dataset(pnnl_grid_file)
        pnnl_lon = ds_coords['XLONG_M'].values[0]
        pnnl_lat = ds_coords['XLAT_M'].values[0]
        ds_coords.close()

        mask = get_elevation_mask(pnnl_lon, pnnl_lat)

        # Load PNNL data
        ds_pnnl = xr.open_dataset(pnnl_data_path)
        date_start_pd = pd.Timestamp(date_start)
        date_end_pd = pd.Timestamp(date_end)
        ds_pnnl_subset = ds_pnnl.sel(time=slice(date_start_pd, date_end_pd))

        if len(ds_pnnl_subset.time) == 0:
            print(f"    [WARN] No PNNL data in time range")
            ds_pnnl.close()
            return None, None

        series_pnnl = calculate_basin_mean(ds_pnnl_subset['PREC_ACC_NC'], mask)
        times_pnnl = pd.to_datetime(ds_pnnl_subset.time.values)

        ds_pnnl.close()

        return series_pnnl, times_pnnl

    except Exception as e:
        print(f"    [ERROR] Error processing PNNL data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_ucla_spatial_mean(date_start, date_end):
    """Extract spatial mean hourly UCLA precipitation increments over elevation mask."""
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

        mask = get_elevation_mask(lon2d_ucla, lat2d_ucla)

        ds_ucla = xr.open_mfdataset(ucla_files_exist, concat_dim='Time', combine='nested')

        # Get total rain and compute increments
        total_rain = ds_ucla['RAINC'] + ds_ucla['RAINNC']
        total_increments = total_rain.diff(dim='Time')

        # Convert Times strings to datetime64 and round to nearest hour
        times_raw_ucla = ds_ucla['Times'].values
        times_ucla_dt = []
        for t in times_raw_ucla[1:]:
            if isinstance(t, bytes):
                t_str = t.decode().replace('_', ' ')
            else:
                t_str = str(t).replace('_', ' ')
            ts = pd.Timestamp(t_str)
            ts_rounded = ts.round('1H')
            times_ucla_dt.append(ts_rounded)
        times_ucla_dt = np.array(times_ucla_dt, dtype='datetime64[ns]')

        # Restore time coordinate with proper datetime values
        total_increments = total_increments.assign_coords(
            Time=('Time', times_ucla_dt)
        )

        series_ucla = calculate_basin_mean(total_increments, mask)
        times_ucla = times_ucla_dt

        ds_ucla.close()
        return series_ucla, times_ucla
    except Exception as e:
        print(f"    [WARN] Error reading UCLA data: {e}")

    return None, None


def plot_mean_comparison(snotel_times, snotel_mean, snotel_elevations,
                         conus_series, conus_times,
                         ucla_series, ucla_times,
                         pnnl_series, pnnl_times,
                         event_date, ar_category):
    """Create comparison plot for mean SNOTEL and spatial mean datasets."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Normalize SNOTEL mean accumulated to start at zero at storm start
    snotel_first_valid = snotel_mean[~np.isnan(snotel_mean)][0] if np.any(~np.isnan(snotel_mean)) else 0
    snotel_normalized = snotel_mean - snotel_first_valid

    # Plot cumulative precipitation from datasets
    if conus_series is not None and len(conus_series) > 0:
        conus_cumulative = conus_series.cumsum()
        ax.plot(conus_times, conus_cumulative.values, marker='D', linewidth=2, markersize=5,
                label='CONUS404', color='#d62728', alpha=0.8)

    if ucla_series is not None and len(ucla_series) > 0:
        ucla_cumulative = ucla_series.cumsum()
        ax.plot(ucla_times, ucla_cumulative.values, marker='v', linewidth=2, markersize=5,
                label='UCLA', color='#9467bd', alpha=0.8)

    if pnnl_series is not None and len(pnnl_series) > 0:
        pnnl_cumulative = pnnl_series.cumsum()
        ax.plot(pnnl_times, pnnl_cumulative.values, marker='s', linewidth=2, markersize=5,
                label='PNNL', color='#ff7f0e', alpha=0.8)

    # Plot SNOTEL mean (normalized)
    ax.plot(snotel_times, snotel_normalized, marker='o', linewidth=2, markersize=6,
            label=f'SNOTEL Mean ({snotel_elevations:.0f} ft)', color='black', alpha=0.8, zorder=5)

    # Formatting
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cumulative Precipitation (mm)', fontsize=11, fontweight='bold')
    ax.set_title(f'Mean Cumulative Precipitation - Elevation Mask\n{event_date} (AR={ar_category})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot
    output_file = OUTPUT_DIR / f"snotel_vs_hourly_mean_{event_date}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    """Main processing loop."""
    print("=" * 80)
    print("SNOTEL Mean vs Hourly Datasets Spatial Mean Comparison")
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
        snotel_precip = snotel_data['precip']  # (time, site)
        stations = snotel_data['stations']

        print(f"  Loaded {len(stations)} SNOTEL stations")

        # Quality check and calculate SNOTEL mean
        print("Quality check and cleaning SNOTEL data...")
        snotel_precip_cleaned = np.copy(snotel_precip)
        for i, station in enumerate(stations):
            cleaned, issues = quality_check_accumulated_precip(snotel_precip[:, i], station['site_id'])
            snotel_precip_cleaned[:, i] = cleaned
            if issues:
                print(f"  {station['site_id']}:")
                for issue in issues:
                    print(issue)
            else:
                print(f"  {station['site_id']}: OK")

        # Calculate SNOTEL mean (require all stations to have data at each time step)
        min_measurements = 14
        valid_stations = []
        valid_precip = []
        for i, station in enumerate(stations):
            n_valid = (~np.isnan(snotel_precip_cleaned[:, i])).sum()
            if n_valid >= min_measurements:
                valid_stations.append(station)
                valid_precip.append(snotel_precip_cleaned[:, i])
                print(f"  [KEEP] {station['site_id']}: {n_valid} valid measurements")
            else:
                print(f"  [SKIP] {station['site_id']}: only {n_valid} measurements (need {min_measurements})")

        if len(valid_stations) == 0:
            print("  [WARN] No valid stations with data for this event")
            continue

        valid_precip = np.array(valid_precip).T  # (time, site)

        # Calculate mean - normalize first
        first_valid_per_station = []
        for i in range(valid_precip.shape[1]):
            col = valid_precip[:, i]
            first_val = col[~np.isnan(col)][0] if np.any(~np.isnan(col)) else np.nan
            first_valid_per_station.append(first_val)

        normalized_precip = np.copy(valid_precip)
        for i, first_val in enumerate(first_valid_per_station):
            normalized_precip[:, i] = valid_precip[:, i] - first_val

        # Mean only at timesteps where all valid stations have data
        all_valid = ~np.isnan(normalized_precip).any(axis=1)
        snotel_mean = np.full(normalized_precip.shape[0], np.nan)
        snotel_mean[all_valid] = np.mean(normalized_precip[all_valid], axis=1)

        mean_elev = np.mean([s['elevation_ft'] for s in valid_stations])

        # Extract spatial mean from datasets
        print("Extracting spatial mean from CONUS404...")
        conus_series, conus_times = extract_conus_spatial_mean(start_date, end_date)
        if conus_series is not None:
            print(f"  CONUS404: {len(conus_series)} time steps")

        print("Extracting spatial mean from UCLA...")
        ucla_series, ucla_times = extract_ucla_spatial_mean(start_date, end_date)
        if ucla_series is not None:
            print(f"  UCLA: {len(ucla_series)} time steps")

        print("Extracting spatial mean from PNNL...")
        pnnl_series, pnnl_times = extract_pnnl_spatial_mean(start_date, end_date)
        if pnnl_series is not None:
            print(f"  PNNL: {len(pnnl_series)} time steps")

        # Create comparison plot
        print("Creating plot...")
        plot_mean_comparison(snotel_times, snotel_mean, mean_elev,
                            conus_series, conus_times,
                            ucla_series, ucla_times,
                            pnnl_series, pnnl_times,
                            event_date, ar_category)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
