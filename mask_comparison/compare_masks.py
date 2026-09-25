"""
Compare HUC8-based masking vs boundary-based masking for the Skagit Basin.

This script visualizes and quantifies the differences between:
1. HUC8 sub-basin masking (Upper Skagit, Sauk, Lower Skagit)
2. Boundary polygon masking (SkagitBoundary.json)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import regionmask

# Configuration
BASE_DIR = "/data0/hernanqd/plots_code/skagit_basin_de"
VAULT_DIR = "/data0/skagit_met/data_transfer/data"

HUC8_GEO = os.path.join(BASE_DIR, "data/GIS/SkagitSubBasin_HUC8.geojson")
BOUNDARY_PATH = "/data0/nksp2/skagit/skagit_2/skagit-met/data/GIS/SkagitBoundary.json"

# Grid files for testing
pnnl_grid_file = os.path.join(VAULT_DIR, "PNNL/SERDP6km.geo_em.d01.nc")
ucla_coords_file = os.path.join(VAULT_DIR, "ucla_era5_d02_daily/static/wrfinput_d02_coord.nc")

OUTPUT_DIR = os.path.join(BASE_DIR, "mask_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_huc8_mask(gdf, lon, lat):
    """Create HUC8 sub-basin mask (same as other scripts)"""
    mask_3d = regionmask.mask_3D_geopandas(gdf, lon, lat)
    region_indices = []
    for r in ["Upper Skagit", "Sauk", "Lower Skagit"]:
        idx = gdf[gdf["Name"].str.contains(r, case=False)].index[0]
        region_indices.append(idx)
    return mask_3d.sel(region=region_indices).any(dim='region').values


def get_boundary_mask(lon, lat, boundary_poly):
    """Create boundary polygon mask (same as create_ar_events_plots.py)"""
    df_pts = pd.DataFrame({'lon': lon.flatten(), 'lat': lat.flatten()})
    gdf_pts = gpd.GeoDataFrame(
        df_pts,
        geometry=gpd.points_from_xy(df_pts.lon, df_pts.lat),
        crs="EPSG:4326"
    )
    inside = gdf_pts.intersects(boundary_poly).values
    return inside.reshape(lon.shape)


def compare_masks(grid_name, lon, lat, gdf, boundary_poly):
    """Compare masks for a specific grid"""
    print(f"\n{'='*70}")
    print(f"Comparing masks for {grid_name}")
    print(f"{'='*70}")
    print(f"Grid shape: {lon.shape}")

    # Create both masks
    mask_huc8 = get_huc8_mask(gdf, lon, lat)
    mask_boundary = get_boundary_mask(lon, lat, boundary_poly)

    # Statistics
    huc8_cells = np.sum(mask_huc8 > 0)
    boundary_cells = np.sum(mask_boundary > 0)
    both_cells = np.sum((mask_huc8 > 0) & (mask_boundary > 0))
    only_huc8 = np.sum((mask_huc8 > 0) & ~mask_boundary)
    only_boundary = np.sum((mask_boundary > 0) & ~(mask_huc8 > 0))

    print(f"\nMask Statistics:")
    print(f"  HUC8 masked cells:        {huc8_cells:,}")
    print(f"  Boundary masked cells:    {boundary_cells:,}")
    print(f"  Both masks overlap:       {both_cells:,}")
    print(f"  Only in HUC8:             {only_huc8:,} ({100*only_huc8/huc8_cells:.1f}%)")
    print(f"  Only in Boundary:         {only_boundary:,} ({100*only_boundary/boundary_cells:.1f}%)")
    print(f"  Intersection/HUC8:        {100*both_cells/huc8_cells:.1f}%")
    print(f"  Intersection/Boundary:    {100*both_cells/boundary_cells:.1f}%")

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # HUC8 mask
    ax = axes[0, 0]
    im1 = ax.imshow(mask_huc8, cmap='Blues', origin='lower')
    ax.set_title('HUC8 Sub-basin Mask', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im1, ax=ax, label='Masked')

    # Boundary mask
    ax = axes[0, 1]
    im2 = ax.imshow(mask_boundary, cmap='Greens', origin='lower')
    ax.set_title('Boundary Polygon Mask', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax, label='Masked')

    # Difference: create a color-coded mask
    # 0 = unmasked, 1 = HUC8 only, 2 = Boundary only, 3 = Both
    diff_mask = np.zeros_like(mask_huc8, dtype=int)
    diff_mask[(mask_huc8 > 0)] = 1
    diff_mask[(mask_boundary > 0)] += 2

    ax = axes[1, 0]
    im3 = ax.imshow(diff_mask, cmap='RdYlBu_r', origin='lower', vmin=0, vmax=3)
    ax.set_title('Mask Comparison (Red=HUC8 only, Blue=Boundary only, Purple=Both)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    cbar = plt.colorbar(im3, ax=ax, label='Mask type', ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['None', 'HUC8', 'Boundary', 'Both'])

    # Overlay both masks
    ax = axes[1, 1]
    ax.imshow(mask_huc8.astype(float), cmap='Blues', alpha=0.5, origin='lower', label='HUC8')
    ax.imshow(mask_boundary.astype(float), cmap='Greens', alpha=0.5, origin='lower', label='Boundary')
    ax.set_title('Mask Overlay (Blue=HUC8, Green=Boundary)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, f'mask_comparison_{grid_name}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {output_file}")
    plt.close()

    return {
        'grid': grid_name,
        'huc8_cells': huc8_cells,
        'boundary_cells': boundary_cells,
        'overlap_cells': both_cells,
        'only_huc8': only_huc8,
        'only_boundary': only_boundary,
        'huc8_coverage': 100 * both_cells / huc8_cells,
        'boundary_coverage': 100 * both_cells / boundary_cells
    }


def plot_all_huc8_regions(gdf_huc8):
    """Plot all HUC8 regions with their names"""
    print("\nCreating full HUC8 regions plot...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # Plot all HUC8 regions
    gdf_huc8_plot = gdf_huc8.to_crs("EPSG:4326")

    # Highlight the three regions being used
    regions_to_highlight = ["Upper Skagit", "Sauk", "Lower Skagit"]
    colors_map = {
        "Upper Skagit": "#1f77b4",  # Blue
        "Sauk": "#ff7f0e",           # Orange
        "Lower Skagit": "#2ca02c"    # Green
    }

    # Plot all regions with light color
    for idx, (i, row) in enumerate(gdf_huc8_plot.iterrows()):
        geom = row.geometry
        region_name = row['Name'] if 'Name' in row else f'Region {idx}'

        # Check if this is one of the three being used
        is_highlighted = any(region in region_name for region in regions_to_highlight)

        if is_highlighted:
            color = colors_map.get(next((r for r in regions_to_highlight if r in region_name), region_name), "#333333")
            edge_color = color
            edge_width = 2.5
            face_color = color
            alpha = 0.3
        else:
            color = '#cccccc'
            edge_color = '#999999'
            edge_width = 0.8
            face_color = '#f0f0f0'
            alpha = 0.3

        # Handle both Polygon and MultiPolygon
        if geom.geom_type == 'Polygon':
            from matplotlib.patches import Polygon as MPLPolygon
            patch = MPLPolygon(geom.exterior.coords, facecolor=face_color, edgecolor=edge_color,
                             linewidth=edge_width, alpha=alpha)
            ax.add_patch(patch)
        elif geom.geom_type == 'MultiPolygon':
            from matplotlib.patches import Polygon as MPLPolygon
            for poly in geom.geoms:
                patch = MPLPolygon(poly.exterior.coords, facecolor=face_color, edgecolor=edge_color,
                                 linewidth=edge_width, alpha=alpha)
                ax.add_patch(patch)

        # Add region name as text at centroid
        try:
            centroid = geom.centroid
            fontsize = 10 if is_highlighted else 8
            fontweight = 'bold' if is_highlighted else 'normal'
            ax.text(centroid.x, centroid.y, region_name,
                    fontsize=fontsize, fontweight=fontweight, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))
        except:
            pass

    # Set axis limits to data extent
    bounds = gdf_huc8_plot.total_bounds
    ax.set_xlim(bounds[0] - 0.05, bounds[2] + 0.05)
    ax.set_ylim(bounds[1] - 0.05, bounds[3] + 0.05)

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title(f'All HUC8 Sub-basins ({len(gdf_huc8)} total) - Colored: Upper Skagit, Sauk, Lower Skagit',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='#1f77b4', label='Upper Skagit (highlighted)'),
        Patch(facecolor='#ff7f0e', edgecolor='#ff7f0e', label='Sauk (highlighted)'),
        Patch(facecolor='#2ca02c', edgecolor='#2ca02c', label='Lower Skagit (highlighted)'),
        Patch(facecolor='#f0f0f0', edgecolor='#999999', label='Other HUC8 regions')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='best', framealpha=0.95)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'all_huc8_regions.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ All HUC8 regions plot saved: {output_file}")
    plt.close()


def plot_shapes_comparison(gdf_huc8, gdf_boundary):
    """Plot the three HUC8 sub-basins (Upper Skagit, Sauk, Lower Skagit) and boundary polygon"""
    print("\nCreating shapefile comparison plot...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Filter to only the three regions being used
    regions_to_plot = ["Upper Skagit", "Sauk", "Lower Skagit"]
    colors_map = {
        "Upper Skagit": "#1f77b4",  # Blue
        "Sauk": "#ff7f0e",           # Orange
        "Lower Skagit": "#2ca02c"    # Green
    }

    gdf_huc8_plot = gdf_huc8.to_crs("EPSG:4326")

    for region_name in regions_to_plot:
        # Find matching regions (case-insensitive substring match)
        mask = gdf_huc8_plot["Name"].str.contains(region_name, case=False, na=False)
        matching_rows = gdf_huc8_plot[mask]

        if len(matching_rows) == 0:
            print(f"  Warning: No region matching '{region_name}' found")
            continue

        color = colors_map.get(region_name, "#333333")

        for idx, (i, row) in enumerate(matching_rows.iterrows()):
            geom = row.geometry

            # Handle both Polygon and MultiPolygon
            if geom.geom_type == 'Polygon':
                ax.plot(*geom.exterior.xy, linewidth=3, color=color, alpha=0.9, label=region_name)
            elif geom.geom_type == 'MultiPolygon':
                for poly_idx, poly in enumerate(geom.geoms):
                    label = region_name if poly_idx == 0 else None
                    ax.plot(*poly.exterior.xy, linewidth=3, color=color, alpha=0.9, label=label)

            # Add region name as text at centroid
            try:
                centroid = geom.centroid
                ax.text(centroid.x, centroid.y, region_name,
                        fontsize=11, fontweight='bold', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2))
            except:
                pass

    # Plot boundary polygon
    gdf_boundary_plot = gdf_boundary.to_crs("EPSG:4326")
    gdf_boundary_plot.plot(ax=ax, facecolor='none', edgecolor='black',
                          linewidth=3.5, linestyle='--', label='Watershed Boundary', zorder=10)

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('HUC8 Sub-basins (Upper Skagit, Sauk, Lower Skagit) vs Watershed Boundary',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, 'shapes_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Shapes comparison plot saved: {output_file}")
    plt.close()


def main():
    print("Loading GIS data...")

    # Load HUC8 regions
    gdf_huc8 = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    print(f"HUC8 regions loaded: {len(gdf_huc8)} regions")

    # Load boundary polygon
    gdf_boundary = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
    if hasattr(gdf_boundary, 'union_all'):
        boundary_poly = gdf_boundary.union_all()
    else:
        boundary_poly = gdf_boundary.unary_union
    print(f"Boundary polygon loaded")

    # Plot all HUC8 regions for context
    plot_all_huc8_regions(gdf_huc8)

    # Plot shapes comparison (three regions only)
    plot_shapes_comparison(gdf_huc8, gdf_boundary)

    results = []

    # Compare PNNL
    if os.path.exists(pnnl_grid_file):
        print(f"\nLoading PNNL grid...")
        ds_pnnl = xr.open_dataset(pnnl_grid_file)
        pnnl_lon = ds_pnnl.XLONG_M.values[0]
        pnnl_lat = ds_pnnl.XLAT_M.values[0]
        ds_pnnl.close()
        result = compare_masks('PNNL', pnnl_lon, pnnl_lat, gdf_huc8, boundary_poly)
        results.append(result)
    else:
        print(f"PNNL grid file not found: {pnnl_grid_file}")

    # Compare UCLA
    if os.path.exists(ucla_coords_file):
        print(f"\nLoading UCLA grid...")
        ds_ucla = xr.open_dataset(ucla_coords_file)
        ucla_lon = ds_ucla.lon2d.values
        ucla_lat = ds_ucla.lat2d.values
        ds_ucla.close()
        result = compare_masks('UCLA', ucla_lon, ucla_lat, gdf_huc8, boundary_poly)
        results.append(result)
    else:
        print(f"UCLA grid file not found: {ucla_coords_file}")

    # Summary table
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print(f"{'='*70}")
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))

        # Save summary to CSV
        csv_path = os.path.join(OUTPUT_DIR, 'mask_comparison_summary.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"\n✓ Summary saved: {csv_path}")


if __name__ == "__main__":
    main()
