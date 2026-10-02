# Comparison Analysis: Your Script vs. Niteesh's Script

## Summary of Findings

The **273 mm difference in Lower Skagit PRISM data** is likely caused by differences in region boundary definitions, not the data itself.

---

## Key Differences

### 1. **Region Definition & Clipping** ⚠️ CRITICAL

**Your Script (`water_year_cumulative.py`):**
```python
def load_regions():
    gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    return gdf
```
- Loads HUC8 regions directly from GeoJSON
- **Does NOT clip to Skagit boundary**
- Uses raw HUC8 polygon boundaries

**Niteesh's Script (`generating_timeseries.py`):**
```python
def load_regions_gdf():
    skagit_gdf = gpd.read_file(BOUNDARY_GEO).to_crs("EPSG:4326")
    huc8 = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    
    # KEY STEP: Clip HUC8 to Skagit boundary
    huc8_3 = gpd.overlay(huc8_3, skagit_gdf, how="intersection")
```
- Loads HUC8 regions **AND** Skagit boundary
- **Clips each HUC8 region to the Skagit boundary** using `overlay(how="intersection")`
- Only includes area within the main Skagit boundary

### 2. **Impact on Lower Skagit**

The Lower Skagit HUC8 region likely extends beyond the main Skagit Basin boundary. When you include extra area outside the boundary:
- **Your script**: Larger area → Higher total precipitation
- **Niteesh's script**: Clipped area → Lower total precipitation

This explains the **~273 mm difference** (9.36% less in Niteesh's version).

---

## Other Differences

### 3. **PRISM Data Source**

**Your Script:**
```python
prism_root = os.path.join(VAULT_DIR, "prism_new_hq")  # Individual TIFF files
```

**Niteesh's Script:**
```python
P_PRISM_DIR = BASE / "prism_ppt"  # Pre-processed Zarr files
```

Different source formats, but both should be the same underlying data.

### 4. **Region Masking Sophistication**

**Your Script:**
```python
mask = regionmask.mask_3D_geopandas(gdf, lon, lat)
```

**Niteesh's Script:**
```python
regs = regionmask.Regions(
    outlines=list(regions_gdf.geometry.values),
    names=list(regions_gdf["Name"].astype(str).values),
    overlap=True,  # Allows overlapping regions
)
```

Niteesh's approach allows overlapping regions (Full Skagit overlaps with subbasins), yours doesn't.

---

## Recommendation to Fix

To match Niteesh's approach, update your `load_regions()` function:

```python
def load_regions():
    """Load HUC8 regions clipped to Skagit boundary"""
    import geopandas as gpd
    
    # Load Skagit boundary
    boundary_geo = os.path.join(BASE_DIR, "data/GIS/SkagitBoundary.json")
    skagit_gdf = gpd.read_file(boundary_geo).to_crs("EPSG:4326")
    
    # Load HUC8 regions
    huc8_gdf = gpd.read_file(HUC8_GEO).to_crs("EPSG:4326")
    
    # Clip HUC8 to Skagit boundary
    sub_names = ["Upper Skagit", "Sauk", "Lower Skagit"]
    huc8_sub = huc8_gdf[huc8_gdf["Name"].isin(sub_names)].copy()
    huc8_clipped = gpd.overlay(huc8_sub, skagit_gdf, how="intersection")
    
    # Add full Skagit
    full_row = gpd.GeoDataFrame(
        {"Name": ["Full Skagit"], "geometry": [skagit_gdf.geometry.iloc[0]]},
        crs="EPSG:4326"
    )
    
    regions_gdf = pd.concat([huc8_clipped, full_row], ignore_index=True)
    return regions_gdf
```

---

## Summary Table

| Aspect | Your Script | Niteesh's Script |
|--------|------------|-----------------|
| Region clipping | ❌ No | ✅ Yes (to boundary) |
| PRISM source | TIFF (new_hq) | Zarr (ppt) |
| Region masking | Basic | Sophisticated (overlap=True) |
| Lower Skagit area | **Larger** | **Smaller (clipped)** |
| Expected difference | ~+273 mm | Baseline |

The **~273 mm difference in Lower Skagit is not a data error** — it's a region definition difference. Niteesh's version is more accurate because it respects the Skagit Basin boundary.
