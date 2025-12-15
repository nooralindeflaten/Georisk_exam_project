from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box


CRS = "EPSG:25833"

# All tiles in one directory
TILES_DIR = Path("../Downloads/Nedlastingspakke")

# Your region polygons
REGIONS = {
    "sogn": "master/regions/sogn_area.gpkg",
    "lillehammer": "master/regions/lillehammer_area.gpkg",
    "oslo": "master/regions/oslo_area.gpkg",
}

OUT_DIR = Path("master/raw/rasters")


def build_tile_index():
    """
    Create a GeoDataFrame with one row per tile:
      - path
      - geometry = bounding box
    """
    tifs = sorted(TILES_DIR.glob("*.tif"))
    if not tifs:
        raise SystemExit(f"No .tif files found in {TILES_DIR}")

    records = []
    crs = None

    for p in tifs:
        with rasterio.open(p) as src:
            if crs is None:
                crs = src.crs
            bounds = src.bounds
            geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            records.append({"path": str(p), "geometry": geom})

    gdf = gpd.GeoDataFrame(records, crs=crs)
    print(f"Indexed {len(gdf)} tiles, CRS={gdf.crs}")
    return gdf


def load_region(region_path: str, crs):
    region = gpd.read_file(region_path, layer="region")
    if len(region) != 1:
        region = region.dissolve(by=None)
    region = region.to_crs(crs)
    return region


def build_region_dtm(region_name: str, region_path: str, tile_index: gpd.GeoDataFrame):
    """
    For a given region:
      - find intersecting tiles
      - mosaic them
      - clip to region polygon
      - save <region>_dtm10.tif
    """
    print(f"\n=== {region_name}: building DTM10 ===")

    # Load region polygon in same CRS as tiles
    region = load_region(region_path, tile_index.crs)
    region_geom = [region.geometry.iloc[0]]  # as list for rasterio.mask

    # Find tiles whose bbox intersects the region
    intersects_mask = tile_index.intersects(region_geom[0].buffer(0))
    tiles_for_region = tile_index[intersects_mask]

    if tiles_for_region.empty:
        print(f"  No tiles intersect region {region_name} – check your areas/CRS.")
        return

    print(f"  Using {len(tiles_for_region)} tiles:")

    srcs = []
    for path_str in tiles_for_region["path"]:
        print(f"    - {path_str}")
        srcs.append(rasterio.open(path_str))

    # Mosaic tiles
    mosaic, transform = merge(srcs)
    src_crs = srcs[0].crs
    for s in srcs:
        s.close()

    # Clip mosaic to exact region polygon
    # mosaic has shape (bands, h, w); we assume single band DTM
    cropped, cropped_transform = mask(
        rasterio.io.MemoryFile().open(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=1,
            dtype=mosaic.dtype,
            crs=src_crs,
            transform=transform,
        ),
        region_geom,
        crop=True,
        filled=True,
        nodata=np.nan,
    )

    out_path = OUT_DIR / f"{region_name}_dtm10.tif"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_meta = {
        "driver": "GTiff",
        "height": cropped.shape[1],
        "width": cropped.shape[2],
        "count": 1,
        "dtype": cropped.dtype,
        "crs": src_crs,
        "transform": cropped_transform,
        "compress": "LZW",
        "nodata": np.nan,
    }

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(cropped[0, :, :], 1)

    print(f"  → Saved clipped DTM10 for {region_name} to {out_path}")


def main():
    tile_index = build_tile_index()

    for region_name, region_path in REGIONS.items():
        build_region_dtm(region_name, region_path, tile_index)


if __name__ == "__main__":
    main()
