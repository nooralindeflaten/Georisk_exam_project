from pathlib import Path
import geopandas as gpd

KOMMUNE_AIO = Path("../../data/raw/vector/admin/admin.gpkg")
from pathlib import Path
import geopandas as gpd


def build_region_from_kommuner(region_name, kommune_names, out_path):
    """
    Build a single region polygon by dissolving selected kommuner
    from kommune_aio, and save as a GeoPackage.

    Parameters
    ----------
    region_name : str
        Name of the region, e.g. "sogn" or "lillehammer".
    kommune_names : list[str]
        List of kommune names that define the region,
        e.g. ["Årdal", "Lærdal", "Luster"].
    out_path : str or Path
        Output GPKG path, e.g. "master/regions/sogn_area.gpkg".
    """

    # Load all kommuner
    komm = gpd.read_file(KOMMUNE_AIO,layer='kommuner').to_crs(25833)

    name_cols = [c for c in komm.columns if "kommunenavn" in c.lower()]
    if not name_cols:
        raise RuntimeError(f"Could not find a kommune name column in {komm.columns}")
    name_col = name_cols[0]

    subset = komm[komm[name_col].isin(kommune_names)].copy()
    if subset.empty:
        raise RuntimeError(f"No kommuner matched {kommune_names} in column {name_col}")

    subset = subset.to_crs("EPSG:25833")

    # Dissolve to one polygon
    dissolved = subset.dissolve(by=None)
    region_geom = dissolved.geometry.iloc[0]

    # Wrap in a GeoDataFrame
    region_gdf = gpd.GeoDataFrame(
        {"region": [region_name]},
        geometry=[region_geom],
        crs="EPSG:25833",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    region_gdf.to_file(out_path, layer="region", driver="GPKG")
    print(f"Saved region '{region_name}' to {out_path}")
    
from pathlib import Path
import geopandas as gpd


def clip_layer_to_region(
    region_gpkg,
    src_gpkg,
    src_layer,
    out_path,
    region_layer="region",
    target_crs="EPSG:25833",
):
    """
    Clip a given layer to a region polygon and save as a new GeoPackage.

    Parameters
    ----------
    region_gpkg : str or Path
        Path to region gpkg (with a single 'region' polygon).
    src_gpkg : str or Path
        Path to source gpkg to be clipped.
    src_layer : str
        Layer name in the source gpkg (e.g. 'houses', 'stations').
    out_path : str or Path
        Output gpkg path for the clipped layer.
    region_layer : str, default "region"
        Layer name in region gpkg.
    target_crs : str, default "EPSG:25833"
        CRS to use for both region and source before clipping.
    """
    region_gpkg = Path(region_gpkg)
    src_gpkg = Path(src_gpkg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load region
    region = gpd.read_file(region_gpkg, layer=region_layer)
    if len(region) != 1:
        print(f"Warning: region file {region_gpkg} has {len(region)} features, using union.")
        region = region.dissolve(by=None)

    region = region.to_crs(target_crs)
    region_geom = region.geometry.iloc[0]

    # Load source layer
    src = gpd.read_file(src_gpkg, layer=src_layer).to_crs(target_crs)

    # Fast spatial filter using bbox first
    src = src[src.intersects(region_geom.buffer(0))]

    # Precise intersection
    clipped = gpd.overlay(src, region, how="intersection")
    if "region" in clipped.columns:
        pass

    clipped.to_file(out_path, driver="GPKG")
    print(f"Saved clipped '{src_layer}' to {out_path} with {len(clipped)} features.")
    
    
    
from pathlib import Path
import geopandas as gpd

BYGNINGSPUNKT_WFS_URL = "https://wfs.geonorge.no/skwms1/wfs.matrikkelen-bygningspunkt"  
BYGNINGSPUNKT_LAYER = "Bygning"  


def fetch_bygningspunkt_for_region(region_gpkg, out_path, region_layer="region"):
    """
    Fetch bygningspunkt (building points) from a WFS service clipped to a region polygon.

    Parameters
    ----------
    region_gpkg : str or Path
        Path to region gpkg with a single polygon.
    out_path : str or Path
        Output GPKG path for the clipped bygningspunkt.
    region_layer : str, default "region"
        Layer name in the region gpkg.
    """
    region_gpkg = Path(region_gpkg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    region = gpd.read_file(region_gpkg, layer=region_layer).to_crs("EPSG:25833")
    if len(region) != 1:
        region = region.dissolve(by=None)
    geom = region.geometry.iloc[0]

    minx, miny, maxx, maxy = geom.bounds

    
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typename": BYGNINGSPUNKT_LAYER,
        "srsName": "EPSG:25833",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:25833",
    }

    import urllib.parse

    url_with_params = BYGNINGSPUNKT_WFS_URL + "?" + urllib.parse.urlencode(params)
    gdf = gpd.read_file(url_with_params)

    gdf = gdf.to_crs("EPSG:25833")

    gdf = gpd.overlay(gdf, region, how="intersection")

    gdf.to_file(out_path, driver="GPKG")
    print(f"Saved {len(gdf)} bygningspunkt to {out_path}")


# Example usage:
fetch_bygningspunkt_for_region(
    region_gpkg="../kolbotn_test/kolbotn_area.gpkg",
    out_path="../kolbotn_test/houses/kolbotn_houses.gpkg"
)

# If i wanted to download as gml it's as easy as this, and a lot quicker
gdf = gpd.read_file("/Users/nooralindeflaten/Downloads/Basisdata_3207_Nordre_Follo_25833_MatrikkelenBygning_GML.gml",layer='Bygning').to_crs(25833)
gdf.to_file("../kolbotn_test/houses/kolbotn_houses_downloaded.gpkg", layer="kolbotn_houses", driver="GPKG")

from pathlib import Path
import subprocess

import geopandas as gpd
from shapely.geometry import box
import rasterio


CRS = "EPSG:25833"

# All your DTM10 tiles in one directory
TILES_DIR = Path("/Users/nooralindeflaten/Downloads/")

# Region polygons
REGIONS = {
    "kolbotn": "../kolbotn_test/kolbotn_area.gpkg",
}

OUT_DIR = Path("../kolbotn_test/raw/rasters")


def build_tile_index():
    """
    Create a GeoDataFrame with one row per tile:
      - path
      - geometry = tile extent (bbox)
    """
    import geopandas as gpd

    tifs = sorted(TILES_DIR.glob("*602_2_10m_z33.tif"))
    if not tifs:
        raise SystemExit(f"No .tif tiles found in {TILES_DIR}")

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


def build_region_dtm(region_name: str, region_path: str, tile_index):
    """
    For a given region:
      - find intersecting tiles from the index
      - mosaic them with gdalbuildvrt
      - clip mosaic with region polygon using gdalwarp -cutline
      - save <region>_dtm10.tif
    """
    print(f"\n=== {region_name}: building DTM10 from tiles ===")

    region = load_region(region_path, tile_index.crs)
    region_geom = region.geometry.iloc[0]

    # Find all tiles whose bbox intersects the region
    intersects = tile_index.intersects(region_geom.buffer(0))
    tiles_for_region = tile_index[intersects]

    if tiles_for_region.empty:
        print(f"  No tiles intersect region {region_name} – check CRS / extents.")
        return

    tile_paths = [str(p) for p in tiles_for_region["path"]]
    print(f"  Using {len(tile_paths)} tiles")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vrt_path = OUT_DIR / f"{region_name}_dtm10_tmp.vrt"
    out_tif = OUT_DIR / f"{region_name}_dtm10.tif"

    # 1) Build VRT mosaic
    print("  -> Building VRT mosaic...")
    cmd_vrt = ["gdalbuildvrt", str(vrt_path)] + tile_paths
    print("     ", " ".join(cmd_vrt))
    subprocess.check_call(cmd_vrt)

    # 2) Warp with cutline to exact region polygon
    print("  -> Clipping with gdalwarp -cutline...")
    cmd_warp = [
        "gdalwarp",
        "-cutline", region_path,
        "-cl", "region",
        "-crop_to_cutline",
        "-t_srs", CRS,
        "-of", "GTiff",
        "-dstnodata", "nan",
        str(vrt_path),
        str(out_tif),
    ]
    print("     ", " ".join(cmd_warp))
    subprocess.check_call(cmd_warp)

    print(f"  → Saved clipped DTM10 for {region_name} to {out_tif}")


def main_tile():
    tile_index = build_tile_index()
    for region_name, region_path in REGIONS.items():
        build_region_dtm(region_name, region_path, tile_index)


# Terrain
CRS = "EPSG:25833"
RES = 10
NODATA = -9999  # numeric nodata for richdem stability

def run(cmd):
    print("Running:", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)


def main_terrain():
    dtm_path = Path("../kolbotn_test/raw/rasters/kolbotn_dtm10.tif")
    dtm_path_clean = Path("../kolbotn_test/processed")
    dtm_path_clean.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        raw = dtm_path
        out = dtm_path_clean / f"{region}_dtm10_proc.tif"

        if not raw.exists():
            raise FileNotFoundError(raw)

        run([
            "gdalwarp",
            "-t_srs", CRS,
            "-tr", str(RES), str(RES),
            "-tap",
            "-r", "bilinear",
            "-dstnodata", str(NODATA),
            "-of", "GTiff",
            "-co", "COMPRESS=LZW",
            str(raw),
            str(out),
        ])

        print(f"✅ {region}: {out}")





import numpy as np
import rioxarray as rxr
import xarray as xr
import richdem as rd
OUT_DIR = Path("../kolbotn_test/processed")
DTM_DIR = Path("../kolbotn_test/raw/rasters")
REGIONS = ['kolbotn']
def clean_dem(dem: rd.rdarray) -> rd.rdarray:
    """
    richdem FlowAccumulation can crash if DEM contains NaNs/inf
    or inconsistent nodata. This function:

      - converts to float64
      - finds the minimum finite elevation
      - replaces any non-finite values with that min
      - sets no_data to that min
    """
    arr = np.array(dem, dtype="float64")

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        raise RuntimeError("DEM has no finite values. Check your input raster.")

    vmin = float(arr[finite_mask].min())

    bad = ~finite_mask
    if bad.any():
        print(f"  [clean_dem] Found {bad.sum()} non-finite cells; replacing with {vmin}")
        arr[bad] = vmin

    dem_clean = rd.rdarray(arr, no_data=vmin)
    dem_clean.geotransform = dem.geotransform
    dem_clean.projection = dem.projection
    return dem_clean

def read_dem(p):
    with rasterio.open(p) as src:
        arr = src.read(1).astype(np.float64)
        nd  = src.nodata
        tfm = src.transform
        prof = src.profile
    return arr, nd


def write_tif(out_path, arr, prof, nd):
    pr = prof.copy()
    pr.update(driver="GTiff", dtype="float64", count=1, nodata=nd,
              compress="LZW", tiled=True, blockxsize=64, blockysize=64)
    with rasterio.open(out_path, "w", **pr) as dst:
        dst.write(arr.astype(np.float64), 1)
    
def build_terrain_for_region(detm,region: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dtm10 = DTM_DIR / f"{region}_dtm10.tif"
    if not dtm10.exists():
        raise FileNotFoundError(f"Missing raw DTM for {region}: {dtm10}")

    out_filled   = OUT_DIR / f"{region}_dtm10_filled.tif"
    out_slope    = OUT_DIR / f"{region}_slope_deg.tif"
    out_flowacc  = OUT_DIR / f"{region}_flowacc_d8.tif"
    out_twi      = OUT_DIR / f"{region}_twi.tif"
    out_aspect   = OUT_DIR / f"{region}_aspect_deg.tif"
    out_curv     = OUT_DIR / f"{region}_curvature.tif"

    print(f"\n=== Processing region: {region} ===")
    print(f"Loading DEM from {dtm10} ...")
    dem_raw = rd.LoadGDAL(str(dtm10))

    print("Cleaning DEM (fix NaNs / nodata) ...")
    dem = clean_dem(dem_raw)
    print("Filling depressions (hydro-correct DEM) ...")
    dem_filled = rd.FillDepressions(dem, epsilon=False, in_place=False)

    print(f"Saving filled DEM to {out_filled} ...")
    rd.SaveGDAL(str(out_filled), dem_filled)

 #   dtm, nodata = read_dem(detm)
  #  print(f"Loading DEM from {dtm} ...")
   # dem = rd.rdarray(dtm,no_data=nodata)
    #print("Filling depressions (hydro-correct DEM) ...")
    # Make a filled copy so original stays untouched
    #dem_filled = rd.FillDepressions(dem, epsilon=False, in_place=False)

    print(f"Saving filled DEM to {out_filled} ...")
    rd.SaveGDAL(str(out_filled), dem_filled)


    print("Computing slope (degrees) ...")
    slope_deg = rd.TerrainAttribute(dem_filled, attrib="slope_degrees")

    print(f"Saving slope raster to {out_slope} ...")
    rd.SaveGDAL(str(out_slope), slope_deg)

   
    print("Computing flow accumulation (D8) ...")
    flowacc = rd.FlowAccumulation(dem_filled, method="D8")

    print(f"Saving flow accumulation raster to {out_flowacc} ...")
    rd.SaveGDAL(str(out_flowacc), flowacc)

    
    print("Computing TWI ...")

    # slope in radians
    slope_rad = rd.TerrainAttribute(dem_filled, attrib="slope_radians")

    # cellsize (assume square cells)
    gt = dem_filled.geotransform
    cellsize = float(gt[1])

    # flowacc is an rdarray; convert to numpy array for math
    fa = np.array(flowacc, dtype="float64")
    sr = np.array(slope_rad, dtype="float64")

    eps = 1e-6

    # contributing area per unit contour length (m)
    # (fa+1) so isolated cells have some area
    As = (fa + 1.0) * cellsize

    # TWI formula
    twi_vals = np.log((As + eps) / (np.tan(sr) + eps))

    twi_vals = np.where(np.isfinite(twi_vals), twi_vals, np.nan)

    twi = rd.rdarray(twi_vals, no_data=np.nan)
    twi.geotransform = dem_filled.geotransform
    twi.projection = dem_filled.projection

    print(f"Saving TWI raster to {out_twi} ...")
    rd.SaveGDAL(str(out_twi), twi)

    print("Done!")


def main_build_terrain():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        dtm_path = DTM_DIR / f"{region}_dtm10.tif"
        build_terrain_for_region(dtm_path,region)


def write_raster(template_path: Path, arr: np.ndarray, out_path: Path):
    with rasterio.open(template_path) as src:
        profile = src.profile

    profile.update(
        dtype="float64",
        nodata=np.nan,
        count=1,
        compress="LZW",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype("float64"), 1)

    print("Wrote", out_path)


def clean(arr: np.ndarray) -> np.ndarray:
    """
    Replace non-finite with NaN, then fill NaNs with median.
    Keeps output stable for ML and visualization.
    """
    arr = np.array(arr, dtype="float64")
    arr = np.where(np.isfinite(arr), arr, np.nan)

    med = np.nanmedian(arr)
    if np.isnan(med):
        # worst-case fallback
        med = 0.0

    arr = np.where(np.isnan(arr), med, arr)
    return arr.astype("float64")


def build_region(region: str):
    dtm_filled = OUT_DIR / f"{region}_dtm10_filled.tif"
    if region == "oslo_2": # I made oslo_2 because I was testing both dtm10 and dtm1
        region = "oslo"
    out_aspect = OUT_DIR / f"{region}_aspect_deg.tif"
    out_curv = OUT_DIR / f"{region}_curvature.tif"

    if not dtm_filled.exists():
        raise FileNotFoundError(f"Missing filled DTM for {region}: {dtm_filled}")

    print(f"\n=== {region.upper()} ===")
    print("Loading filled DEM from", dtm_filled)

    with rasterio.open(dtm_filled) as src:
        dem = src.read(1).astype("float64")
        nodata = src.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    # richdem object
    rd_dem = rd.rdarray(dem, no_data=np.nan)

    print("Computing aspect (degrees)...")
    aspect = rd.TerrainAttribute(rd_dem, attrib="aspect")  # 0–360

    print("Computing curvature...")
    curvature = rd.TerrainAttribute(rd_dem, attrib="curvature")

    aspect = clean(aspect)
    curvature = clean(curvature)

    write_raster(dtm_filled, aspect, out_aspect)
    write_raster(dtm_filled, curvature, out_curv)

    print("Done.")


def main_curve_asp():
    for region in REGIONS:
        build_region(region)


import urllib.parse
REGIONS = {
    "kolbotn": "../kolbotn_test/kolbotn_area.gpkg",
}

HYDRO_LAYERS = [
    {
        "name": "vannforing_stasjoner",
        "url": "https://nve.geodataonline.no/arcgis/rest/services/HydrologiskeData3/MapServer/0/",
        "geometry_type": "POINT",  # vannføringsstasjoner
    },
    {
        "name": "rivers",
        "url": "https://kart.nve.no/enterprise/rest/services/Elvenett1/MapServer/2/",
        "geometry_type": "MULTILINESTRING",  # elvenett
    },
    {
        "name": "lakes",
        "url": "https://nve.geodataonline.no/arcgis/rest/services/Innsjodatabase2/MapServer/5/",
        "geometry_type": "MULTIPOLYGON",  # innsjødatabase
    },
    {
        "name": "catchments",
        "url": "https://nve.geodataonline.no/arcgis/rest/services/Nedborfelt2/MapServer/1/",
        "geometry_type": "MULTIPOLYGON",  # nedbørfelt
    },
]

HAZARD_LAYERS = [{
    "url": "https://nve.geodataonline.no/arcgis/rest/services/Flomsoner2/MapServer",
    "name": {
        0:"flomsone_analyseomrade",
        13:"flomsone_10",
        14:"flomsone_20",
        15:"flomsone_50",
        16:"flomsone_100",
        17:"flomsone_200",
        18:"flomsone_500",
        19:"flomsone_1000",
        20:"flomsone_20_klima",
        21:"flomsone_200_klima",
        22:"flomsone_1000_klima",
    },
    "geometry_type": "MULTIPOLYGON"
}, {
    "url": "https://nve.geodataonline.no/arcgis/rest/services/SkredHendelser1/MapServer",
    "name": {0:"landslides"},
    "geometry_type": "POINT"
}]


def get_region_bbox(region_gpkg, region_layer="region", crs=CRS):
    region = gpd.read_file(region_gpkg, layer=region_layer)
    if len(region) != 1:
        region = region.dissolve(by=None)
    region = region.to_crs(crs)
    minx, miny, maxx, maxy = region.total_bounds
    return minx, miny, maxx, maxy


def build_arcgis_query_url(layer_url, bbox_25833):
    """
    Build an ArcGIS REST GetFeature (GeoJSON) URL for a given layer + bbox.

    layer_url is already on the form:
      https://.../MapServer/<layer>/
    """
    minx, miny, maxx, maxy = bbox_25833
    bbox_param = f"{minx},{miny},{maxx},{maxy},25833"

    params = {
        "where": "1=1",
        "geometry": bbox_param,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "25833",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "25833",
        "f": "geojson",
    }

    return layer_url.rstrip("/") + "/query?" + urllib.parse.urlencode(params)


def fetch_layer_for_region(
    region_name: str,
    region_gpkg: str,
    out_gpkg: Path,
    layer_name: str,
    layer_url: str,
    geometry_type: str,
    region_layer: str = "region",
    crs: str = CRS,
):
    """
    Use ogr2ogr to fetch a single ArcGIS MapServer layer for a region
    and write/append it into <region>_hydro.gpkg as layer_name.
    """
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    bbox = get_region_bbox(region_gpkg, region_layer=region_layer, crs=crs)
    url = build_arcgis_query_url(layer_url, bbox)

    print(f"[{region_name}] {layer_name}")
    print(f"  URL: {url}")

    # FIRST layer in this file: use -overwrite
    # Later layers: use -update so they are added to the same GPKG.
    if out_gpkg.exists():
        action_flag = "-update"
    else:
        action_flag = "-overwrite"

    cmd = [
        "ogr2ogr",
        "-f", "GPKG",
        action_flag,
        str(out_gpkg),
        url,
        "-nln", layer_name,
        "-t_srs", crs,
        "-clipsrc", str(region_gpkg),
        "-clipsrclayer", region_layer,
        "-nlt", geometry_type,
    ]

    print("  Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"  → wrote layer '{layer_name}' to {out_gpkg}\n")



def build_all_region_hydro():
    """
    For each region (sogn, lillehammer, oslo) we have hydro layers
    Vannførings stasjoner, catchments, rivers and lakes. 
    """
    for region_name, region_gpkg in REGIONS.items():
        out_gpkg = Path(f"../kolbotn_test/raw/vector/hydro/{region_name}_hydro.gpkg")

        # Start fresh each run so we don't accumulate old stuff
        if out_gpkg.exists():
            print(f"[{region_name}] Removing existing {out_gpkg}")
            out_gpkg.unlink()

        print(f"\n=== Building hydro for region: {region_name} ===")
        for layer_cfg in HYDRO_LAYERS:
            fetch_layer_for_region(
                region_name=region_name,
                region_gpkg=region_gpkg,
                out_gpkg=out_gpkg,
                layer_name=layer_cfg["name"],
                layer_url=layer_cfg["url"],
                geometry_type=layer_cfg["geometry_type"],
            )

        print(f"=== Done {region_name}, hydro at {out_gpkg} ===\n")

def build_all_region_hazard():
    """
    For each region I've created hazard layers
    """
    for region_name, region_gpkg in REGIONS.items():
        out_gpkg = Path(f"../kolbotn_test/raw/vector/hazards/{region_name}_hazards.gpkg")

        if out_gpkg.exists():
            print(f"[{region_name}] Removing existing {out_gpkg}")
            out_gpkg.unlink()

        print(f"\n=== Building hydro for region: {region_name} ===")
        for layer_cfg in HAZARD_LAYERS:
            layer_url = layer_cfg["url"]
            geometry_type = layer_cfg["geometry_type"]
            for layer_num, layer_name in layer_cfg["name"].items():
                l_url = layer_url + "/" + str(layer_num) + "/"
                fetch_layer_for_region(
                    region_name=region_name,
                    region_gpkg=region_gpkg,
                    out_gpkg=out_gpkg,
                    layer_name=layer_name,
                    layer_url=l_url,
                    geometry_type=geometry_type,
                )

        print(f"=== Done {region_name}, hydro at {out_gpkg} ===\n")

