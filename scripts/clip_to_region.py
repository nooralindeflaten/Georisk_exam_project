from pathlib import Path
import geopandas as gpd

KOMMUNE_AIO = Path("data/raw/vector/admin/admin.gpkg")  # adjust path/name
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

    dissolved = subset.dissolve(by=None)
    region_geom = dissolved.geometry.iloc[0]

    region_gdf = gpd.GeoDataFrame(
        {"region": [region_name]},
        geometry=[region_geom],
        crs="EPSG:25833",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    region_gdf.to_file(out_path, layer="region", driver="GPKG")
    print(f"Saved region '{region_name}' to {out_path}")
    
def region_example_usage():
    build_region_from_kommuner(
    region_name="sogn",
    kommune_names=["Årdal", "Lærdal", "Luster"],
    out_path="master/regions/sogn_area.gpkg",
    )

    build_region_from_kommuner(
        region_name="lillehammer",
        kommune_names=["Lillehammer"],
        out_path="master/regions/lillehammer_area.gpkg",
    )

    build_region_from_kommuner(
        region_name="oslo",
        kommune_names=["Oslo", "Nordre Follo"],
        out_path="master/regions/oslo_area.gpkg",
    )

import subprocess
from pathlib import Path
import urllib.parse

import geopandas as gpd
regions = {
    "sogn": "master/regions/sogn_area.gpkg",
    "lillehammer": "master/regions/lillehammer_area.gpkg",
    "oslo": "master/regions/oslo_area.gpkg",
}

HYDRO_SERVICE = "https://nve.geodataonline.no/arcgis/rest/services/HydrologiskeData3/MapServer/0/"
RIVERS = "https://kart.nve.no/enterprise/rest/services/Elvenett1/MapServer/2/"
LAKES = "https://nve.geodataonline.no/arcgis/rest/services/Innsjodatabase2/MapServer/5/"
CATCH = "https://nve.geodataonline.no/arcgis/rest/services/Nedborfelt2/MapServer/1/"

import subprocess
from pathlib import Path
import urllib.parse

import geopandas as gpd


CRS = "EPSG:25833"


REGIONS = {
    "sogn": "master/regions/sogn_area.gpkg",
    "lillehammer": "master/regions/lillehammer_area.gpkg",
    "oslo": "master/regions/oslo_area.gpkg",
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


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# main driver: loops regions × layers
# ----------------------------------------------------------------------
def build_all_region_hydro():
    """
    For each region (sogn, lillehammer, oslo):
      - create master/raw/vector/hydro/<region>_hydro.gpkg
      - populate it with layers: stations, rivers, lakes, catchments
        according to HYDRO_LAYERS config.
    """
    for region_name, region_gpkg in REGIONS.items():
        out_gpkg = Path(f"master/raw/vector/hydro/{region_name}_hydro.gpkg")

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



import urllib.parse
from pathlib import Path
import subprocess

import geopandas as gpd
from shapely.geometry import box
import rasterio

REGIONS = {
    "oslo": "../regions/oslo_area.gpkg",
    "lillehammer": "../regions/lillehammer_area.gpkg",
    "sogn": "../regions/sogn_area.gpkg",
}

CRS = "EPSG:25833"
RES = 10
NODATA = -9999  # numeric nodata for richdem stability


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


HAZARD_LAYERS = [
    {
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
    },
    {
        "url": "https://nve.geodataonline.no/arcgis/rest/services/SkredHendelser1/MapServer",
        "layers": [
            {"num": 0, "name": "landslides", "geometry_type": "POINT"},
            {"num": 1, "name": "landslide_trigger_points", "geometry_type": "POINT"},
            {"num": 2, "name": "landslide_runout_points", "geometry_type": "POINT"},
            {"num": 3, "name": "landslide_source_areas", "geometry_type": "MULTIPOLYGON"},
            {"num": 4, "name": "landslide_runout_areas", "geometry_type": "MULTIPOLYGON"},
        ]
    }
]



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
        out_gpkg = Path(f"../raw/vector/hazards/{region_name}_hazards.gpkg")

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

