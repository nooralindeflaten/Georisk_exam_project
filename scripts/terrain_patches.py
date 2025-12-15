from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
from tqdm import tqdm

# Function for paths
def default_raster_paths_for_region(region,base_dir="master/"):
    """
    Simple convention-based builder.
    Adjust names if your files differ.
    """
    raw_raster_dir = Path(base_dir+"raw/rasters")
    processed_raster_dir = Path(base_dir+"processed/rasters") 
    return {
        "dtm": raw_raster_dir / f"{region}_dtm10.tif",
        "slope": processed_raster_dir / f"{region}_slope_deg.tif",
        "aspect": processed_raster_dir / f"{region}_aspect_deg.tif",
        "curv": processed_raster_dir / f"{region}_curvature.tif",
        "flowacc": processed_raster_dir / f"{region}_flowacc_d8.tif",
        "twi": processed_raster_dir / f"{region}_twi.tif",
    }

def extract_stack_patches(self):
        """ 
        open rasters and extract patches around houses
        """
        rasters = default_raster_paths_for_region(self.region)
        keys = ["dtm", "slope", "aspect", "curv", "flowacc", "twi"]
        srcs = {k: rasterio.open(rasters[k]) for k in keys}
        
        dem_src = srcs["dtm"]
        res_x, res_y = dem_src.res
        assert abs(res_x - res_y) < 1e-6, "Rasters must have square pixels"
        res = res_x
        if self.houses_gdf.crs != dem_src.crs:
            self.houses_gdf = self.houses_gdf.to_crs(dem_src.crs)
        half_patch_size = int((self.patches_m / res) / 2)
        patch_pixels = half_patch_size * 2
        
        patches = []
        keep_rows = []
        
        for idx, row in tqdm(self.houses_gdf.iterrows(), total=len(self.houses_gdf), desc=f"{self.region} patches"):
            geometry = row.geometry
            x, y = geometry.x, geometry.y
            row_idx, col_idx = dem_src.index(x, y)
            
            window = Window(
                col_off=col_idx - half_patch_size,
                row_off=row_idx - half_patch_size,
                width=patch_pixels,
                height=patch_pixels
            )
            
            chans = []
            ok = True
            for k in keys:
                arr = srcs[k].read(1, window=window)
                if arr.shape != (patch_pixels, patch_pixels):
                    ok = False
                    break
                chans.append(arr)

            if not ok:
                continue

            stack = np.stack(chans, axis=0).astype("float32")
            
            stack = np.where(np.isfinite(stack), stack, np.nan)
            for c in range(stack.shape[0]):
                med = np.nanmedian(stack[c])
                stack[c] = np.where(np.isnan(stack[c]), med, stack[c])

            patches.append(stack)
            keep_rows.append(idx)

        # close rasters
        for s in srcs.values():
            s.close()

        houses_ok = self.houses_gdf.loc[keep_rows].copy()

        if len(patches) == 0:
            patches_t = torch.empty((0, len(keys), patch_pixels, patch_pixels), dtype=torch.float32)
        else:
            patches_t = torch.from_numpy(np.stack(patches, axis=0)).float()

        return houses_ok, patches_t
    

class TerrainBuilderStack:
    def __init__(self, region, raster_paths, houses_gdf,patches_m=200):
        self.region = region
        self.raster_paths = raster_paths  # dict: {"dtm": Path, ...}
        self.houses_gdf = houses_gdf
        self.patches_m = patches_m
        self._sources = None
        self.label_flood = "hazard_class_flom"
        self.label_landslide = "landslide_hazard_level"

    def _open_sources(self):
        if self._sources is None:
            keys = ["dtm", "slope", "aspect", "curv", "flowacc", "twi"]
            self._sources = {k: rasterio.open(self.raster_paths[k]) for k in keys}

            # ensure house CRS matches rasters
            dtm_crs = self._sources["dtm"].crs
            if self.houses_gdf.crs != dtm_crs:
                self.houses_gdf = self.houses_gdf.to_crs(dtm_crs)

        return self._sources

    def close(self):
        if self._sources:
            for src in self._sources.values():
                src.close()
        self._sources = None

    def extract_one_patch(self, geom):
        srcs = self._open_sources()
        dtm_src = srcs["dtm"]

        res_x, res_y = dtm_src.res
        assert abs(res_x - res_y) < 1e-6, "Rasters must have square pixels"
        res = float(res_x)

        half_pixels = int((self.patches_m / 2) / res)
        patch_pixels = half_pixels * 2

        row_idx, col_idx = dtm_src.index(geom.x, geom.y)

        window = Window(
            col_idx - half_pixels,
            row_idx - half_pixels,
            patch_pixels,
            patch_pixels,
        )

        keys = ["dtm", "slope", "aspect", "curv", "flowacc", "twi"]
        patches = [srcs[k].read(1, window=window) for k in keys]

        # Skip edge cases
        if any(p.shape != (patch_pixels, patch_pixels) for p in patches):
            return None

        stack = np.stack(patches, axis=0).astype("float32")

        stack = np.where(np.isfinite(stack), stack, np.nan)
        for c in range(stack.shape[0]):
            med = np.nanmedian(stack[c])
            if np.isnan(med):
                med = 0.0
            stack[c] = np.where(np.isnan(stack[c]), med, stack[c])

        return stack
    
    
