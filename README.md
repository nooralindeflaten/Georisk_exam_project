# GeoRisk — Flood & Landslide Susceptibility at Building Level (Norway)

Geospatial hazard modelling project built for **Applied Machine Learning (Noroff)**.  
Goal: predict **building-level flood hazard classes** (and prototype **landslide susceptibility**) by combining:

- **Terrain rasters** (DTM-derived)
- **Hydrology / catchment context** (distance-to-water + discharge normals + catchment attributes)
- **Spatial neighborhood context** (KNN graph between nearby buildings)

This repo is **model-focused**. It intentionally does **not** include every small “helper/loader” function or full data dumps.

---

## What’s inside

### Tasks
- **Flood (multiclass)**: `hazard_class_flom ∈ {0,1,2}`
- **Landslide (prototype)**: simple “event count → hazard level” encoding `landslide_hazard_level ∈ {0,1,2}`

### Core features
Terrain channels used across CNN/GNN experiments:
- `dtm`, `slope`, `aspect`, `curv`, `flowacc`, `twi`

Tabular context (examples):
- `dist_to_river`, `dist_to_lake`, `dist_to_hyd`
- catchment/runoff normals (e.g. `QNormal_*`)
- optional landslide-derived proximity features (used carefully to avoid leakage)

### Model families
1. **Baselines (tabular)**  
   Logistic Regression + Random Forest on engineered numeric features.
2. **Baseline Flood CNN**  
   Small CNN over **6-channel terrain patches** (e.g. 20×20 pixels = 200m×200m @ 10m).
3. **Multimodal House-Graph model (prototype)**  
   Terrain CNN → `z_terrain`  
   Tabular MLP → `z_tab`  
   Fusion MLP → node embedding  
   GNN backbone (GCN) over a **KNN building graph** → hazard head(s)

---

## Repository layout 


- `flood_cnn.ipynb` — baseline CNN training & evaluation
- `cnn_gnn_mlp.ipynb` — multimodal prototype (CNN + MLP + GNN)
- `04_model_landslide_risk.ipynb` — landslide experiments / notes
- `pre_processing.py` — link-table cleaning, aggregation, target encoding, joins
- `knn_graph_spatial.py` — KNN graph construction + saving/loading bundles
- `terrain_patches.py` — patch extraction utilities (stack 6 raster layers)
- `dtm_builder.py` — DTM/derived rasters (helpers used during preprocessing)
- `clip_to_region.py` — clip rasters/vectors to AOI

---

## Data expectations


```
data/
  raw/
    vector/
      houses/                  # GeoPackages with building points (midpoints)
    rasters/                   # raw DTM10 per region (or merged/clipped)
  processed/
    rasters/                   # derived rasters per region (slope/aspect/...)
    links/                     # precomputed link tables (flood/hydro/landslide)
    graphs/                    # saved KNN graphs (.pt)
    channel_stats/             # per-region normalization stats (.npz)
```

**CRS expectation**: `EPSG:25833` (UTM33) for consistent meters-based distances and patch sizing.



---

## Quickstart

### 1) Create env + install deps
You’ll need the usual geospatial + ML stack:

- Python 3.10+ (recommended)
- `numpy`, `pandas`
- `geopandas`, `shapely`, `rasterio`
- `scikit-learn`
- `torch`
- (optional) `torch-geometric` for the GNN prototype

Example:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Flood CNN (recommended entry point)
Open and run:
- `flood_cnn.ipynb`

This uses:
- on-the-fly patch extraction (boundless windows to avoid edge crashes)
- per-patch NaN median fill
- optional per-region channel normalization (`.npz` stats)

### 3) Build a KNN graph (for GNN prototype)
```bash
python knn_graph_spatial.py
```

Or call the functions from the notebook to generate:
- `house_id`, `pos`, `edge_index` (+ optional undirected edges)

### 4) Multimodal model (CNN + MLP + GNN)
Open and run:
- `cnn_gnn_mlp.ipynb`

Notes:
- This is a **prototype** intended to prove the end-to-end pipeline.
- Expect to tweak node batching / masks / graph design (KNN vs radius vs hydrology-aware edges).

---

## Practical notes / limitations

- **Data quality & label noise** matters a lot in Norway hazard layers. Treat results as *susceptibility modelling*, not insurance-grade risk scoring.
- **Spatial leakage** is real: distance-to-event features can make models look “perfect” if you’re not careful.
- The graph model is harder to stabilize than the pure CNN baseline. It’s included mainly as a scalable direction.

---

## Reproducibility

This repo is intended to be “good enough to understand & rerun the models” rather than a fully packaged library.

If something is missing:
- search the notebooks first (most glue code lives there)
- then check `pre_processing.py`, `terrain_patches.py`, and `knn_graph_spatial.py`

---

## License



---

## Author

Noora Rehim Lindeflaten
