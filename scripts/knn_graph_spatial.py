from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from shapely.geometry import Point


def build_knn_graph_from_houses(
    houses_gdf,
    k: int = 8,
    house_id_col: str = "bygningsnummer",
    label_col: str | None = None,
    add_edge_dist: bool = True,
    device: str = "cpu",
):
    """
    Build a PyG Data graph from a houses GeoDataFrame using KNN edges.

    Returns:
        data: torch_geometric.data.Data
            data.pos        -> [N, 2] float
            data.edge_index -> [2, E] long
            data.edge_attr  -> [E, 1] float (optional distances)
            data.y          -> [N] long/float (optional)
            data.house_id   -> [N] long/string-ish tensor (stored as python list if needed)
    """
    houses_gdf.set_geometry("geometry", inplace=True)
    xs = houses_gdf.geometry.x.to_numpy()
    ys = houses_gdf.geometry.y.to_numpy()
    pos = np.column_stack([xs, ys]).astype("float32")

    pos_t = torch.from_numpy(pos).to(device)

    # Build KNN edge_index (includes self-loops by default if loop=True)
    # GNN use loop=False
    edge_index = knn_graph(pos_t, k=k, loop=False)

    data = Data()
    data.pos = pos_t
    data.edge_index = edge_index

    # Optional edge distances
    if add_edge_dist:
        # edge_index shape: [2, E]
        src = edge_index[0]
        dst = edge_index[1]
        d = (pos_t[src] - pos_t[dst]).pow(2).sum(dim=1).sqrt()  # Euclidean distance
        data.edge_attr = d.view(-1, 1)

    # Optional labels. I'll use flood and landslide but need to encode/norm them
    # Will use labels if time
    if label_col is not None and label_col in houses_gdf.columns:
        y = houses_gdf[label_col].to_numpy()
        y_t = torch.tensor(y, dtype=torch.long, device=device)
        data.y = y_t

    if house_id_col in houses_gdf.columns:
        ids = houses_gdf[house_id_col].to_numpy()
        # If these are large ints, this is fine:
        try:
            data.house_id = torch.tensor(ids, dtype=torch.long)
        except Exception:
            # fallback: store as python list
            data.house_id = list(ids)

    return data
