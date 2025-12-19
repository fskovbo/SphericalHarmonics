import glob
import os
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.OrganoidMesh import OrganoidMesh
from src.cell_graph_functions import *
from src.nuclei_to_mesh_projection import *
from src.crypt_extraction import compute_vocabulary_encoding
from src.mesh_analysis import compute_hks
from src.utils import filter_organoid_paths


# ---------------------------------------------------------------------
# Dataset configuration (mirrors preprocess_organoids.py)
# ---------------------------------------------------------------------
data_dir = '../NicoleData/20250929/fractal_output'

timepoints = ['day2', 'day2p5', 'day3', 'day3p5', 'day4', 'day4p5', 'day4p5-more']   # extend as needed: ['day3', 'day3p5', ...]
zarr_names = {tp: 'r0.zarr' for tp in timepoints}
rounds     = {tp: '0_fused_zillum_registered' for tp in timepoints}
meshes     = {tp: 'nnorg_linked_multi_annotated_class' for tp in timepoints}

wells = {
    'day1p5': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
    'day2': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
    'day2p5': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
    'day3': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'B02', 'B03'],
    'day3p5': ['A01', 'A02', 'A03', 'A04', 'B03'],
    'day4': ['A02', 'A03', 'A04', 'A05', 'A06', 'B01', 'B02'],
    'day4p5': ['A06', 'B06'],
    'day4p5-more': ['C01', 'C02', 'C03', 'C04', 'C05', 'C06'],
}

hks_times = [1.0, 2.0, 4.0, 8.0, 25.0]


# Path to the combined nuclei table (adjust if different)
CELLS_CSV = "../NicoleData/features_csv/features/cell_types_class.csv"   # <-- adjust


# ---------------------------------------------------------------------
# Helper: preprocess a single organoid -> graph with HKS + encoding
# ---------------------------------------------------------------------
def create_organoid_graph_from_mesh(
    mesh_path,
    tp,
    well,
    cells_df,
    resolve_duplicates=True,
):
    """
    Build the cell adjacency graph for a single organoid and save to .npz.

    Returns
    -------
    G : networkx.Graph
    dist_mat : (V, N_cells) geodesic distances (if computed)
    vertex_owner : (V,)
    proj_vertex_ids : (N_cells,)
    """
    organoid_id = os.path.splitext(os.path.basename(mesh_path))[0]
    label_uid = f"{tp}_{well}_{organoid_id}"

    nuclei_df_org = cells_df[cells_df["label_uid"] == label_uid].copy()
    if nuclei_df_org.empty:
        print(f"  [WARN] No nuclei found for label_uid={label_uid}, skipping")
        return None, None, None, None

    # 1) load membrane mesh as an OrganoidMesh
    mesh = OrganoidMesh(mesh_path)

    # 2) extract per-cell attributes
    nuclei_xyz, markers_bin = extract_cell_attributes(nuclei_df_org)

    _, nuclei_xyz = center_and_rescale_mesh(mesh, nuclei_xyz)
    markers_bin = filter_lgr5_coexpression(markers_bin) 

    # 3) project nuclei -> mesh vertices (using geometry from the mesh object)
    proj_vertex_ids, proj_points = project_nuclei_to_mesh(
        nuclei_xyz,
        mesh,
        resolve_duplicates=resolve_duplicates,
    )

    # store eigen-decomposition for geodesics (if needed by compute_geodesics)
    mesh._eig_decomp()

    # 4) geodesic distances + Voronoi assignment
    dist_mat, vertex_owner = compute_geodesic_voronoi(mesh, proj_vertex_ids)

    # 5) build graph from Voronoi partition (now pass mesh, not mesh_f)
    G = build_cell_graph_from_voronoi(
        nuclei_xyz,
        markers_bin,
        mesh,            
        vertex_owner,
        proj_vertex_ids,
        proj_points,
    )

    G.graph["label_uid"] = label_uid   # store metadata in the graph object

    return G, dist_mat, vertex_owner, proj_vertex_ids, mesh


# ---------------------------------------------------------------------
# Main loop over full dataset
# ---------------------------------------------------------------------
def preprocess_all_organoid_graphs():
    """
    Preprocess graphs for all organoids across timepoints and wells.

    For each organoid mesh:
      - build cell graph from nuclei + membrane mesh
      - add HKS & encoding fields to nodes
      - save graph as pickled NetworkX object (.gpickle)
      - record metadata in an index.csv per timepoint

    Graphs are saved under: ../NicoleData/graphs/<timepoint>/<index>.gpickle
    """

    # nuclei table
    cells_df = pd.read_csv(CELLS_CSV)

    # vocab
    vocab = np.load('./sim/vocab.npz', allow_pickle=True)

    # same discard list as mesh preproc
    discard_ids = np.load('../NicoleData/combined_labels_to_discard.npy', allow_pickle=True)

    graphs_base = os.path.join('..', 'NicoleData', 'graphs')
    os.makedirs(graphs_base, exist_ok=True)

    for tp in timepoints:
        print(f"\n=== Preprocessing graphs for timepoint: {tp} ===")
        zarr_name = zarr_names[tp]
        round_name = rounds[tp]
        mesh_name = meshes[tp]

        tp_out_dir = os.path.join(graphs_base, tp)
        os.makedirs(tp_out_dir, exist_ok=True)

        org_counter = 0

        for well in wells[tp]:
            base_dir = os.path.join(data_dir, tp, zarr_name, well[0], well[1:], round_name)
            mesh_dir = os.path.join(base_dir, 'meshes', mesh_name)
            organoid_paths = sorted(glob.glob(os.path.join(mesh_dir, "*.vtp")))

            kept_paths, _ = filter_organoid_paths(organoid_paths, discard_ids, tp, well)
            print(f"  {well}: {len(kept_paths)} organoids kept")

            for path in tqdm(kept_paths, desc=f"{tp} {well}"):
                organoid_id = os.path.splitext(os.path.basename(path))[0]
                label_uid = f"{tp}_{well}_{organoid_id}"

                try:
                    G, _, _, _, mesh = create_organoid_graph_from_mesh(
                        path,
                        tp,
                        well,
                        cells_df
                    )
                except Exception as e:
                    print(f"    Error processing {path}: {e}")
                    continue

                if G is None:
                    # e.g. missing nuclei for this label_uid
                    continue

                # Compute HKS and encoding, then append to graph
                hks = compute_hks(mesh, t=hks_times, coeffs=False)   # (V, T)
                add_vertex_field_to_graph(G, hks, "hks")

                # store times as graph-level metadata (no need to duplicate per node)
                G.graph["hks_times"] = hks_times

                # Compute encoing
                encoding, _, _, _ = compute_vocabulary_encoding(vocab, mesh)
                add_vertex_field_to_graph(G, encoding, "vocab_encoding")

                # where to save this graph
                graph_path = os.path.join(tp_out_dir, f"{org_counter}.gpickle")

                # label_uid_out should equal label_uid; use it just in case
                save_cell_graph(graph_path, G)

                org_counter += 1


if __name__ == "__main__":
    preprocess_all_organoid_graphs()
