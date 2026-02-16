import glob
import os
import csv
import pandas as pd
from tqdm import tqdm

from mesh.OrganoidMesh import OrganoidMesh
from src.mesh_analysis import *
from src.nonlocal_correlations import *
from src.cell_graph_functions import *
from src.utils import *


# Strip timepoint prefix from well names if present
def clean_well_name(well_str):
    # e.g. "day3p5_A04" -> "A04"
    if "_" in well_str:
        return well_str.split("_")[-1]
    return well_str

# === Base data info (same as before) ===
data_dir = '../NicoleData/20250929/fractal_output'
tp_original = 'day3p5'                     # original timepoint where data lives
tp_selected = 'day3p5-selected'            # new synthetic timepoint name
zarr_name = 'r0.zarr'
round_name = '0_fused_zillum_registered'
mesh_name = 'nnorg_linked_multi_annotated_class'

# --- Helper function for single organoid (unchanged) ---
def preprocess_single_organoid(path: str, save_path: str):
    organoid_id = os.path.splitext(os.path.basename(path))[0]
    parts = os.path.normpath(path).split(os.sep)
    well_letter = parts[-6]
    well_number = parts[-5]
    well = f"{well_letter}{well_number}"

    mesh = OrganoidMesh()
    mesh.load_mesh_from_file(path)
    mesh.align_with_pca()
    mesh.compute_spectral_coefficients(lmax=15)

    vertex_areas = mesh.calc_vertex_areas()
    volume = mesh.calc_mesh_volume()

    hks, hks_coeffs = compute_hks(mesh, t=[1.0, 4.0, 25.0])
    # hks_removed_l0, _ = mesh.remove_lowest_modes(coeffs=hks_coeffs, l_remove=1)
    # fields = np.concatenate([hks, hks_removed_l0], axis=1)
    fields = hks

    cell_center_vertices = mesh.get_centroid_vertices()
    centroids, cell_areas, cell_weighted_fields, _ = mesh.compute_cell_statistics(fields)
    fields_cell = cell_weighted_fields
    markers_cell = mesh.marker_fields[cell_center_vertices, :]
    marker_names = np.array(mesh.marker_names)

    dist_heat = compute_geodesics(mesh, t=None, sources=cell_center_vertices)
    dist_heat = dist_heat[:, cell_center_vertices]

    cell_graph = build_cell_graph(mesh)
    graph_edges = np.array(cell_graph.edges(), dtype=int)
    graph_n_nodes = cell_graph.number_of_nodes()

    np.savez_compressed(
        save_path,
        fields_cell=fields_cell,
        markers_cell=markers_cell,
        marker_names=marker_names,
        cell_areas=cell_areas,
        dist_heat=dist_heat,
        graph_edges=graph_edges,
        graph_n_nodes=graph_n_nodes,
        vertex_areas=vertex_areas,
        volume=volume,
        organoid_id=organoid_id,
        well=well,
    )

# --- New function to preprocess ONLY selected organoids ---
def preprocess_selected_organoids():
    """
    Preprocess only the organoids listed in the CSV file for day3p5.
    Saves results under the new synthetic timepoint 'day3p5-selected'.
    """
    # === Load selected wells and labels from CSV ===
    mesh_table = pd.read_csv('../NicoleData/day3p5_goodmeshes.csv', sep=';')
    selected_wells = mesh_table['well'].astype(str).apply(clean_well_name).tolist()
    selected_labels = mesh_table['label'].astype(str).tolist()

    print(f"Loaded {len(selected_wells)} selected organoids from CSV")

    preproc_base = os.path.join('..', 'NicoleData', 'preprocessed')
    tp_out_dir = os.path.join(preproc_base, tp_selected)
    os.makedirs(tp_out_dir, exist_ok=True)

    index_path = os.path.join(tp_out_dir, "index.csv")
    index_entries = []
    org_counter = 0

    for well, label in zip(selected_wells, selected_labels):
        base_dir = os.path.join(data_dir, tp_original, zarr_name, well[0], well[1:], round_name)
        mesh_dir = os.path.join(base_dir, 'meshes', mesh_name)
        organoid_path = os.path.join(mesh_dir, f"{label}.vtp")

        if not os.path.exists(organoid_path):
            print(f"⚠ File not found for well={well}, label={label}")
            continue

        save_path = os.path.join(tp_out_dir, f"{org_counter}.npz")
        try:
            preprocess_single_organoid(organoid_path, save_path)
            index_entries.append([org_counter, well, label, organoid_path])
            org_counter += 1
        except Exception as e:
            print(f"Error processing {organoid_path}: {e}")
            continue

    # Write index CSV
    with open(index_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "well", "organoid_id", "original_path"])
        writer.writerows(index_entries)

    print(f"\n✅ Preprocessed {len(index_entries)} selected organoids")
    print(f"Results stored in: {tp_out_dir}")

# Only run when executed as a script
if __name__ == "__main__":
    preprocess_selected_organoids()
