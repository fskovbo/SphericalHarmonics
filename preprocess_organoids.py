import glob
import os
import csv

from src.OrganoidMesh import OrganoidMesh
from src.mesh_analysis import *
from src.nonlocal_correlations import *
from src.cell_graph_functions import *
from src.utils import *


data_dir = '../NicoleData/20250929/fractal_output'
timepoints = ['day3', 'day3p5', 'day4', 'day4p5', 'day4p5-more']
zarr_names = {tp: 'r0.zarr' for tp in timepoints}
rounds = {tp: '0_fused_zillum_registered' for tp in timepoints}
meshes = {tp: 'nnorg_linked_multi_annotated_class' for tp in timepoints}

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

# --- Helper function for single organoid ---
def preprocess_single_organoid(path: str, save_path: str):
    """
    Preprocess a single organoid mesh and save coarse-grained data to a .npz file.

    Parameters
    ----------
    path : str
        Path to the input .vtp mesh file.
    save_path : str
        Path to save the preprocessed .npz file.
    """
    # --- Extract well and organoid ID from path ---
    organoid_id = os.path.splitext(os.path.basename(path))[0]

    # Example path structure:
    # .../<timepoint>/<zarr>/<well_letter>/<well_number>/<round>/meshes/.../42.vtp
    # e.g.: ../NicoleData/20250929/fractal_output/day4p5-more/r0.zarr/C/01/0_fused_zillum_registered/meshes/nnorg_linked_multi_annotated_class/42.vtp
    parts = os.path.normpath(path).split(os.sep)
    # last few parts contain [..., <well_letter>, <well_number>, <round>, meshes, ...]
    well_letter = parts[-5]
    well_number = parts[-4]
    well = f"{well_letter}{well_number}"

    # --- Load and process mesh ---
    mesh = OrganoidMesh()
    mesh.load_mesh_from_file(path)
    mesh.align_with_pca()
    mesh.compute_spectral_coefficients(lmax=15)

    vertex_areas = mesh.calc_vertex_areas()
    volume = mesh.calc_mesh_volume()

    # HKS fields
    hks, hks_coeffs = compute_hks(mesh, t=[1.0])
    hks_removed_l0, _ = mesh.remove_lowest_modes(coeffs=hks_coeffs, l_remove=1)
    fields = np.concatenate([hks, hks_removed_l0], axis=1)

    # Coarse-grain to cell level
    cell_center_vertices = mesh.get_centroid_vertices()
    centroids, cell_areas, cell_weighted_fields, _ = mesh.compute_cell_statistics(fields)
    fields_cell = cell_weighted_fields
    markers_cell = mesh.marker_fields[cell_center_vertices, :]
    marker_names = np.array(mesh.marker_names)

    # Geodesic distances between cell centers
    dist_heat = compute_geodesics(mesh, t=None, sources=cell_center_vertices)
    dist_heat = dist_heat[:, cell_center_vertices]

    # Cell adjacency graph
    cell_graph = build_cell_graph(mesh)
    graph_edges = np.array(cell_graph.edges(), dtype=int)
    graph_n_nodes = cell_graph.number_of_nodes()

    # --- Save all data ---
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


# --- Main loop ---
def preprocess_all_organoids():
    """
    Preprocess all organoids across timepoints and wells.
    Saves results to: ../NicoleData/preprocessed/<timepoint>/<index>.npz
    Also creates an index CSV mapping <index> → (well, organoid_id, original_path).
    """
    discard_ids = np.load('../NicoleData/combined_labels_to_discard.npy', allow_pickle=True)
    preproc_base = os.path.join('..', 'NicoleData', 'preprocessed')
    os.makedirs(preproc_base, exist_ok=True)

    for tp in timepoints:
        print(f"\n=== Preprocessing timepoint: {tp} ===")
        zarr_name = zarr_names[tp]
        round_name = rounds[tp]
        mesh_name = meshes[tp]

        # Output folder for this timepoint
        tp_out_dir = os.path.join(preproc_base, tp)
        os.makedirs(tp_out_dir, exist_ok=True)

        # Prepare index CSV
        index_path = os.path.join(tp_out_dir, "index.csv")
        index_entries = []

        org_counter = 0  # running index per timepoint

        for well in wells[tp]:
            base_dir = os.path.join(data_dir, tp, zarr_name, well[0], well[1:], round_name)
            mesh_dir = os.path.join(base_dir, 'meshes', mesh_name)
            organoid_paths = sorted(glob.glob(os.path.join(mesh_dir, "*.vtp")))

            kept_paths, _ = filter_organoid_paths(organoid_paths, discard_ids, tp, well)
            print(f"  {well}: {len(kept_paths)} organoids kept")

            for path in tqdm(kept_paths, desc=f"{tp} {well}"):
                organoid_id = os.path.splitext(os.path.basename(path))[0]
                save_path = os.path.join(tp_out_dir, f"{org_counter}.npz")

                try:
                    preprocess_single_organoid(path, save_path)
                except Exception as e:
                    print(f"    Error processing {path}: {e}")
                    continue

                # Add entry to index
                index_entries.append([org_counter, well, organoid_id, path])

                org_counter += 1

        # Write index CSV after processing all wells for this timepoint
        with open(index_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["index", "well", "organoid_id", "original_path"])
            writer.writerows(index_entries)

        print(f"Saved {len(index_entries)} entries to index file: {index_path}")



# Only run when executed as a script
if __name__ == "__main__":
    preprocess_all_organoids()