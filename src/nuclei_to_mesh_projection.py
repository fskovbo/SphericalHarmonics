import numpy as np
import pandas as pd

import igl

from sklearn.neighbors import NearestNeighbors
from src.OrganoidMesh import OrganoidMesh
from src.mesh_analysis import compute_geodesics


# ===================================================================
# Load nuclei data from table
# ===================================================================

# MARKER_BIN_COLS = [
#     "LGR.bin", "CHROMA.bin", "CYCD.bin", "MUC.bin", "ALDOB.bin",
#     "GLUC.bin", "CYCA.bin", "AGR.bin", "SERO.bin", "LYZ.bin",
# ]

# def extract_cell_attributes(nuclei_df_org):
#     nuclei_xyz = nuclei_df_org[["0.x_pos_pix", "0.y_pos_pix", "0.z_pos_pix_scaled"]].to_numpy(float)
#     markers_bin = nuclei_df_org[MARKER_BIN_COLS].to_numpy()
#     return nuclei_xyz, markers_bin


# MARKER_COLS = [
#     '0.C02.percentile99_class', # LGR5
#     '0.C03.percentile99_class', # chroma
#     '0.C04.percentile99_class', # aldoB
#     '1.C02.percentile99_class', # Sero
#     '1.C03.percentile99_class', # Lyz
#     '1.C04.percentile99_class', # Agr2
#     '2.C04.percentile99_class', # ki67
# ]

def extract_cell_attributes(nuclei_df_org, marker_cols, pos_cols=None):
    if pos_cols is None:
        pos_cols = ["0.x_pos_pix", "0.y_pos_pix", "0.z_pos_pix_scaled"]

    nuclei_xyz = nuclei_df_org[pos_cols].to_numpy(float)
    markers = nuclei_df_org[marker_cols].to_numpy()

    markers_bin = (markers > 0.0) # binarize markers

    return nuclei_xyz, markers_bin


# ===================================================================
# Projection of nuclei coordinates to mesh
# ===================================================================

def compute_face_normals_and_centroids(v, f):
    """
    v: (V,3), f: (F,3) int
    returns (normals: (F,3), centroids: (F,3))
    """
    tri = v[f]  # (F,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals /= norm
    centroids = (v0 + v1 + v2) / 3.0
    return normals, centroids


def project_nuclei_to_mesh(
    nuclei_xyz,
    mesh: OrganoidMesh,
    resolve_duplicates=False,
):
    """
    Project nuclei onto the mesh by nearest vertex, with optional
    duplicate-resolution using neighboring vertices.

    Parameters
    ----------
    nuclei_xyz : (N_cells, 3)
        Nucleus positions.
    mesh : OrganoidMesh containing vertices and faces
    resolve_duplicates : bool
        If True, detect nuclei that project to the same vertex and
        shift all but one to neighboring vertices

    Returns
    -------
    proj_vertex_ids : (N_cells,) int
        Vertex index used as geodesic source for each cell.
    proj_points : (N_cells, 3)
        Projected points on the mesh surface (at vertices).
    """
    nuclei_xyz = np.asarray(nuclei_xyz)
    v = np.asarray(mesh.v)
    f = np.asarray(mesh.f)

    N_cells = nuclei_xyz.shape[0]
    proj_vertex_ids = np.empty(N_cells, dtype=np.int64)
    proj_points = np.empty_like(nuclei_xyz)

    # ------------------------------------------------------------------
    # 1) Basic projection: nearest vertex for each nucleus
    # ------------------------------------------------------------------
    nn = NearestNeighbors(n_neighbors=1).fit(v)
    dists, idxs = nn.kneighbors(nuclei_xyz)  # (N_cells, 1)
    vid = idxs[:, 0]                         # nearest vertex index for each cell

    proj_vertex_ids[:] = vid
    proj_points[:] = v[vid]                  # project to the vertex position

    # ------------------------------------------------------------------
    # 2) Optional: handle duplicate projections using tangential direction
    # ------------------------------------------------------------------
    if not resolve_duplicates:
        return proj_vertex_ids, proj_points

    # Find vertices with >1 projected cell
    unique_vids, counts = np.unique(proj_vertex_ids, return_counts=True)
    dup_vids = unique_vids[counts > 1]
    dup_counts = counts[counts > 1]

    if len(dup_vids) == 0:
        print("Duplicate projections: none.")
        return proj_vertex_ids, proj_points

    total_pairs = int(np.sum(dup_counts * (dup_counts - 1) // 2))
    print(
        f"Duplicate projections: {len(dup_vids)} vertices with duplicates, "
        f"{total_pairs} duplicate cell pairs."
    )

    # ---- Build vertex adjacency (1-ring neighbors)
    V = v.shape[0]
    neighbors = [[] for _ in range(V)]
    for a, b, c in f:
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])
    neighbors = [list(set(nl)) for nl in neighbors]

    # ---- Precompute vertex normals (for tangential projection)
    #     Normal = average of adjacent face normals
    face_normals, _ = compute_face_normals_and_centroids(v, f)
    vertex_normals = np.zeros_like(v)
    for fi, (a, b, c) in enumerate(f):
        n = face_normals[fi]
        vertex_normals[a] += n
        vertex_normals[b] += n
        vertex_normals[c] += n
    # normalize
    vn_norm = np.linalg.norm(vertex_normals, axis=1)
    vn_norm[vn_norm == 0] = 1.0
    vertex_normals /= vn_norm[:, None]

    # ---- Process each duplicated vertex
    for v_id, n_here in zip(dup_vids, dup_counts):
        idxs_here = np.where(proj_vertex_ids == v_id)[0]
        if n_here <= 1:
            continue

        neighs = neighbors[v_id]
        if len(neighs) == 0:
            continue

        base_pos = v[v_id]
        normal = vertex_normals[v_id]

        # Assign one cell to stay at v_id
        keep_idx = idxs_here[0]

        # Remaining cells
        rest = idxs_here[1:]

        # For each cell: compute tangential displacement direction
        # relative to the *mean* nucleus coordinate of the cluster
        nucleus_block = nuclei_xyz[idxs_here]
        center = nucleus_block.mean(axis=0)

        tangential_vectors = []
        for cell_idx in rest:
            d = nuclei_xyz[cell_idx] - center
            d_perp = np.dot(d, normal) * normal
            d_tan = d - d_perp
            if np.linalg.norm(d_tan) < 1e-12:
                d_tan = np.random.randn(3)  # fallback small jitter
            tangential_vectors.append((cell_idx, d_tan / np.linalg.norm(d_tan)))

        # Determine directions from v₀ to neighbors
        neigh_dirs = []
        for u in neighs:
            d = v[u] - base_pos
            # project neighbor direction to tangent plane
            d_perp = np.dot(d, normal) * normal
            d_tan = d - d_perp
            if np.linalg.norm(d_tan) < 1e-12:
                continue
            neigh_dirs.append((u, d_tan / np.linalg.norm(d_tan)))

        # Assign each cell to best matching neighbor based on cosine similarity
        used = set()
        for cell_idx, d_tan in tangential_vectors:
            # score each neighbor by alignment
            scores = [
                (np.dot(d_tan, ndir), u)
                for (u, ndir) in neigh_dirs if u not in used
            ]
            if len(scores) == 0:
                # fallback: leave at original vertex
                continue
            best_score, best_u = max(scores, key=lambda x: x[0])
            used.add(best_u)

            proj_vertex_ids[cell_idx] = best_u
            proj_points[cell_idx] = v[best_u]

    return proj_vertex_ids, proj_points


def assign_cell_quantity_to_vertices(vertex_owner, cell_quantity):
    """
    Map a per-cell quantity onto the mesh vertices based on vertex_owner.

    Parameters
    ----------
    vertex_owner : (V,) array of int
        For each vertex, index of the owning cell (0..N_cells-1).
    cell_quantity : (N_cells,) or (N_cells, K) array
        Quantity defined per cell. Can be scalar or vector.

    Returns
    -------
    vertex_field : (V,) or (V, K) array
        Field defined per vertex.
    """

    vertex_owner = np.asarray(vertex_owner)
    cell_quantity = np.asarray(cell_quantity)

    # Validate indices
    if vertex_owner.min() < 0:
        raise ValueError("vertex_owner contains negative indices (unassigned vertices).")
    if vertex_owner.max() >= len(cell_quantity):
        raise ValueError("vertex_owner refers to a cell index beyond cell_quantity size.")

    # Simple, fast: gather rows by indexing
    vertex_field = cell_quantity[vertex_owner]

    return vertex_field



# ===================================================================
# Filtering of marker co-expression and coordinate scaling
# ===================================================================

def filter_lgr5_coexpression(
    markers_bin: np.ndarray,
    marker_names: list = None,
    coexpress_markers=(
        "LYZ.bin",     # Lysozyme → Paneth
        "MUC.bin",     # Mucin 2 → Goblet
        "AGR.bin",     # Agr2 → Goblet/Paneth
        "SERO.bin",    # Serotonin → Enterochromaffin
        "GLUC.bin",    # Glucagon → Enteroendocrine
        "CHROMA.bin",  # Chromogranin A → Enteroendocrine
    ),
    lgr5_marker="LGR.bin",
):
    """
    Turn off LGR5 positivity (set to 0) for cells coexpressing any 
    differentiation markers.

    Parameters
    ----------
    markers_bin : (N_cells, K) ndarray
        Binarized marker array from extract_cell_attributes.
    marker_names : list of str
        Column names corresponding to markers_bin columns.
        If None, defaults to MARKER_BIN_COLS.
    coexpress_markers : tuple of str
        Names of markers indicating differentiation.
    lgr5_marker : str
        Name of the LGR5 marker column to be suppressed.

    Returns
    -------
    filtered : (N_cells, K) ndarray
        Modified marker array.
    """
    if marker_names is None:
        marker_names = [
            "LGR.bin", "CHROMA.bin", "CYCD.bin", "MUC.bin", "ALDOB.bin",
            "GLUC.bin", "CYCA.bin", "AGR.bin", "SERO.bin", "LYZ.bin",
        ]

    filtered = markers_bin.copy()

    # Find LGR5 column index
    try:
        lgr5_idx = marker_names.index(lgr5_marker)
    except ValueError:
        raise ValueError(f"LGR5 marker '{lgr5_marker}' not found in marker_names.")

    # Indices of markers indicating differentiation
    coexpr_idx = [marker_names.index(m) for m in coexpress_markers if m in marker_names]

    if len(coexpr_idx) == 0:
        # Nothing to filter
        return filtered

    # Cells expressing any differentiation markers
    coexpr_mask = filtered[:, coexpr_idx].any(axis=1)

    # Set LGR5 to 0 for coexpressing cells
    filtered[coexpr_mask, lgr5_idx] = 0

    return filtered


def center_and_rescale_mesh(
    mesh: OrganoidMesh,
    nuclei_xyz: np.ndarray = None,
    scale: float = 10.0,
    inplace: bool = True,
):
    """
    Center mesh and nuclei together and rescale both by a fixed factor.

    Parameters
    ----------
    mesh : OrganoidMesh
        Mesh object containing .v (V, 3).
    nuclei_xyz : (N_cells, 3) ndarray
        Original nucleus coordinates.
    scale : float, default 10.0
        Factor by which to divide all centered coordinates.
    inplace : bool, default True
        If True, modifies mesh.v in-place.

    Returns
    -------
    v_new : (V, 3)
        Centered + rescaled mesh vertices.
    nuclei_new : (N_cells, 3)
        Centered + rescaled nuclei positions.
    center : (3,)
        Common center that was subtracted.
    scale : float
        The scaling factor used.
    """
    v = np.asarray(mesh.v)
    center = v.mean(axis=0)

    v_centered = v - center
    v_scaled = v_centered / scale

    if inplace:
        mesh.v = v_scaled

    if nuclei_xyz is not None:
        nuclei_xyz = np.asarray(nuclei_xyz)
        nuclei_centered = nuclei_xyz - center
        nuclei_scaled = nuclei_centered / scale

        return v_scaled, nuclei_scaled

    return v_scaled


# ===================================================================
# Voronoi tesselation
# ===================================================================

def compute_geodesic_voronoi(mesh, proj_vertex_ids):
    """
    Assign mesh vertices according to geodesic distance to nuclei projections.

    Parameters
    ----------
    mesh : OrganoidMesh
        Mesh object containing .v (V, 3).    
    proj_vertex_ids: (N, 1)
        Indices of vertices of projected nuclei

    Returns:
    ----------
    dist_mat : (V, N_cells) geodesic distances
    cell_label_field : (V,) index of closest cell for each vertex
    """
    dist_mat = compute_geodesics(mesh, t=None, sources=proj_vertex_ids)  # your existing function
    cell_label_field = np.argmin(dist_mat, axis=0)
    return dist_mat, cell_label_field