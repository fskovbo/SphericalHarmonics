import numpy as np


# ===================================================================
# Load nuclei data from table
# ===================================================================

MARKER_BIN_COLS = [
    "LGR.bin", "CHROMA.bin", "CYCD.bin", "MUC.bin", "ALDOB.bin",
    "GLUC.bin", "CYCA.bin", "AGR.bin", "SERO.bin", "LYZ.bin",
]

def extract_cell_attributes(nuclei_df_org):
    nuclei_xyz = nuclei_df_org[["0.x_pos_pix", "0.y_pos_pix", "0.z_pos_pix_scaled"]].to_numpy(float)
    markers_bin = nuclei_df_org[MARKER_BIN_COLS].to_numpy()
    return nuclei_xyz, markers_bin


# ===================================================================
# Projection of nuclei coordinates to mesh
# ===================================================================




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


# ===================================================================
# Voronoi tesselation
# ===================================================================





# ===================================================================
# Single function for running projection+tesselation
# ===================================================================



