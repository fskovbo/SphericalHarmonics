

"""
crypt_analysis.py
=================

This module provides a unified set of functions for quantitative analysis of
intestinal organoid crypt geometry and cell composition on triangulated surface meshes.

The core idea is to construct a 1D crypt coordinate system from a complex 3D surface,
allowing geometric and biological quantities to be studied as a function of distance
from the crypt bottom.

The functions implement the following pipeline:

1) Crypt–neck interface detection
   - Identify mesh vertices forming the boundary between crypt and neck/villus regions.

2) Crypt bottom selection and crypt length definition
   - Choose a robust crypt bottom cell based on geodesic distances.
   - Define a scalar crypt length that is stable across the crypt boundary.

3) Distance field construction
   - Compute normalized distances from the crypt bottom to mesh vertices and cells.
   - This defines a 1D coordinate along the crypt axis that can extend into the villus.

4) Geometric analysis
   - Compute crypt circumference as iso-contour lengths of the distance field.
   - Quantify how crypt shape changes along the crypt axis.

5) Cell-type and marker analysis
   - Bin cell marker positivity as a function of crypt distance.
   - Enables spatial profiling of cell fate along the crypt.

6) Spatial correlation analysis
   - Compute bottom-anchored two-point correlations of vertex fields (e.g. HKS)
     as a function of geodesic distance.
   - Provides a nonlocal characterization of crypt structure.

The functions are designed to be:
- mesh-based (vertex-level resolution),
- geodesic-aware,
- numerically robust,
- and modular, so that alternative definitions of crypt bottom, boundary, or
  distance normalization can be easily tested.

Typical usage pattern:
    boundary -> bottom -> normalized distance -> geometry / markers / correlations
"""



import numpy as np
import heapq
from src.cell_graph_functions import *



# ============================================================
# Patch utilities
# ============================================================

def as_patch_list(x):
    """Normalize region definitions to list[set[int]]."""
    if x is None:
        return []
    if isinstance(x, set):
        return [x]
    if isinstance(x, (list, tuple)):
        return [set(p) for p in x if p is not None and len(p) > 0]
    return []


# ============================================================
# Crypt–neck boundary extraction
# ============================================================

# def crypt_neck_boundary_vertices(
#     mesh,           # mesh object with v (V,3), f (F,3)
#     vertex_owner,   # (V,) array, owning cell index for each vertex
#     crypt_cells,    # iterable[int], crypt cell indices
#     neck_cells,     # iterable[int], neck/villus cell indices
# ):
#     """
#     Return mesh vertices on the crypt–neck boundary (biased to neck side).

#     Output
#     ------
#     boundary_vertex_ids : (K,) ndarray[int]
#         Neck vertices that have at least one crypt neighbor.
#     """
#     vertex_owner = np.asarray(vertex_owner, dtype=np.int64)
#     crypt_cells = np.asarray(list(crypt_cells), dtype=np.int64)
#     neck_cells = np.asarray(list(neck_cells), dtype=np.int64)

#     V = vertex_owner.shape[0]
#     faces = np.asarray(mesh.f, dtype=np.int64)

#     is_crypt = np.isin(vertex_owner, crypt_cells)
#     is_neck = np.isin(vertex_owner, neck_cells)

#     neighbors = [[] for _ in range(V)]
#     for a, b, c in faces:
#         neighbors[a].extend([b, c])
#         neighbors[b].extend([a, c])
#         neighbors[c].extend([a, b])
#     neighbors = [np.unique(n) for n in neighbors]

#     boundary = np.zeros(V, dtype=bool)
#     for v in range(V):
#         if not is_neck[v]:
#             continue
#         if np.any(is_crypt[neighbors[v]]):
#             boundary[v] = True

#     return np.sort(np.nonzero(boundary)[0])




# ============================================================
# Crypt bottom selection (single scalar length L*)
# ============================================================


# def find_crypt_bottom(
#     dist_mat,             # (N_cells, V) OR (N_crypt_cells, V) array, geodesic distances
#     crypt_cells=None,     # iterable[int] or None. If None: dist_mat already restricted to crypt cells
#     boundary_vertex_ids=None,  # (K,) array[int], boundary vertex indices (global vertex ids)
#     candidates="all",     # "all" or int, subsample candidate bottoms (over crypt rows)
#     score="cv",           # "cv", "iqr_over_median", "range_over_median"
#     length_stat="median", # "median" or "mean"
#     return_row_index=False, # if True: also return row index into the (possibly restricted) dist_mat
# ):
#     """
#     Choose a crypt bottom by minimizing variation of distances from bottom->boundary vertices.

#     Two modes
#     ---------
#     (A) Full dist_mat (all cells):
#         - Provide crypt_cells (global cell ids)
#         - dist_mat is (N_cells, V)

#     (B) Crypt-only dist_mat:
#         - Set crypt_cells=None
#         - dist_mat is already restricted to crypt cells: shape (N_crypt_cells, V)
#         - Returned bottom_cell_id will then be the *row index* into this restricted matrix
#           unless you pass crypt_cells explicitly.

#     Outputs
#     -------
#     bottom_cell_id : int
#         If crypt_cells is provided: global cell id.
#         If crypt_cells is None: row index into the provided dist_mat.
#     L_star : float
#         Scalar crypt length (median or mean distance to boundary vertices).
#     score_value : float
#         Variation score for the chosen bottom.
#     (optional) bottom_row : int
#         Row index into the internal candidate matrix (useful for debugging / mode B).
#     """
#     dist_mat = np.asarray(dist_mat, dtype=float)
#     if boundary_vertex_ids is None:
#         raise ValueError("boundary_vertex_ids must be provided.")
#     boundary_vertex_ids = np.asarray(boundary_vertex_ids, dtype=np.int64)
#     if boundary_vertex_ids.size == 0:
#         raise ValueError("boundary_vertex_ids empty.")

#     # --- Build candidate row indices ---
#     if crypt_cells is None:
#         # dist_mat rows are already crypt cells
#         crypt_rows = np.arange(dist_mat.shape[0], dtype=np.int64)
#         crypt_ids = None
#     else:
#         crypt_ids = np.fromiter(crypt_cells, dtype=np.int64)
#         if crypt_ids.size == 0:
#             raise ValueError("crypt_cells empty.")
#         crypt_rows = crypt_ids  # rows into full dist_mat

#     # Optional subsampling of candidates for speed
#     if isinstance(candidates, int) and candidates > 0 and crypt_rows.size > candidates:
#         pick = np.linspace(0, crypt_rows.size - 1, candidates).astype(int)
#         crypt_rows = crypt_rows[pick]
#         if crypt_ids is not None:
#             crypt_ids = crypt_ids[pick]

#     # --- Distances candidate->boundary vertices ---
#     # Works in both modes:
#     # - Mode A: dist_mat[crypt_rows] selects those cell rows
#     # - Mode B: crypt_rows are 0..N_crypt-1
#     D = dist_mat[crypt_rows][:, boundary_vertex_ids]
#     D = np.where(np.isfinite(D), D, np.nan)

#     # Scalar length per candidate
#     if length_stat == "median":
#         L = np.nanmedian(D, axis=1)
#     elif length_stat == "mean":
#         L = np.nanmean(D, axis=1)
#     else:
#         raise ValueError("length_stat must be 'median' or 'mean'.")

#     L_safe = np.maximum(L, 1e-8)

#     # Variation score
#     if score == "cv":
#         s = np.nanstd(D, axis=1) / L_safe
#     elif score == "iqr_over_median":
#         q75 = np.nanpercentile(D, 75, axis=1)
#         q25 = np.nanpercentile(D, 25, axis=1)
#         s = (q75 - q25) / L_safe
#     elif score == "range_over_median":
#         s = (np.nanmax(D, axis=1) - np.nanmin(D, axis=1)) / L_safe
#     else:
#         raise ValueError("score must be 'cv', 'iqr_over_median', or 'range_over_median'.")

#     # Penalize candidates with too many NaNs
#     frac_nan = np.mean(~np.isfinite(D), axis=1)
#     s = np.where(frac_nan > 0.5, np.inf, s)

#     best_row_local = int(np.nanargmin(s))   # index into crypt_rows array
#     L_star = float(L_safe[best_row_local])
#     score_val = float(s[best_row_local])

#     if crypt_cells is None:
#         # Return row index into the provided crypt-only dist_mat
#         bottom_id = int(crypt_rows[best_row_local])
#         if return_row_index:
#             return bottom_id, L_star, score_val, bottom_id
#         return bottom_id, L_star, score_val

#     # crypt_cells provided -> return global cell id
#     bottom_cell_id = int(crypt_rows[best_row_local])
#     if return_row_index:
#         return bottom_cell_id, L_star, score_val, best_row_local
#     return bottom_cell_id, L_star, score_val










# def calculate_crypt_distance(
#     mesh,                    # mesh with v,f,laplacian,mass_matrix
#     G,                       # cell graph with node attribute 'proj_vertex'
#     bottom_cell_id,          # int, crypt bottom cell id
#     neck_boundary_cells,     # iterable[int], neck/villus cells touching crypt
#     compute_geodesics_fn,    # function(mesh, sources, t=None) -> (S,V) or (V,)
#     t=None,                  # heat time (optional)
# ):
#     """
#     Compute crypt distance using a single heat-method solve from the crypt bottom,
#     then normalize by mean distance to NECK BOUNDARY CELL CENTERS.

#     Outputs
#     -------
#     dnorm_vertices : (V,) float
#         normalized bottom->vertex distance
#     dist_vertices : (V,) float
#         raw bottom->vertex distance
#     dnorm_cellcenters : (N_cells,) float
#         normalized distance at cell centers (proj_vertex)
#     dist_cellcenters : (N_cells,) float
#         raw distance at cell centers (proj_vertex)
#     L_mean : float
#         mean bottom->neck_boundary_cell_center distance used for normalization
#     """
#     neck_boundary_cells = np.asarray(sorted(set(neck_boundary_cells)), dtype=np.int64)
#     if neck_boundary_cells.size == 0:
#         raise ValueError("neck_boundary_cells empty.")

#     proj_vertex_ids = get_proj_vertex_ids(G)          # (N_cells,)
#     bottom_vertex = int(proj_vertex_ids[int(bottom_cell_id)])

#     # --- heat-method geodesics from bottom vertex ---
#     D = compute_geodesics_fn(mesh, sources=np.array([bottom_vertex], dtype=np.int64), t=t)
#     D = np.asarray(D, dtype=float)
#     dist_vertices = D[0] if D.ndim == 2 else D

#     # --- evaluate at cell centers ---
#     dist_cellcenters = dist_vertices[proj_vertex_ids]   # (N_cells,)

#     # --- normalization length based on mean distance to boundary neck cell centers ---
#     L_vals = dist_cellcenters[neck_boundary_cells]
#     L_vals = L_vals[np.isfinite(L_vals)]
#     if L_vals.size == 0:
#         raise RuntimeError("Boundary neck cell center distances invalid/disconnected.")

#     L_mean = float(np.mean(L_vals))
#     L_mean = max(L_mean, 1e-8)

#     dnorm_vertices = dist_vertices / L_mean
#     dnorm_cellcenters = dist_cellcenters / L_mean

#     return dnorm_vertices, dist_vertices, dnorm_cellcenters, dist_cellcenters, L_mean



# ============================================================
# Distance normalization
# ============================================================

# def normalize_crypt_length(
#     dist_mat,        # (N_cells, V) array, geodesic distances
#     bottom_cell_idx,  # int, crypt bottom cell index
#     L_star,          # float, scalar crypt length
#     vertex_owner=None, # (V,) array or None, owning cell index per vertex
# ):
#     """
#     Compute normalized distance to crypt bottom.

#     Output
#     ------
#     dnorm_vertices : (V,) array
#     dnorm_cells : (N_cells,) array or None
#     """
#     dist_mat = np.asarray(dist_mat, dtype=float)
#     L_star = max(float(L_star), 1e-8)

#     dist_bottom = dist_mat[int(bottom_cell_idx), :]
#     dnorm_v = dist_bottom / L_star

#     if vertex_owner is None:
#         return dnorm_v, None

#     vertex_owner = np.asarray(vertex_owner, dtype=np.int64)
#     n_cells = dist_mat.shape[0]
#     dnorm_c = np.full(n_cells, np.nan)

#     for c in range(n_cells):
#         mask = (vertex_owner == c)
#         if np.any(mask):
#             vals = dnorm_v[mask]
#             vals = vals[np.isfinite(vals)]
#             if vals.size:
#                 dnorm_c[c] = np.mean(vals)

#     return dnorm_v, dnorm_c


# ============================================================
# Circumference from iso-contours
# ============================================================




# ============================================================
# Marker binning
# ============================================================



# ============================================================
# Anchored correlation
# ============================================================

# def crypt_anchored_correlation(
#     X_vertices,              # (V,) array, field on vertices (e.g. HKS at fixed time)
#     crypt_dist,             # (V,) array, distance from crypt bottom to vertices
#     bin_edges,               # (B+1,) array, distance bins
#     weights=None,            # (V,) array or None, vertex weights (e.g. areas)
#     connected=True,          # bool, subtract global mean if True
#     normalized=False,        # bool, divide by global variance if True
#     exclude_self=True,       # bool, exclude bottom vertex from bins
# ):
#     """
#     Compute bottom-anchored two-point correlation vs distance.

#     Output
#     ------
#     corr : (B,) array
#     counts : (B,) array
#     bin_centers : (B,) array
#     """
#     X = np.asarray(X_vertices, dtype=float)
#     d = np.asarray(crypt_dist, dtype=float)
#     be = np.asarray(bin_edges, dtype=float)

#     B = len(be) - 1
#     centers = 0.5 * (be[:-1] + be[1:])

#     w = np.ones_like(X) if weights is None else np.asarray(weights, dtype=float)
#     valid = np.isfinite(X) & np.isfinite(d) & np.isfinite(w)

#     bottom_v = int(np.nanargmin(d))
#     if exclude_self:
#         valid[bottom_v] = False

#     mu = np.average(X[valid], weights=w[valid])
#     var = np.average((X[valid] - mu) ** 2, weights=w[valid])
#     var = max(var, 1e-12)

#     Xb = X[bottom_v]
#     if connected:
#         Xb_eff = Xb - mu
#         Y = X - mu
#     else:
#         Xb_eff = Xb
#         Y = X

#     bin_ids = np.digitize(d, be) - 1
#     corr = np.full(B, np.nan)
#     counts = np.zeros(B)

#     for b in range(B):
#         m = valid & (bin_ids == b)
#         if np.any(m):
#             counts[b] = w[m].sum()
#             corr[b] = Xb_eff * np.sum(w[m] * Y[m]) / counts[b]

#     if normalized:
#         corr /= var

#     return corr, counts, centers






# ============================================================
# Vertex area weights
# ============================================================

# def vertex_areas(mesh):  # mesh with mass_matrix
#     """Return per-vertex area weights from mesh.mass_matrix diagonal."""
#     M = mesh.mass_matrix
#     if hasattr(M, "diagonal"):
#         return np.asarray(M.diagonal(), dtype=float)
#     return np.asarray(np.diag(M), dtype=float)



# ===========================================================
# Crypt length adjustment
# ===========================================================







# ===========================================================
# Scalar metrics for clustering crypts based on morphology
# ===========================================================


def compute_crypt_metrics(
    s,                 # (B,) bin centers (or any s-sampling of the profile)
    C,                 # (B,) circumference profile C(s) on the same grid
    L,                 # crypt length (scalar)
    mu_hks,            # (B,) mean HKS vs s (same bins)
    std_hks,           # (B,) std HKS vs s (same bins)
    s_max=1.0,         # compute indices over s in [0, s_max]
    eps=1e-12,
):
    """
    Compute three scalars in [0,1] for a single crypt:

      BI = 1 - C(s=1) / max_{s<=1} C(s)     (openness / budding)
      CI = (mean_{s<=1} std_hks) / (mean_{s<=1} mu_hks)   (scale-invariant curvature irregularity)
      EI = L/C(s=1) (elongatedness)
      
    Notes
    -----
    - If s does not contain exactly 1.0, C(s=1) is obtained by linear interpolation.
    - If max C is nonpositive/invalid, BI is set to 0.
    - CI is clipped to [0,1] after mapping with CI/(CI+1) to keep it bounded.
    """
    s = np.asarray(s, dtype=float)
    C = np.asarray(C, dtype=float)
    mu = np.asarray(mu_hks, dtype=float)
    sd = np.asarray(std_hks, dtype=float)

    if not (s.ndim == C.ndim == mu.ndim == sd.ndim == 1):
        raise ValueError("Inputs must be 1D arrays.")
    if not (len(s) == len(C) == len(mu) == len(sd)):
        raise ValueError("All inputs must have the same length.")

    # Mask to s<=s_max and finite values
    in_rng = (s <= float(s_max))
    mC = in_rng & np.isfinite(C)
    mK = in_rng & np.isfinite(mu) & np.isfinite(sd)

    # --- BI ---
    BI = 0.0
    EI = 0.0
    if np.sum(mC) >= 2:
        sC = s[mC]
        CC = C[mC]

        # C at s=1 (interpolate)
        C1 = CC[-1] #np.interp(1.0, sC, CC, left=np.nan, right=np.nan)
        Cmax = np.nanmax(CC)

        if np.isfinite(C1) and np.isfinite(Cmax) and Cmax > eps:
            BI = 1.0 - float(C1) / float(Cmax)
            BI = float(np.clip(BI, 0.0, 1.0))

            EI = float( L/C1 )

    # --- CI (scale-invariant) ---
    CI_raw = np.nan
    if np.sum(mK) >= 2:
        mu_mean = float(np.nanmean(mu[mK]))
        sd_mean = float(np.nanmean(sd[mK]))
        if np.isfinite(mu_mean) and mu_mean > eps and np.isfinite(sd_mean) and sd_mean >= 0:
            CI_raw = sd_mean / mu_mean

    # Map to [0,1] with a saturating transform (monotone, robust)
    # CI_raw = 0 -> 0, CI_raw -> inf -> 1
    if np.isfinite(CI_raw):
        CI = float(CI_raw / (CI_raw + 1.0))
        CI = float(np.clip(CI, 0.0, 1.0))
    else:
        CI = 0.0

    return BI, CI, EI


import numpy as np

def budding_index(
    s,        # (B,) bin centers (or s-sampling)
    C,        # (B,) circumference profile C(s)
    s_max=1.0,
    eps=1e-12,
):
    """
    Compute budding index (BI) for a single crypt:

        BI = 1 - C(s=1) / max_{s<=1} C(s)

    Returns
    -------
    BI : float in [0,1]

    Notes
    -----
    - Uses only bins with s <= s_max and finite C.
    - If s does not contain exactly 1.0, uses the last valid value
      (same behavior as your current code).
    - If profile is invalid, returns 0.
    """
    s = np.asarray(s, dtype=float)
    C = np.asarray(C, dtype=float)

    if s.ndim != 1 or C.ndim != 1 or len(s) != len(C):
        raise ValueError("s and C must be 1D arrays of equal length.")

    m = (s <= float(s_max)) & np.isfinite(C)
    if np.sum(m) < 2:
        return 0.0

    sC = s[m]
    CC = C[m]

    # circumference at s≈1 (use last valid bin ≤ s_max)
    C1 = CC[-1]
    Cmax = np.nanmax(CC)

    if not (np.isfinite(C1) and np.isfinite(Cmax) and Cmax > eps):
        return 0.0

    BI = 1.0 - float(C1) / float(Cmax)
    return float(np.clip(BI, 0.0, 1.0))
