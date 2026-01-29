

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

def crypt_neck_boundary_vertices(
    mesh,           # mesh object with v (V,3), f (F,3)
    vertex_owner,   # (V,) array, owning cell index for each vertex
    crypt_cells,    # iterable[int], crypt cell indices
    neck_cells,     # iterable[int], neck/villus cell indices
):
    """
    Return mesh vertices on the crypt–neck boundary (biased to neck side).

    Output
    ------
    boundary_vertex_ids : (K,) ndarray[int]
        Neck vertices that have at least one crypt neighbor.
    """
    vertex_owner = np.asarray(vertex_owner, dtype=np.int64)
    crypt_cells = np.asarray(list(crypt_cells), dtype=np.int64)
    neck_cells = np.asarray(list(neck_cells), dtype=np.int64)

    V = vertex_owner.shape[0]
    faces = np.asarray(mesh.f, dtype=np.int64)

    is_crypt = np.isin(vertex_owner, crypt_cells)
    is_neck = np.isin(vertex_owner, neck_cells)

    neighbors = [[] for _ in range(V)]
    for a, b, c in faces:
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])
    neighbors = [np.unique(n) for n in neighbors]

    boundary = np.zeros(V, dtype=bool)
    for v in range(V):
        if not is_neck[v]:
            continue
        if np.any(is_crypt[neighbors[v]]):
            boundary[v] = True

    return np.sort(np.nonzero(boundary)[0])


# def crypt_neck_boundary_cells(
#     G,                # cell graph (networkx)
#     crypt_cells,      # set[int]
#     neck_patches,     # list[set[int]] (villus/neck regions)
# ):
#     """
#     Return neck/villus cells that are at the interface with the given crypt,
#     i.e. neck cells that either overlap or are 1-hop neighbors of crypt cells.

#     Output
#     ------
#     neck_touching : set[int]
#     """
#     crypt_cells = set(crypt_cells)

#     # 1-hop neighbors of crypt
#     crypt_neighbors = set()
#     for u in crypt_cells:
#         crypt_neighbors.update(G.neighbors(u))

#     neck_touching = set()
#     for neck in neck_patches:
#         neck = set(neck)
#         if (neck & crypt_cells) or (neck & crypt_neighbors):
#             neck_touching |= neck

#     return neck_touching


def crypt_neck_boundary_cells(
    G,                   # nx.Graph
    crypt_cells,         # iterable[int]
    boundary_patches=None,  # list[set[int]] or None (neck + villus patches)
    mode="neck_side",    # "neck_side" or "both_sides"
    restrict_to_patches=True,
):
    """
    Extract the crypt–boundary interface on the CELL GRAPH.

    What you usually want for crypt length normalization is the *interface ring*:
      - neck_side: boundary/villus cells that touch crypt (1-hop neighbors)
      - both_sides: additionally include the crypt cells that touch boundary

    Parameters
    ----------
    boundary_patches:
        Regions you consider "boundary" (e.g. villus + neck). If None, we treat
        all non-crypt cells as potential boundary (usually NOT what you want).
    restrict_to_patches:
        If True, only cells in union(boundary_patches) are considered boundary.

    Returns
    -------
    boundary_cells : set[int]
    """
    crypt_cells = set(crypt_cells)
    if len(crypt_cells) == 0:
        return set()

    if boundary_patches is None:
        boundary_set = set(G.nodes())
        restrict_to_patches = False
    else:
        # boundary_patches is list[set[int]]
        boundary_set = set()
        for p in boundary_patches:
            boundary_set |= set(p)

    # Collect boundary-side neighbors of crypt
    neck_side = set()
    crypt_side = set()

    for u in crypt_cells:
        for v in G.neighbors(u):
            v = int(v)
            if v in crypt_cells:
                continue
            if (not restrict_to_patches) or (v in boundary_set):
                neck_side.add(v)
                crypt_side.add(int(u))

    if mode == "neck_side":
        return neck_side
    if mode == "both_sides":
        return neck_side | crypt_side

    raise ValueError("mode must be 'neck_side' or 'both_sides'.")


# ============================================================
# Crypt bottom selection (single scalar length L*)
# ============================================================

# def find_crypt_bottom(
#     dist_mat,             # (N_cells, V) array, cell->vertex geodesic distances
#     crypt_cells,          # iterable[int], crypt cell indices
#     boundary_vertex_ids,  # (K,) array[int], boundary vertex indices
#     candidates="all",     # "all" or int, subsample crypt candidates
#     score="cv",           # "cv", "iqr_over_median", "range_over_median"
#     length_stat="median", # "median" or "mean"
# ):
#     """
#     Choose crypt bottom cell minimizing variation of distances to boundary vertices.

#     Output
#     ------
#     bottom_cell_id : int
#     L_star : float      (scalar crypt length)
#     score_value : float (variation score)
#     """
#     dist_mat = np.asarray(dist_mat, dtype=float)
#     crypt_idx = np.fromiter(crypt_cells, dtype=np.int64)
#     boundary_vertex_ids = np.asarray(boundary_vertex_ids, dtype=np.int64)

#     if crypt_idx.size == 0 or boundary_vertex_ids.size == 0:
#         raise ValueError("crypt_cells or boundary_vertex_ids empty.")

#     if isinstance(candidates, int) and crypt_idx.size > candidates:
#         pick = np.linspace(0, crypt_idx.size - 1, candidates).astype(int)
#         crypt_idx = crypt_idx[pick]

#     D = dist_mat[crypt_idx][:, boundary_vertex_ids]
#     D = np.where(np.isfinite(D), D, np.nan)

#     if length_stat == "median":
#         L = np.nanmedian(D, axis=1)
#     elif length_stat == "mean":
#         L = np.nanmean(D, axis=1)
#     else:
#         raise ValueError("length_stat must be 'median' or 'mean'.")

#     L_safe = np.maximum(L, 1e-8)

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

#     frac_nan = np.mean(~np.isfinite(D), axis=1)
#     s = np.where(frac_nan > 0.5, np.inf, s)

#     best = int(np.nanargmin(s))
#     return int(crypt_idx[best]), float(L_safe[best]), float(s[best])


def find_crypt_bottom(
    dist_mat,             # (N_cells, V) OR (N_crypt_cells, V) array, geodesic distances
    crypt_cells=None,     # iterable[int] or None. If None: dist_mat already restricted to crypt cells
    boundary_vertex_ids=None,  # (K,) array[int], boundary vertex indices (global vertex ids)
    candidates="all",     # "all" or int, subsample candidate bottoms (over crypt rows)
    score="cv",           # "cv", "iqr_over_median", "range_over_median"
    length_stat="median", # "median" or "mean"
    return_row_index=False, # if True: also return row index into the (possibly restricted) dist_mat
):
    """
    Choose a crypt bottom by minimizing variation of distances from bottom->boundary vertices.

    Two modes
    ---------
    (A) Full dist_mat (all cells):
        - Provide crypt_cells (global cell ids)
        - dist_mat is (N_cells, V)

    (B) Crypt-only dist_mat:
        - Set crypt_cells=None
        - dist_mat is already restricted to crypt cells: shape (N_crypt_cells, V)
        - Returned bottom_cell_id will then be the *row index* into this restricted matrix
          unless you pass crypt_cells explicitly.

    Outputs
    -------
    bottom_cell_id : int
        If crypt_cells is provided: global cell id.
        If crypt_cells is None: row index into the provided dist_mat.
    L_star : float
        Scalar crypt length (median or mean distance to boundary vertices).
    score_value : float
        Variation score for the chosen bottom.
    (optional) bottom_row : int
        Row index into the internal candidate matrix (useful for debugging / mode B).
    """
    dist_mat = np.asarray(dist_mat, dtype=float)
    if boundary_vertex_ids is None:
        raise ValueError("boundary_vertex_ids must be provided.")
    boundary_vertex_ids = np.asarray(boundary_vertex_ids, dtype=np.int64)
    if boundary_vertex_ids.size == 0:
        raise ValueError("boundary_vertex_ids empty.")

    # --- Build candidate row indices ---
    if crypt_cells is None:
        # dist_mat rows are already crypt cells
        crypt_rows = np.arange(dist_mat.shape[0], dtype=np.int64)
        crypt_ids = None
    else:
        crypt_ids = np.fromiter(crypt_cells, dtype=np.int64)
        if crypt_ids.size == 0:
            raise ValueError("crypt_cells empty.")
        crypt_rows = crypt_ids  # rows into full dist_mat

    # Optional subsampling of candidates for speed
    if isinstance(candidates, int) and candidates > 0 and crypt_rows.size > candidates:
        pick = np.linspace(0, crypt_rows.size - 1, candidates).astype(int)
        crypt_rows = crypt_rows[pick]
        if crypt_ids is not None:
            crypt_ids = crypt_ids[pick]

    # --- Distances candidate->boundary vertices ---
    # Works in both modes:
    # - Mode A: dist_mat[crypt_rows] selects those cell rows
    # - Mode B: crypt_rows are 0..N_crypt-1
    D = dist_mat[crypt_rows][:, boundary_vertex_ids]
    D = np.where(np.isfinite(D), D, np.nan)

    # Scalar length per candidate
    if length_stat == "median":
        L = np.nanmedian(D, axis=1)
    elif length_stat == "mean":
        L = np.nanmean(D, axis=1)
    else:
        raise ValueError("length_stat must be 'median' or 'mean'.")

    L_safe = np.maximum(L, 1e-8)

    # Variation score
    if score == "cv":
        s = np.nanstd(D, axis=1) / L_safe
    elif score == "iqr_over_median":
        q75 = np.nanpercentile(D, 75, axis=1)
        q25 = np.nanpercentile(D, 25, axis=1)
        s = (q75 - q25) / L_safe
    elif score == "range_over_median":
        s = (np.nanmax(D, axis=1) - np.nanmin(D, axis=1)) / L_safe
    else:
        raise ValueError("score must be 'cv', 'iqr_over_median', or 'range_over_median'.")

    # Penalize candidates with too many NaNs
    frac_nan = np.mean(~np.isfinite(D), axis=1)
    s = np.where(frac_nan > 0.5, np.inf, s)

    best_row_local = int(np.nanargmin(s))   # index into crypt_rows array
    L_star = float(L_safe[best_row_local])
    score_val = float(s[best_row_local])

    if crypt_cells is None:
        # Return row index into the provided crypt-only dist_mat
        bottom_id = int(crypt_rows[best_row_local])
        if return_row_index:
            return bottom_id, L_star, score_val, bottom_id
        return bottom_id, L_star, score_val

    # crypt_cells provided -> return global cell id
    bottom_cell_id = int(crypt_rows[best_row_local])
    if return_row_index:
        return bottom_cell_id, L_star, score_val, best_row_local
    return bottom_cell_id, L_star, score_val



def get_proj_vertex_ids(G):
    """
    Return proj_vertex_ids : (N_cells,) int array, from node attribute 'proj_vertex'.
    """
    n = G.number_of_nodes()
    return np.array([int(G.nodes[i]["proj_vertex"]) for i in range(n)], dtype=np.int64)


def dijkstra_cellgraph(G, edge_w, source):
    """
    Single-source Dijkstra on a NetworkX-style graph, using a provided edge weight lookup.

    Parameters
    ----------
    G : nx.Graph with nodes 0..N-1
    edge_w : callable(u, v) -> float
        Edge weight function.
    source : int

    Returns
    -------
    dist : (N,) float array
    """
    n = G.number_of_nodes()
    dist = np.full(n, np.inf, dtype=float)
    dist[int(source)] = 0.0

    pq = [(0.0, int(source))]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist[u]:
            continue
        for v in G.neighbors(u):
            v = int(v)
            w = edge_w(u, v)
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def find_crypt_bottom_lightweight(
    G,                    # nx.Graph, cell adjacency already stored
    mesh,                 # mesh with v (V,3)
    crypt_cells,          # iterable[int], crypt cell ids
    neck_boundary_cells,  # iterable[int], neck/villus cells touching crypt
    candidates="all",     # "all" or int, subsample crypt candidates
    score="cv",           # "cv", "iqr_over_median", "range_over_median"
):
    """
    Lightweight crypt bottom selection:

    Minimizes variation of distances from candidate bottom cell to NECK BOUNDARY CELL CENTERS.
    Distances are approximated by shortest paths on the CELL GRAPH using edge weights:
        w(u,v) = || x_u - x_v ||_2
    where x_u is the 3D position of the projected cell-center vertex on the mesh.

    Outputs
    -------
    bottom_cell_id : int
    score_value : float
    """
    crypt_cells = np.asarray(sorted(set(crypt_cells)), dtype=np.int64)
    neck_boundary_cells = np.asarray(sorted(set(neck_boundary_cells)), dtype=np.int64)
    if crypt_cells.size == 0:
        raise ValueError("crypt_cells empty.")
    if neck_boundary_cells.size == 0:
        raise ValueError("neck_boundary_cells empty.")

    # projected mesh vertex for each cell
    proj_vertex_ids = get_proj_vertex_ids(G)
    vpos = np.asarray(mesh.v, dtype=float)

    # edge weight lookup using projected vertex positions
    def edge_w(u, v):
        pu = vpos[proj_vertex_ids[u]]
        pv = vpos[proj_vertex_ids[v]]
        return float(np.linalg.norm(pu - pv))

    # candidate subsampling
    cand = crypt_cells
    if isinstance(candidates, int) and candidates > 0 and cand.size > candidates:
        pick = np.linspace(0, cand.size - 1, candidates).astype(int)
        cand = cand[pick]

    best_id = None
    best_score = np.inf

    for c in cand:
        dist = dijkstra_cellgraph(G, edge_w=edge_w, source=int(c))
        D = dist[neck_boundary_cells]
        D = D[np.isfinite(D)]
        if D.size == 0:
            continue

        med = float(np.median(D))
        med = max(med, 1e-8)

        if score == "cv":
            s = float(np.std(D) / med)
        elif score == "iqr_over_median":
            q75, q25 = np.percentile(D, [75, 25])
            s = float((q75 - q25) / med)
        elif score == "range_over_median":
            s = float((np.max(D) - np.min(D)) / med)
        else:
            raise ValueError("score must be 'cv', 'iqr_over_median', or 'range_over_median'.")

        if s < best_score:
            best_score = s
            best_id = int(c)

    if best_id is None:
        raise RuntimeError("No valid bottom candidate found (graph disconnected?)")

    return best_id, float(best_score)



def calculate_crypt_distance(
    mesh,                    # mesh with v,f,laplacian,mass_matrix
    G,                       # cell graph with node attribute 'proj_vertex'
    bottom_cell_id,          # int, crypt bottom cell id
    neck_boundary_cells,     # iterable[int], neck/villus cells touching crypt
    compute_geodesics_fn,    # function(mesh, sources, t=None) -> (S,V) or (V,)
    t=None,                  # heat time (optional)
):
    """
    Compute crypt distance using a single heat-method solve from the crypt bottom,
    then normalize by mean distance to NECK BOUNDARY CELL CENTERS.

    Outputs
    -------
    dnorm_vertices : (V,) float
        normalized bottom->vertex distance
    dist_vertices : (V,) float
        raw bottom->vertex distance
    dnorm_cellcenters : (N_cells,) float
        normalized distance at cell centers (proj_vertex)
    dist_cellcenters : (N_cells,) float
        raw distance at cell centers (proj_vertex)
    L_mean : float
        mean bottom->neck_boundary_cell_center distance used for normalization
    """
    neck_boundary_cells = np.asarray(sorted(set(neck_boundary_cells)), dtype=np.int64)
    if neck_boundary_cells.size == 0:
        raise ValueError("neck_boundary_cells empty.")

    proj_vertex_ids = get_proj_vertex_ids(G)          # (N_cells,)
    bottom_vertex = int(proj_vertex_ids[int(bottom_cell_id)])

    # --- heat-method geodesics from bottom vertex ---
    D = compute_geodesics_fn(mesh, sources=np.array([bottom_vertex], dtype=np.int64), t=t)
    D = np.asarray(D, dtype=float)
    dist_vertices = D[0] if D.ndim == 2 else D

    # --- evaluate at cell centers ---
    dist_cellcenters = dist_vertices[proj_vertex_ids]   # (N_cells,)

    # --- normalization length based on mean distance to boundary neck cell centers ---
    L_vals = dist_cellcenters[neck_boundary_cells]
    L_vals = L_vals[np.isfinite(L_vals)]
    if L_vals.size == 0:
        raise RuntimeError("Boundary neck cell center distances invalid/disconnected.")

    L_mean = float(np.mean(L_vals))
    L_mean = max(L_mean, 1e-8)

    dnorm_vertices = dist_vertices / L_mean
    dnorm_cellcenters = dist_cellcenters / L_mean

    return dnorm_vertices, dist_vertices, dnorm_cellcenters, dist_cellcenters, L_mean



# ============================================================
# Distance normalization
# ============================================================

def normalize_crypt_length(
    dist_mat,        # (N_cells, V) array, geodesic distances
    bottom_cell_idx,  # int, crypt bottom cell index
    L_star,          # float, scalar crypt length
    vertex_owner=None, # (V,) array or None, owning cell index per vertex
):
    """
    Compute normalized distance to crypt bottom.

    Output
    ------
    dnorm_vertices : (V,) array
    dnorm_cells : (N_cells,) array or None
    """
    dist_mat = np.asarray(dist_mat, dtype=float)
    L_star = max(float(L_star), 1e-8)

    dist_bottom = dist_mat[int(bottom_cell_idx), :]
    dnorm_v = dist_bottom / L_star

    if vertex_owner is None:
        return dnorm_v, None

    vertex_owner = np.asarray(vertex_owner, dtype=np.int64)
    n_cells = dist_mat.shape[0]
    dnorm_c = np.full(n_cells, np.nan)

    for c in range(n_cells):
        mask = (vertex_owner == c)
        if np.any(mask):
            vals = dnorm_v[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                dnorm_c[c] = np.mean(vals)

    return dnorm_v, dnorm_c


# ============================================================
# Circumference from iso-contours
# ============================================================

def crypt_circumference(
    mesh,           # mesh with v (V,3), f (F,3)
    crypt_dist,     # (V,) distance to crypt bottom for each vertex
    levels,         # (K,) array, iso-contour levels
):
    """
    Compute iso-contour lengths (circumference) of a scalar field on the mesh.

    Output
    ------
    C : (K,) array
    """
    v = np.asarray(mesh.v, dtype=float)
    f = np.asarray(mesh.f, dtype=np.int64)
    s = np.asarray(crypt_dist, dtype=float)
    levels = np.asarray(levels, dtype=float)

    tri_pos = v[f]
    tri_s = s[f]
    tri_min = np.min(tri_s, axis=1)
    tri_max = np.max(tri_s, axis=1)

    C = np.zeros(len(levels), dtype=float)

    def tri_isoseg(p, sval, t):
        inter = []
        a, b, c = sval

        if (a < t and b > t) or (a > t and b < t):
            w = (t - a) / (b - a)
            inter.append(p[0] + w * (p[1] - p[0]))
        if (b < t and c > t) or (b > t and c < t):
            w = (t - b) / (c - b)
            inter.append(p[1] + w * (p[2] - p[1]))
        if (c < t and a > t) or (c > t and a < t):
            w = (t - c) / (a - c)
            inter.append(p[2] + w * (p[0] - p[2]))

        return inter if len(inter) == 2 else None

    for li, t in enumerate(levels):
        active = (tri_min <= t) & (tri_max >= t)
        total = 0.0
        for p, sval in zip(tri_pos[active], tri_s[active]):
            seg = tri_isoseg(p, sval, t)
            if seg is not None:
                total += np.linalg.norm(seg[1] - seg[0])
        C[li] = total

    return C


# ============================================================
# Marker binning
# ============================================================

def bin_marker_positivity(
    markers,   # (N_cells, M) array, marker positivity per cell
    distance,  # (N_cells,) array, distance coordinate per cell
    bin_edges, # (B+1,) array, bin edges
):
    """
    Bin marker positivity vs distance.

    Output
    ------
    counts_pos : (M, B) array
    counts_total : (B,) array
    """
    markers = np.asarray(markers)
    distance = np.asarray(distance)
    bin_edges = np.asarray(bin_edges)

    N, M = markers.shape
    B = len(bin_edges) - 1
    bin_ids = np.digitize(distance, bin_edges) - 1

    counts_pos = np.zeros((M, B), dtype=int)
    counts_total = np.zeros(B, dtype=int)

    for b in range(B):
        mask = (bin_ids == b)
        if np.any(mask):
            counts_total[b] = mask.sum()
            counts_pos[:, b] = markers[mask].astype(bool).sum(axis=0)

    return counts_pos, counts_total


# ============================================================
# Anchored correlation
# ============================================================

def crypt_anchored_correlation(
    X_vertices,              # (V,) array, field on vertices (e.g. HKS at fixed time)
    crypt_dist,             # (V,) array, distance from crypt bottom to vertices
    bin_edges,               # (B+1,) array, distance bins
    weights=None,            # (V,) array or None, vertex weights (e.g. areas)
    connected=True,          # bool, subtract global mean if True
    normalized=False,        # bool, divide by global variance if True
    exclude_self=True,       # bool, exclude bottom vertex from bins
):
    """
    Compute bottom-anchored two-point correlation vs distance.

    Output
    ------
    corr : (B,) array
    counts : (B,) array
    bin_centers : (B,) array
    """
    X = np.asarray(X_vertices, dtype=float)
    d = np.asarray(crypt_dist, dtype=float)
    be = np.asarray(bin_edges, dtype=float)

    B = len(be) - 1
    centers = 0.5 * (be[:-1] + be[1:])

    w = np.ones_like(X) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(X) & np.isfinite(d) & np.isfinite(w)

    bottom_v = int(np.nanargmin(d))
    if exclude_self:
        valid[bottom_v] = False

    mu = np.average(X[valid], weights=w[valid])
    var = np.average((X[valid] - mu) ** 2, weights=w[valid])
    var = max(var, 1e-12)

    Xb = X[bottom_v]
    if connected:
        Xb_eff = Xb - mu
        Y = X - mu
    else:
        Xb_eff = Xb
        Y = X

    bin_ids = np.digitize(d, be) - 1
    corr = np.full(B, np.nan)
    counts = np.zeros(B)

    for b in range(B):
        m = valid & (bin_ids == b)
        if np.any(m):
            counts[b] = w[m].sum()
            corr[b] = Xb_eff * np.sum(w[m] * Y[m]) / counts[b]

    if normalized:
        corr /= var

    return corr, counts, centers



def field_stats_along_crypt(
    mesh_field,        # (V,) field on mesh vertices
    s_vertices,        # (V,) normalized distance per vertex (e.g., dnorm_vertices)
    bin_edges,         # (B+1,) shared bin edges
    weights=None,      # (V,) optional per-vertex weights (e.g., vertex areas)
):
    """
    Compute mean and std of a vertex-defined field as a function of s using shared bins.

    For each bin b spanning (bin_edges[b], bin_edges[b+1]]:
      - collect vertices whose s falls in that bin
      - compute (weighted) mean and std of mesh_field over those vertices

    Parameters
    ----------
    mesh_field : array-like, shape (V,)
        Scalar field defined on mesh vertices.
    s_vertices : array-like, shape (V,)
        Normalized distances for vertices (same length as mesh_field).
    bin_edges : array-like, shape (B+1,)
        Bin edges along s (shared across crypts/meshes).
    weights : array-like, shape (V,), optional
        Nonnegative weights per vertex. If provided, computes weighted mean/std.

    Returns
    -------
    mean : (B,) float
    std : (B,) float
    count : (B,) int
        Number of vertices used per bin (after finite-value masking).
    """
    x = np.asarray(mesh_field, dtype=float)
    s = np.asarray(s_vertices, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)

    if x.shape != s.shape:
        raise ValueError("mesh_field and s_vertices must have the same shape.")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be 1D with length >= 2.")

    B = edges.size - 1
    mean = np.full(B, np.nan, dtype=float)
    std  = np.full(B, np.nan, dtype=float)
    count = np.zeros(B, dtype=int)

    if weights is None:
        w = None
        m = np.isfinite(x) & np.isfinite(s)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != x.shape:
            raise ValueError("weights must have the same shape as mesh_field.")
        m = np.isfinite(x) & np.isfinite(s) & np.isfinite(w) & (w >= 0)

    if not np.any(m):
        return mean, std, count

    x = x[m]
    s = s[m]
    if w is not None:
        w = w[m]

    # Bin assignment: b in [0, B-1]
    # Use right-open bins [edge_i, edge_{i+1}) except include the last edge in last bin.
    b = np.searchsorted(edges, s, side="right") - 1
    valid = (b >= 0) & (b < B)
    x = x[valid]
    b = b[valid]
    if w is not None:
        w = w[valid]

    # Counts
    count = np.bincount(b, minlength=B).astype(int)

    if w is None:
        # sums and sums of squares per bin
        sx = np.bincount(b, weights=x, minlength=B)
        sx2 = np.bincount(b, weights=x*x, minlength=B)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sx / count
            var = sx2 / count - mean*mean
            var = np.maximum(var, 0.0)
            std = np.sqrt(var)
        mean[count == 0] = np.nan
        std[count == 0] = np.nan
        return mean, std, count

    # Weighted
    sw = np.bincount(b, weights=w, minlength=B)
    sxw = np.bincount(b, weights=w*x, minlength=B)
    sx2w = np.bincount(b, weights=w*x*x, minlength=B)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sxw / sw
        var = sx2w / sw - mean*mean
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

    mean[sw == 0] = np.nan
    std[sw == 0] = np.nan
    return mean, std, count



# ============================================================
# Vertex area weights
# ============================================================

def vertex_areas(mesh):  # mesh with mass_matrix
    """Return per-vertex area weights from mesh.mass_matrix diagonal."""
    M = mesh.mass_matrix
    if hasattr(M, "diagonal"):
        return np.asarray(M.diagonal(), dtype=float)
    return np.asarray(np.diag(M), dtype=float)



# ===========================================================
# Crypt length adjustment
# ===========================================================


from scipy.signal import savgol_filter

def adjust_cryptlength_by_circumference(
    s,
    C,
    search_interval=(0.75, 1.25),
    window_length=9,
    polyorder=3,
    min_prominence=0.0,
):
    """
    Return s_star (new "crypt length" in normalized units) from a circumference profile C(s).

    Logic (always smoothed, always prefer minimum):
      1) Smooth C(s) with Savitzky–Golay.
      2) Look for local minima of C in search_interval. If any, pick the deepest
         (optionally requiring a minimum relative prominence) and return s_star at that point.
      3) Otherwise look for an "inflection point": a local maximum of d2C/ds2 in the
         interval with dC/ds > 0. Pick the strongest and return its s as s_star.
      4) If neither exists, return 1.0 (i.e., keep distances as-is).

    """
    s = np.asarray(s, dtype=float)
    C = np.asarray(C, dtype=float)
    if s.ndim != 1 or C.ndim != 1 or s.size != C.size:
        raise ValueError("s and C must be 1D arrays with the same length.")
    if s.size < 7:
        return 1.0

    # sort by s
    order = np.argsort(s)
    s0 = s[order]
    C0 = C[order]

    # fill NaNs in C
    m = np.isfinite(C0)
    if np.sum(m) < 7:
        return 1.0
    Cfill = np.interp(s0, s0[m], C0[m])

    # Savitzky–Golay smoothing (good for derivatives)
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, s0.size if s0.size % 2 == 1 else s0.size - 1)
    wl = max(wl, 5)
    po = int(polyorder)
    po = min(po, wl - 2)
    Cs = savgol_filter(Cfill, window_length=wl, polyorder=po, mode="interp")

    # derivatives
    d1 = np.gradient(Cs, s0)
    d2 = np.gradient(d1, s0)

    lo, hi = map(float, search_interval)
    if lo > hi:
        lo, hi = hi, lo
    win = np.where((s0 >= lo) & (s0 <= hi))[0]
    if win.size < 3:
        return 1.0

    # local minima within window
    idx = win[(win > 0) & (win < len(Cs) - 1)]
    minima = idx[(Cs[idx - 1] > Cs[idx]) & (Cs[idx] < Cs[idx + 1])]

    # optional prominence filter
    if minima.size and min_prominence > 0:
        keep = []
        for i in minima:
            left = win[win <= i]
            right = win[win >= i]
            if left.size == 0 or right.size == 0:
                continue
            mref = max(np.max(Cs[left]), np.max(Cs[right]), 1e-12)
            prom = (mref - Cs[i]) / mref
            if prom >= float(min_prominence):
                keep.append(i)
        minima = np.asarray(keep, dtype=np.int64)

    if minima.size:
        i_star = minima[np.argmin(Cs[minima])]  # deepest minimum
        s_star = float(s0[i_star])
        return s_star if (np.isfinite(s_star) and s_star > 0) else 1.0

    # fallback: inflection = local max of d2 with d1>0
    cand = idx[d1[idx] > 0]
    infl = cand[(d2[cand - 1] < d2[cand]) & (d2[cand] > d2[cand + 1])]

    if infl.size:
        i_star = infl[np.argmax(d2[infl])]
        s_star = float(s0[i_star])
        return s_star if (np.isfinite(s_star) and s_star > 0) else 1.0

    return 1.0


def assign_crypts_by_neckline(dnorm_cells_per_crypt, s_thresh=1.0):
    """
    Assign each cell to at most one crypt using the rule:
      - cell is eligible for crypt k if dnorm_cells_per_crypt[k, cell] < s_thresh
      - if eligible for multiple crypts, assign to the crypt with minimal distance

    Parameters
    ----------
    dnorm_cells_per_crypt : array, shape (K, N_cells)
        For each crypt k, the adjusted normalized distance s for each cell center.
        (This should already include your division by s_star.)
    s_thresh : float
        Threshold for membership (default 1.0).

    Returns
    -------
    crypt_patches_new : list[set[int]]
        Updated crypt patches as disjoint sets of cell ids (length K).
    best_crypt : (N_cells,) int
        Assigned crypt index per cell, or -1 if unassigned.
    best_dist : (N_cells,) float
        Best distance per cell (inf if unassigned).
    """
    D = np.asarray(dnorm_cells_per_crypt, dtype=float)
    if D.ndim != 2:
        raise ValueError("dnorm_cells_per_crypt must be a 2D array of shape (K, N_cells).")

    K, N = D.shape
    best_dist = np.full(N, np.inf, dtype=float)
    best_crypt = np.full(N, -1, dtype=int)

    for k in range(K):
        dk = D[k]
        m = np.isfinite(dk) & (dk < float(s_thresh)) & (dk < best_dist)
        best_dist[m] = dk[m]
        best_crypt[m] = k

    crypt_patches_new = [set(np.where(best_crypt == k)[0].tolist()) for k in range(K)]
    return crypt_patches_new, best_crypt, best_dist


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
