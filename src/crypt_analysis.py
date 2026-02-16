

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


def compute_crypt_axis(
    G,
    mesh,
    crypt_patches,
    boundary_patches,
    geodesic_fn,
    geodesic_kwargs=None,
):
    """
    Compute per-feature (crypt) distance fields from feature bottoms.

    Inputs
    ------
    G : networkx.Graph
        Cell graph. Nodes are cells (0..N-1). Must store:
        - "proj_vertex" : projected mesh vertex id for each cell center
    mesh : OrganoidMesh
        Surface mesh with attributes:
        - v : (V,3) vertices
        - f : (F,3) faces
    crypt_patches : list[set[int]]
        Cell-id sets defining each feature (crypt).
    boundary_patches : list[set[int]]
        Cell-id sets defining boundary regions used to normalize length.
    geodesic_fn : callable
        Geodesic routine called as geodesic_fn(mesh, sources=[...], **kwargs)
        Must return (S,V) or (V,) distances.
    geodesic_kwargs : dict or None
        Extra keyword args forwarded to geodesic_fn.

    Returns
    -------
    draw_vertices_all : (K, V) float
        Raw geodesic distances from each feature bottom to all vertices.
        NaN for invalid features.
    dnorm_vertices_all : (K, V) float
        Normalized vertex distances (divided by boundary mean length).
        NaN for invalid features.
    L_mean_all : (K,) float
        Mean boundary distance per feature (normalization length).
        NaN for invalid features.
    bottom_vertex_ids : (K,) int
        Bottom projected vertex id per feature (-1 if invalid).
    """
    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    K = len(crypt_patches)
    V = mesh.v.shape[0]

    bottom_cell_ids = np.full(K, -1, dtype=np.int32)
    bottom_vertex_ids = np.full(K, -1, dtype=np.int32)
    boundary_cells_list = [None] * K

    # --- 1) find bottom + boundary per feature ---
    for j, crypt_cells in enumerate(crypt_patches):
        if not crypt_cells:
            continue

        boundary_cells = crypt_neck_boundary_cells(G, crypt_cells, boundary_patches)
        if not boundary_cells:
            continue

        bottom_cell_id, _ = find_crypt_bottom_lightweight(
            G=G,
            mesh=mesh,
            crypt_cells=crypt_cells,
            neck_boundary_cells=boundary_cells,
        )

        bottom_cell_ids[j] = bottom_cell_id
        bottom_vertex_ids[j] = G.nodes[bottom_cell_id]["proj_vertex"]
        boundary_cells_list[j] = list(boundary_cells)

    valid_js = np.where(bottom_vertex_ids >= 0)[0]

    draw_vertices_all = np.full((K, V), np.nan, dtype=float)
    dnorm_vertices_all = np.full((K, V), np.nan, dtype=float)
    L_mean_all = np.full(K, np.nan, dtype=float)

    if len(valid_js) == 0:
        return draw_vertices_all, dnorm_vertices_all, L_mean_all, bottom_vertex_ids

    # --- 2) geodesics from all valid bottoms ---
    sources = [int(s) for s in bottom_vertex_ids[valid_js]]
    D_multi = np.asarray(geodesic_fn(mesh, sources=sources, **geodesic_kwargs))
    if D_multi.ndim == 1:
        D_multi = D_multi[None, :]

    # --- 3) normalize per feature ---
    for r, j in enumerate(valid_js):
        dist_vertices = D_multi[r]  # (V,)
        draw_vertices_all[j] = dist_vertices

        boundary_vertex_ids = graph_get(
            G, "proj_vertex",
            nodes=boundary_cells_list[j],
            dtype=np.int32
        )

        L_mean = float(np.mean(dist_vertices[boundary_vertex_ids]))
        L_mean = max(L_mean, 1e-12)

        dnorm_vertices_all[j] = dist_vertices / L_mean
        L_mean_all[j] = L_mean

    return draw_vertices_all, dnorm_vertices_all, L_mean_all, bottom_vertex_ids





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

def norm_dist_to_neckline(
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



def rescale_crypt_axis_by_circumference(
    mesh,
    dnorm_vertices,
    bin_centers,
    search_interval,
    L_mean=None,
    window_length=9,
    polyorder=3,
    min_prominence=0.05,
):
    """
    Rescale feature axis so that the narrowest circumference occurs at s=1,
    without recomputing circumference on the mesh after rescaling.

    Inputs
    ------
    mesh : OrganoidMesh
        Mesh used by crypt_circumference().
    dnorm_vertices : (V,) or (K,V) array
        Normalized distances at mesh vertices (per feature).
    bin_centers : (B,) array
        Canonical axis levels (used elsewhere; only axis exposed to caller).
    search_interval : (lo, hi)
        Interval for s_star search. Also defines internal axis stretching via hi.
    L_mean : optional, scalar or (K,) array
        Length proxy in physical units (per feature). Rescaled as L_mean * s_star.
    window_length, polyorder, min_prominence :
        Passed to norm_dist_to_neckline().

    Returns (ALWAYS BATCHED)
    -----------------------
    CC_rescaled : (K, B) float
        Circumference curves sampled on bin_centers after axis rescaling.
    dnorm_vertices_rescaled : (K, V) float
        Rescaled vertex distances: dnorm_vertices / s_star.
    L_mean_rescaled : (K,) float or None
        Rescaled length proxy: L_mean * s_star (or None if L_mean not given).
    """
    s = np.asarray(bin_centers, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("bin_centers must be a non-empty 1D array")

    lo, hi = float(search_interval[0]), float(search_interval[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError("search_interval must be (lo, hi) with 0 < lo < hi")

    dv = np.asarray(dnorm_vertices, dtype=float)
    if dv.ndim == 1:
        dv = dv[None, :]  # (1, V)
    elif dv.ndim != 2:
        raise ValueError("dnorm_vertices must be shape (V,) or (K,V)")

    K, V = dv.shape

    Lm = None
    if L_mean is not None:
        Lm = np.asarray(L_mean, dtype=float)
        if Lm.ndim == 0:
            Lm = np.full(K, float(Lm), dtype=float)
        if Lm.ndim != 1 or Lm.shape[0] != K:
            raise ValueError("L_mean must be scalar or shape (K,) matching dnorm_vertices' K")

    # wide axis to avoid extrapolation for s_query = s * s_star
    s_wide = np.linspace(0.0, float(s.max()) * hi, s.size)

    CC_rescaled = np.full((K, s.size), np.nan, dtype=float)
    s_star = np.ones(K, dtype=float)

    for k in range(K):
        CC0 = np.asarray(crypt_circumference(mesh=mesh, crypt_dist=dv[k], levels=s_wide), dtype=float)

        ss = norm_dist_to_neckline(
            s=s_wide,
            C=CC0,
            search_interval=(lo, hi),
            window_length=window_length,
            polyorder=polyorder,
            min_prominence=min_prominence,
        )
        if not (np.isfinite(ss) and ss > 0):
            ss = 1.0
        s_star[k] = ss

        s_query = np.clip(s * ss, s_wide[0], s_wide[-1])
        CC_rescaled[k] = np.interp(s_query, s_wide, CC0)

    dv_rescaled = dv / np.maximum(s_star[:, None], 1e-12)
    Lm_rescaled = (Lm * s_star) if Lm is not None else None

    return CC_rescaled, dv_rescaled, Lm_rescaled




def assign_features_by_distance(dnorm_per_feature, s_thresh=1.0):
    """
    Assign each item (cell, vertex, etc.) to at most one feature using
    a normalized-distance threshold and nearest-feature rule.

    Rule
    ----
    An item i is eligible for feature k if:
        dnorm_per_feature[k, i] < s_thresh

    If multiple features qualify, the item is assigned to the feature
    with the smallest distance.

    This works identically whether “items” are:
      - cells  → distances at cell centers
      - vertices → distances at mesh vertices
      - any other indexed objects

    Parameters
    ----------
    dnorm_per_feature : array, shape (K, N_items)
        Normalized distances from each feature k to each item i.
        Example:
            K = number of crypts/features
            N_items = number of cells OR vertices
        Distances should already include any axis rescaling (e.g. / s_star).
    s_thresh : float
        Threshold for membership (default 1.0).

    Returns
    -------
    feature_patches : list[set[int]]
        Disjoint sets of assigned item indices, one set per feature (length K).
    best_feature : (N_items,) int
        Assigned feature index per item, or -1 if unassigned.
    best_dist : (N_items,) float
        Winning (smallest) distance per item, or +inf if unassigned.
    """
    D = np.asarray(dnorm_per_feature, dtype=float)
    if D.ndim != 2:
        raise ValueError("dnorm_per_feature must have shape (K, N_items)")

    K, N_items = D.shape

    best_dist = np.full(N_items, np.inf, dtype=float)
    best_feature = np.full(N_items, -1, dtype=int)

    for k in range(K):
        dk = D[k]
        mask = np.isfinite(dk) & (dk < float(s_thresh)) & (dk < best_dist)
        best_dist[mask] = dk[mask]
        best_feature[mask] = k

    feature_patches = [
        set(np.where(best_feature == k)[0].tolist())
        for k in range(K)
    ]

    return feature_patches, best_feature, best_dist



from src.mesh_analysis import compute_geodesics_dijkstra
from src.crypt_extraction import seed_regions_by_vocab, grow_crypts_toward_necks

def segment_crypts_organoid(
    G,                     # networkx cell-graph
    mesh,                  # OrganoidMesh 
    bin_centers,           # (B,) axis used for circumference curves + rescaling
    crypt_vocab_idx,       # iterable[int] indices into vocab_encoding indicating "crypt" vocab bins
    neck_vocab_idx=None,   # iterable[int] or None; indices indicating "neck" vocab bins
    crypt_seed_thresh=0.2, # threshold on max(vocab_encoding[crypt_vocab_idx]) to seed crypt
    neck_seed_thresh=0.5,  # threshold on max(vocab_encoding[neck_vocab_idx]) to seed neck
    min_crypt_seed_size=10,# minimum connected-component size for crypt (and neck if enabled)
    grow_steps=2,          # iterations for grow_crypts_toward_necks; set 0 to disable
    geodesic_fn=compute_geodesics_dijkstra,  # geodesics(mesh, sources=[...], **geodesic_kwargs)
    geodesic_kwargs=None,  # dict of kwargs forwarded to geodesic_fn
    search_interval=(0.75, 1.25),  # allowed stretch range for finding min circumference
    window_length=9,       # smoothing window for adjust_cryptlength_by_circumference
    polyorder=3,           # polynomial order for adjust_cryptlength_by_circumference smoothing
    min_prominence=0.05,   # peak prominence threshold for adjust_cryptlength_by_circumference
    debug=False,           # if True, also return intermediate regions + internals
):
    """
    One-stop organoid segmentation + crypt-axis computation.

    Pipeline
    --------
    1) seed crypt/neck/villus regions from vocab_encoding
    2) optionally grow crypts toward necks
    3) compute crypt axis (raw + normalized) using geodesics from crypt bottoms
    4) rescale axis by circumference so minimum aligns with s=1 on bin_centers
    5) map vertex distances to cell-center distances
    6) finalize crypt membership by neckline (s_thresh=1.0)
    7) villus_final = all cells not assigned to any crypt

    Returns
    -------
    crypts_final : list[set[int]]
    villi_final  : list[set[int]]          # single patch: all non-crypt cells
    dnorm_v      : (K, V) float            # rescaled normalized distances at vertices
    draw_v       : (K, V) float            # raw geodesic distances at vertices
    L_crypt       : (K,) float             # mean boundary distance (rescaled consistently with dnorm_v)
    C            : (K, B) float            # circumference curves aligned to bin_centers
    dbg          : dict (only if debug=True)
    """
    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    # --- 1) seed regions ---
    crypts_seed, necks_seed, villi_seed = seed_regions_by_vocab(
        G,
        crypt_vocab_idx=crypt_vocab_idx,
        crypt_thresh=crypt_seed_thresh,
        min_crypt_region_size=min_crypt_seed_size,
        neck_vocab_idx=neck_vocab_idx,
        neck_thresh=neck_seed_thresh,
    )

    # --- 2) optionally grow/refine ---
    if grow_steps and grow_steps > 0:
        crypts_grow, necks_grow, villi_grow = grow_crypts_toward_necks(
            G,
            crypt_regions=crypts_seed,
            neck_regions=necks_seed,
            N=grow_steps,
            min_villus_region_size=1,  # fixed default
        )
    else:
        crypts_grow, necks_grow, villi_grow = crypts_seed, necks_seed, villi_seed

    boundary_patches = villi_grow + necks_grow

    # --- 3) compute crypt axis (raw + normalized) ---
    draw_v, dnorm_v, L_crypt, bottom_vertex_ids = compute_crypt_axis(
        G=G,
        mesh=mesh,
        crypt_patches=crypts_grow,
        boundary_patches=boundary_patches,
        geodesic_fn=geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
    )

    # --- 4) rescale by circumference (always returns (K,B), (K,V), (K,)) ---
    Circ, dnorm_v, L_crypt = rescale_crypt_axis_by_circumference(
        mesh=mesh,
        dnorm_vertices=dnorm_v,
        bin_centers=bin_centers,
        search_interval=search_interval,
        L_mean=L_crypt,
        window_length=window_length,
        polyorder=polyorder,
        min_prominence=min_prominence,
    )

    # --- 5) vertex -> cell center distances ---
    proj_vertex_ids = graph_get(G, "proj_vertex", dtype=np.int32)
    dnorm_c = dnorm_v[:, proj_vertex_ids]  # (K, N_cells)

    # --- 6) finalize crypt membership by neckline (default s_thresh=1.0) ---
    crypts_final, best_feature, best_dist = assign_features_by_distance(dnorm_c)

    # --- 7) villus = all cells not in any crypt (single patch) ---
    N_cells = G.number_of_nodes()
    crypt_union = set().union(*crypts_final) if crypts_final else set()
    villus_cells = set(range(N_cells)) - crypt_union
    villi_final = [villus_cells] if villus_cells else []

    if not debug:
        return crypts_final, villi_final, dnorm_v, draw_v, L_crypt, Circ

    dbg = {
        "crypts_seed": crypts_seed,
        "necks_seed": necks_seed,
        "villi_seed": villi_seed,
        "crypts_grow": crypts_grow,
        "necks_grow": necks_grow,
        "villi_grow": villi_grow,
        "boundary_patches": boundary_patches,
        "bottom_vertex_ids": bottom_vertex_ids,
        "best_feature": best_feature,  # (N_cells,)
        "best_dist": best_dist,        # (N_cells,)
    }

    return crypts_final, villi_final, dnorm_v, draw_v, L_crypt, Circ, dbg







# ===========================================================
# Filtering of crypt candidates based on marker content
# ===========================================================

def filter_crypt_by_markers(
    G,
    crypt_cells,
    pos_markers=None,     # list[int]
    neg_markers=None,     # list[int]
    pos_min=1,            # threshold: count OR fraction (see mode)
    neg_min=1,            # threshold: count OR fraction (see mode)
    roi_frac=None,        # e.g. 0.3 -> use cells with dist_bottom <= 0.3
    dist_bottom=None,     # (N_cells,) or (K, N_cells) normalized distances
    require_all_pos=True,
    mode="count",         # "count" (default) or "frac"/"percent"
):
    """
    Filter crypt(s) by marker content.

    Supports either:
      - mode="count": thresholds are absolute numbers of ROI cells
      - mode="frac" / "percent": thresholds are fractions of ROI cells (0..1)

    Inputs
    ------
    G : networkx.Graph
        Node attribute "markers_bin" must be array-like of length M per cell.
    crypt_cells : set[int] OR list[set[int]]
        Cell-id indices belonging to one crypt or many crypts.
    pos_markers, neg_markers : list[int] or None
        Marker indices. Positive markers must be present in enough ROI cells;
        negative markers reject if present in enough ROI cells.
    pos_min : float or int
        If mode="count": minimum number of ROI cells positive for each pos marker.
        If mode="frac": minimum fraction of ROI cells positive for each pos marker.
        (Better name: pos_min_count / pos_min_frac)
    neg_min : float or int
        If mode="count": reject if >= this many ROI cells positive for each neg marker.
        If mode="frac": reject if >= this fraction of ROI cells positive for each neg marker.
        (Better name: neg_min_count / neg_min_frac)
    roi_frac : float or None
        If given, restrict ROI to crypt cells with dist_bottom <= roi_frac.
    dist_bottom : array or None
        Normalized distance(s) used for ROI selection.
        - single crypt: (N_cells,)
        - many crypts: (K, N_cells) matching crypt order
    require_all_pos : bool
        If True: all pos_markers must pass. If False: at least one pos_marker must pass.
    mode : str
        "count" or "frac"/"percent"

    Returns
    -------
    keep : bool OR np.ndarray of bool shape (K,)
        Whether each crypt passes the filter.
    """
    mode = str(mode).lower()
    if mode in ("fraction", "fractions", "frac", "percent", "percentage"):
        mode = "frac"
    elif mode != "count":
        raise ValueError("mode must be 'count' or 'frac'/'percent'")

    # ---- normalize crypt input ----
    single = isinstance(crypt_cells, set)
    crypt_list = [crypt_cells] if single else (list(crypt_cells) if crypt_cells is not None else [])
    K = len(crypt_list)

    if K == 0:
        return False if single else np.zeros(0, dtype=bool)

    pos_markers = [] if pos_markers is None else [int(k) for k in pos_markers]
    neg_markers = [] if neg_markers is None else [int(k) for k in neg_markers]

    # Nothing to filter on => keep non-empty crypts
    if not pos_markers and not neg_markers:
        out = np.array([len(p) > 0 for p in crypt_list], dtype=bool)
        return bool(out[0]) if single else out

    if mode == "frac":
        # interpret thresholds as fractions
        if not (0.0 <= float(pos_min) <= 1.0) or not (0.0 <= float(neg_min) <= 1.0):
            raise ValueError("In mode='frac', pos_min and neg_min must be fractions in [0,1].")

    # Pull markers once: (N_cells, M)
    N = G.number_of_nodes()
    markers = np.asarray([G.nodes[i]["markers_bin"] for i in range(N)])
    # If markers are bool/0-1, sum(axis=0) gives counts.

    # dist_bottom normalization: allow (N,) or (K,N)
    Db = None
    if roi_frac is not None:
        if dist_bottom is None:
            raise ValueError("dist_bottom required when roi_frac is used.")
        Db = np.asarray(dist_bottom, float)
        if Db.ndim == 1:
            Db = np.broadcast_to(Db[None, :], (K, Db.shape[0]))
        if Db.ndim != 2 or Db.shape[0] != K or Db.shape[1] != N:
            raise ValueError("dist_bottom must be shape (N_cells,) or (K, N_cells) matching crypt_cells.")

    keep = np.zeros(K, dtype=bool)

    for j, patch in enumerate(crypt_list):
        if not patch:
            continue

        idx = np.fromiter(patch, dtype=np.int64)
        if idx.size == 0:
            continue

        # ROI selection
        if roi_frac is not None:
            dj = Db[j, idx]
            roi = idx[np.isfinite(dj) & (dj <= float(roi_frac))]
            if roi.size == 0:
                continue
        else:
            roi = idx

        n_roi = int(roi.size)

        # counts per marker index among ROI cells
        counts = markers[roi].sum(axis=0)  # (M,)

        # Convert thresholds depending on mode
        if mode == "count":
            pos_thr = float(pos_min)
            neg_thr = float(neg_min)
        else:
            # fraction thresholds -> convert to counts for consistent comparisons
            pos_thr = float(pos_min) * n_roi
            neg_thr = float(neg_min) * n_roi

        # Positive marker rule
        if pos_markers:
            ok = [(counts[k] >= pos_thr) for k in pos_markers]
            if require_all_pos:
                if not all(ok):
                    continue
            else:
                if not any(ok):
                    continue

        # Negative marker rule
        if any(counts[k] >= neg_thr for k in neg_markers):
            continue

        keep[j] = True

    return bool(keep[0]) if single else keep


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

def compute_BI(
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
