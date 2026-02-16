import numpy as np
import heapq

def build_edge_adjacency(mesh_v: np.ndarray, mesh_f: np.ndarray):
    """
    Build undirected adjacency list with Euclidean edge lengths.
    Returns: nbrs, wts
      nbrs[i] = array of neighbor vertex indices
      wts[i]  = array of corresponding edge lengths
    """
    v = np.asarray(mesh_v, dtype=np.float64)
    f = np.asarray(mesh_f, dtype=np.int64)

    # All directed edges from faces (a->b, b->c, c->a) and the reverse
    e01 = f[:, [0, 1]]
    e12 = f[:, [1, 2]]
    e20 = f[:, [2, 0]]
    E = np.vstack([e01, e12, e20])
    E = np.vstack([E, E[:, ::-1]])  # add reverse edges

    I = E[:, 0]
    J = E[:, 1]

    # Compute edge lengths
    W = np.linalg.norm(v[I] - v[J], axis=1)

    # Sort by (I, J) so we can collapse duplicates by keeping min weight
    order = np.lexsort((J, I))
    I = I[order]; J = J[order]; W = W[order]

    # Collapse duplicate (I,J) edges
    same = (I[1:] == I[:-1]) & (J[1:] == J[:-1])
    keep = np.ones(len(I), dtype=bool)
    keep[1:][same] = False  # keep first occurrence
    # For duplicates, first occurrence may not be min; take min via reduceat
    # Find starts of each (I,J) group
    group_starts = np.r_[0, np.where(~same)[0] + 1]
    Wmin = np.minimum.reduceat(W, group_starts)

    Iu = I[keep]
    Ju = J[keep]
    Wu = Wmin  # aligned with kept first-of-group edges

    Vn = v.shape[0]
    # Build adjacency lists
    counts = np.bincount(Iu, minlength=Vn)
    offsets = np.cumsum(np.r_[0, counts])

    nbrs = [None] * Vn
    wts  = [None] * Vn

    # Fill contiguous arrays then slice
    nbr_all = np.empty(len(Iu), dtype=np.int64)
    wt_all  = np.empty(len(Iu), dtype=np.float64)

    cursor = offsets[:-1].copy()
    for a, b, w in zip(Iu, Ju, Wu):
        k = cursor[a]
        nbr_all[k] = b
        wt_all[k] = w
        cursor[a] += 1

    for i in range(Vn):
        a, b = offsets[i], offsets[i+1]
        nbrs[i] = nbr_all[a:b]
        wts[i]  = wt_all[a:b]

    return nbrs, wts


def voronoi_on_mesh_multisource(nbrs, wts, sources):
    """
    Multi-source Dijkstra (geodesic Voronoi on the edge graph).

    Parameters
    ----------
    nbrs, wts : adjacency lists from build_edge_adjacency
    sources : (N_cells,) int
        Projected vertex id per cell. Rejected cells should have -1.

    Returns
    -------
    owner_cell : (V,) int64
        For each mesh vertex, the owning *cell index* (0..N_cells-1).
        -1 means no owner (e.g. all sources rejected).
    dist : (V,) float64
        Distance to the nearest source (edge-graph geodesic).
        np.inf if owner_cell == -1.
    """
    sources = np.asarray(sources, dtype=np.int64)
    V = len(nbrs)

    dist = np.full(V, np.inf, dtype=np.float64)
    owner_cell = np.full(V, -1, dtype=np.int64)

    # Keep only valid sources (>=0)
    valid_cells = np.flatnonzero(sources >= 0)   # indices into original cells
    if valid_cells.size == 0:
        return owner_cell, dist

    valid_source_vertices = sources[valid_cells]  # mesh vertex ids

    heap = []

    # Seed heap with all valid sources, but avoid duplicates (multiple cells on same vertex)
    # If duplicates exist, first one in valid_cells wins; you can change that policy if needed.
    for cell_idx, s in zip(valid_cells, valid_source_vertices):
        # s is guaranteed >= 0 here
        if dist[s] > 0.0:
            dist[s] = 0.0
            owner_cell[s] = cell_idx
            heapq.heappush(heap, (0.0, s))

    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        ou = owner_cell[u]
        for v, w in zip(nbrs[u], wts[u]):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                owner_cell[v] = ou
                heapq.heappush(heap, (nd, v))

    return owner_cell, dist