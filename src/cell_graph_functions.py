import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pickle

from src.OrganoidMesh import OrganoidMesh


def positive_indices_and_labels(X: np.ndarray):
    """
    For each column of X, return the row indices of positive entries,
    and a binary (0/1) mask of the same shape as X.
    
    Parameters
    ----------
    X : np.ndarray
        Input array (2D or higher, but column logic applies to last axis).
    
    Returns
    -------
    indices_per_col : list of np.ndarray
        List of arrays, where indices_per_col[j] contains the row indices
        of positive entries in column j.
    labels : np.ndarray
        Binary mask of the same shape as X (1 if >0, else 0).
    """
    # binary mask (0/1)
    labels = (X > 0).astype(int)
    
    # handle only the first two axes for "rows" and "columns"
    indices_per_col = [np.where(labels[:, j])[0] for j in range(X.shape[1])]
    
    return indices_per_col, labels


def build_cell_graph_from_mesh(mesh):
    """
    Build a cell adjacency graph from an OrganoidMeshMarkers (or compatible) object.

    Args
    ----
    mesh : OrganoidMeshMarkers
        Must have attributes:
        - f : (F, 3) ndarray of triangle faces
        - cell_label_field : (V,) ndarray of contiguous cell labels

    Returns
    -------
    G : networkx.Graph
        Undirected cell adjacency graph. Nodes correspond to unique cell IDs.
    """

    G = nx.Graph()

    # Ensure contiguous integer labeling
    C = mesh.cell_label_field.max() + 1
    G.add_nodes_from(range(C))

    # Build adjacency edges based on shared faces
    for tri in mesh.f:
        labs = mesh.cell_label_field[tri]
        labs = labs[labs >= 0]
        if len(labs) <= 1:
            continue
        uniq = np.unique(labs)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                G.add_edge(int(uniq[i]), int(uniq[j]))

    return G


def build_cell_graph_from_voronoi(
    nuclei_xyz,
    markers_bin,
    mesh: OrganoidMesh,
    vertex_owner,
    proj_vertex_ids,
    proj_points,
):
    """
    Build networkx graph with adjacency derived from face neighbors.

    Parameters
    ----------
    nuclei_xyz : (N_cells, 3)
    markers_bin : (N_cells, n_markers)
    mesh : OrganoidMesh
        Mesh object containing .f for faces.
    vertex_owner : (V,)
        For each vertex, index of owning cell (0..N_cells-1).
    proj_vertex_ids : (N_cells,)
    proj_points : (N_cells, 3)
    """
    G = nx.Graph()
    N_cells = len(nuclei_xyz)

    # add nodes
    for i in range(N_cells):
        G.add_node(
            i,
            centroid=nuclei_xyz[i].tolist(),
            markers_bin=markers_bin[i].tolist(),
            proj_vertex=int(proj_vertex_ids[i]),
            proj_point=proj_points[i].tolist(),
        )

    owners = vertex_owner
    f = mesh.f  #  faces from mesh

    # add edges from faces
    for tri in f:
        a, b, c = tri
        ca, cb, cc = owners[a], owners[b], owners[c]

        if ca < 0 or cb < 0 or cc < 0:
            continue

        for (ci, cj) in [(ca, cb), (cb, cc), (cc, ca)]:
            if ci == cj:
                continue
            u = int(ci)
            v = int(cj)
            if u != v:
                G.add_edge(u, v)

    return G


def add_vertex_field_to_graph(
    G: nx.Graph,
    vertex_field: np.ndarray,
    attr_name: str,
):
    """
    Attach a mesh-defined field to each node in the cell graph.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes have a "proj_vertex" attribute giving the
        mesh vertex ID associated with that cell.
    vertex_field : (V,) or (V, K) ndarray
        Field defined on mesh vertices.
        - If 1D (V,), each node gets a scalar (float/int).
        - If 2D (V, K), each node gets a 1D np.ndarray of length K.
    attr_name : str
        Name under which to store the field for each node.

    Returns
    -------
    G : networkx.Graph
        Graph with added node attributes (modified in-place).
    """
    vertex_field = np.asarray(vertex_field)

    for n in G.nodes:
        v_id = G.nodes[n]["proj_vertex"]   # mesh vertex index
        value = vertex_field[v_id]

        if np.ndim(value) == 0:
            # scalar: store as Python float/int
            G.nodes[n][attr_name] = value.item()
        else:
            # vector: store as a 1D numpy array
            G.nodes[n][attr_name] = np.array(value, copy=True)

    return G



def save_cell_graph(path, G):
    """
    Save the organoid cell graph using Python pickle.
    Preserves all edges, attributes, metadata.
    """
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved graph to {path}")


def load_cell_graph(path):
    """
    Load a saved organoid cell graph.

    Returns
    -------
    G : networkx.Graph
    """

    with open(path, "rb") as f:
        G = pickle.load(f)

    return G



def load_cell_graph_from_npz(data: np.lib.npyio.NpzFile) -> nx.Graph:
    """
    Reconstruct a NetworkX graph from edges and node count stored in an NPZ file.
    This matches the new preprocessing format where `edges` and `n_nodes`
    are explicitly saved.
    """
    if "graph_edges" not in data.files or "graph_n_nodes" not in data.files:
        raise KeyError("NPZ file must contain 'edges' and 'n_nodes' to load the cell graph.")

    edges = data["graph_edges"]
    n_nodes = int(data["graph_n_nodes"])

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    if edges.size > 0:
        edges = edges.reshape(-1, 2)
        G.add_edges_from(edges.tolist())

    return G


def kNN_marker_composition(cell_graph, positive_labels, focal_marker_idx, k=3):
    """
    For cells positive for a given focal marker, compute the average composition
    of all markers among their k nearest neighbors.

    Args
    ----
    cell_graph : networkx.Graph
        Graph of cell adjacency (nodes = 0..N-1).
    positive_labels : (N, M) ndarray of {0,1}
        Binary marker positivity for each cell (N cells, M markers).
    focal_marker_idx : int
        Index of the focal marker (column in positive_labels).
    k : int
        Number of nearest neighbors to consider.

    Returns
    -------
    avg_comp : (M,) ndarray
        Mean fraction of neighbors positive for each marker.
    std_comp : (M,) ndarray
        Standard deviation across focal cells.
    sem_comp : (M,) ndarray
        Standard error of the mean across focal cells.
    comp_accum : (N_focal, M) ndarray
        Raw per-cell compositions (rows = focal cells, cols = markers).
    """
    N, M = positive_labels.shape
    focal_cells = np.where(positive_labels[:, focal_marker_idx] == 1)[0]
    if len(focal_cells) == 0:
        raise ValueError("No cells positive for the focal marker")

    # Precompute shortest-path distances
    sp_lengths = dict(nx.all_pairs_shortest_path_length(cell_graph))

    comp_accum = np.zeros((len(focal_cells), M), dtype=float)

    for idx, cell in enumerate(focal_cells):
        dists = sp_lengths[cell]
        neighbors_sorted = sorted(
            dists.items(),
            key=lambda x: x[1] if x[0] != cell else np.inf
        )
        k_neighbors = [n for n, d in neighbors_sorted[:k]]

        if len(k_neighbors) > 0:
            comp_accum[idx] = positive_labels[k_neighbors].mean(axis=0)

    # Average, spread, SEM
    avg_comp = comp_accum.mean(axis=0)
    std_comp = comp_accum.std(axis=0)
    sem_comp = std_comp / np.sqrt(len(focal_cells))

    return avg_comp, std_comp, sem_comp, comp_accum



def build_weight_matrix_kNN(G, k=1, row_standardize=False):
    """
    Build NxN weight matrix W from a cell adjacency graph G,
    considering up to k-nearest neighbors along the graph edges.

    Args
    ----
    G : networkx.Graph
        Undirected graph of cell adjacency (nodes can be arbitrary labels).
    k : int
        Number of nearest neighbors along graph edges to include.
        k=1 corresponds to direct neighbors.
    row_standardize : bool
        If True, divide each row by its sum so rows sum to 1.

    Returns
    -------
    W : (N, N) ndarray
        Weight matrix of size N x N.
    node_list : list
        Maps row index -> graph node.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    W = np.zeros((N, N), dtype=float)

    # For each node, find neighbors up to k steps
    for n in nodes:
        visited = set([n])
        frontier = set([n])
        for step in range(k):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(G.neighbors(node))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        visited.remove(n)  # exclude self
        i = idx[n]
        for neighbor in visited:
            j = idx[neighbor]
            W[i, j] = 1.0

    if row_standardize:
        row_sums = W.sum(axis=1)
        nz = row_sums != 0
        W[nz] = (W[nz].T / row_sums[nz]).T

    return W, nodes


def build_weight_matrix_distance(D, decay_length=1.0, row_standardize=False):
    """
    Build NxN weight matrix using exponentially decaying function of geodesic distance.

    Args
    ----
    D : (N, N) ndarray
        Symmetric distance matrix between cells.
    decay_length : float
        Characteristic distance lambda for exponential decay.
    row_standardize : bool
        Whether to row-normalize W.

    Returns
    -------
    W : (N, N) ndarray
        Weight matrix.
    """
    W = np.exp(-D / decay_length)
    np.fill_diagonal(W, 0.0)  # no self-weight

    if row_standardize:
        row_sums = W.sum(axis=1)
        nz = row_sums != 0
        W[nz] = (W[nz].T / row_sums[nz]).T

    return W


def compute_morans_I(X, W, Y=None):
    """
    Compute global and local Moran's I or cross-Moran's I.

    Args
    ----
    X : (N, M) ndarray
        First set of markers (binary or continuous).
    W : (N, N) ndarray
        Spatial weights matrix (binary adjacency or row-standardized).
    Y : (N, M) ndarray, optional
        Second set of markers for cross-Moran's I. If None, compute regular Moran's I.

    Returns
    -------
    global_I : (M,) or (M,M) ndarray
        Global Moran's I (regular) or cross-Moran's I.
    local_I : (N,M) or (N,M,M) ndarray
        Local Moran's I values for each cell.
    Z_X : (N,M) ndarray
        Mean-centered X values.
    Z_Y : (N,M) ndarray
        Mean-centered Y values (same as X if Y=None).
    m2_X : (M,) ndarray
        Variance denominators for X markers.
    """
    N, M = X.shape
    x_mean = X.mean(axis=0)
    Z_X = X - x_mean
    m2_X = np.sum(Z_X**2, axis=0) / N
    S0 = W.sum()

    if Y is None:
        # Regular Moran's I
        global_I = (N / S0) * np.sum(Z_X * (W @ Z_X), axis=0) / np.sum(Z_X**2, axis=0)
        WZ = W @ Z_X
        local_I = (Z_X / m2_X[None, :]) * WZ
        Z_Y = Z_X
    else:
        # Cross Moran's I
        y_mean = Y.mean(axis=0)
        Z_Y = Y - y_mean
        global_I = (N / S0) * (Z_X.T @ W @ Z_Y) / np.sum(Z_X**2, axis=0)[:, None]
        WZ_Y = W @ Z_Y
        local_I = (Z_X[:, :, None] / m2_X[None, :, None]) * WZ_Y[:, None, :]

    return global_I, local_I, Z_X, Z_Y, m2_X


def permutation_test_morans_I(X, W, Y=None, n_perms=999, seed=None):
    """
    Permutation test for global Moran's I or cross-Moran's I.

    Args
    ----
    X : (N, M) ndarray
        First set of markers.
    W : (N, N) ndarray
        Spatial weights matrix.
    Y : (N, M) ndarray, optional
        Second set of markers for cross-Moran's I. If None, compute regular Moran's I.
    n_perms : int
        Number of permutations.
    seed : int, optional
        Random seed.

    Returns
    -------
    global_I_obs : (M,) or (M,M) ndarray
        Observed global Moran's I.
    local_I_obs : (N,M) or (N,M,M) ndarray
        Observed local Moran's I.
    global_I_perm : (n_perms, M) or (n_perms, M, M) ndarray
        Permuted global Moran's I values.
    p_values : (M,) or (M,M) ndarray
        Empirical two-sided p-values.
    """
    rng = np.random.default_rng(seed)
    N, M = X.shape

    # Observed values
    global_I_obs, local_I_obs, Z_X, Z_Y, m2_X = compute_morans_I(X, W, Y=Y)

    # Prepare permutation array
    if Y is None:
        global_I_perm = np.zeros((n_perms, M))
        for p in range(n_perms):
            perm_idx = rng.permutation(N)
            X_perm = X[perm_idx, :]
            global_I_perm[p], _, _, _, _ = compute_morans_I(X_perm, W)
    else:
        global_I_perm = np.zeros((n_perms, M, M))
        for p in range(n_perms):
            perm_idx = rng.permutation(N)
            Y_perm = Y[perm_idx, :]
            global_I_perm[p], _, _, _, _ = compute_morans_I(X, W, Y=Y_perm)

    # Empirical two-sided p-values
    p_values = np.mean(np.abs(global_I_perm) >= np.abs(global_I_obs)[None, ...], axis=0)

    return global_I_obs, local_I_obs, global_I_perm, p_values


def local_field_statistics(field, marker_masks, G=None, k=0, areas=None):
    """
    Compute local statistics (mean, std, count) of one or more fields for multiple markers.
    Can include k-nearest-neighbor aggregation along a graph.

    Args
    ----
    field : (N,) or (N,F) ndarray
        Continuous field(s) defined per cell.
    marker_masks : (N,) or (N,M) ndarray of {0,1}
        Binary masks indicating positive cells per marker.
    G : networkx.Graph, optional
        Cell adjacency graph for kNN aggregation. Only used if k>0.
    k : int
        Number of nearest neighbors along graph edges to include.
    areas : (N,) ndarray, optional
        Weights per cell (used only if k>0 and you want weighted averaging).

    Returns
    -------
    means : (M,F) ndarray
        Mean field per marker
    stds : (M,F) ndarray
        Std per marker
    counts : (M,) ndarray
        Number of cells considered per marker (excluding neighbors)
    """

    # Ensure field is 2D: (N, F)
    field = np.atleast_2d(field)
    if field.shape[0] == 1 and field.shape[1] != 1:
        # shape (1, N) -> (N, 1)
        field = field.T

    # Ensure marker_masks is 2D: (N, M)
    marker_masks = np.atleast_2d(marker_masks)
    if marker_masks.shape[0] == 1 and marker_masks.shape[1] != 1:
        marker_masks = marker_masks.T

    N, F = field.shape
    N2, M = marker_masks.shape
    assert N == N2, "field and marker_masks must have same length"

    means = np.zeros((M, F), dtype=float)
    stds  = np.zeros((M, F), dtype=float)
    counts = np.zeros(M, dtype=int)

    if k > 0:
        if G is None:
            raise ValueError("G must be provided for kNN aggregation with k>0")
        # Build adjacency list
        idx = {n:i for i,n in enumerate(G.nodes())}
        # Precompute neighbors up to k steps for each node
        neighbors_list = []
        for n in G.nodes():
            visited = set([n])
            frontier = set([n])
            for step in range(k):
                next_frontier = set()
                for node in frontier:
                    next_frontier.update(G.neighbors(node))
                next_frontier -= visited
                visited.update(next_frontier)
                frontier = next_frontier
                if not frontier:
                    break
            visited.remove(n)  # exclude self
            neighbors_list.append([idx[nb] for nb in visited])
        # neighbors_list[i] = list of neighbor indices for cell i

        # Aggregate field over neighbors
        field_agg = np.zeros_like(field)
        for i in range(N):
            if neighbors_list[i]:
                neighbor_vals = field[neighbors_list[i]]
                if areas is not None:
                    weights = areas[neighbors_list[i]]
                    field_agg[i] = np.average(neighbor_vals, axis=0, weights=weights)
                else:
                    field_agg[i] = neighbor_vals.mean(axis=0)
            else:
                field_agg[i] = field[i]
        field = field_agg

    for m in range(M):
        mask = marker_masks[:, m] > 0
        counts[m] = mask.sum()
        if counts[m] > 0:
            means[m, :] = field[mask].mean(axis=0)
            stds[m, :]  = field[mask].std(axis=0)
        else:
            means[m, :] = np.nan
            stds[m, :]  = np.nan

    return means, stds, counts
