import numpy as np
import networkx as nx
import plotly.graph_objects as go


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


def remap_labels_to_contiguous(labels):
    """
    labels: (V,) int array, may contain negative -1 for unlabeled vertices.
    returns:
      labels_contig: (V,) ints in -1 or [0..C-1]
      mapping: dict old_label -> new_index
    """
    labs = np.asarray(labels)
    uniq = np.unique(labs[labs >= 0])
    mapping = {int(old): new for new, old in enumerate(uniq)}
    labels_contig = np.full_like(labs, -1)
    for old, new in mapping.items():
        labels_contig[labs == old] = new
    return labels_contig, mapping


def build_cell_graph(faces, vertex_cell_labels):
    """
    Build a cell adjacency graph from mesh faces and contiguous cell labels.

    Args
    ----
    faces : (F,3) ndarray of int
        Triangle mesh faces (indices into vertex array).
    vertex_cell_labels : (V,) ndarray of int
        Contiguous cell labels per vertex, in [0..C-1] or -1 for unlabeled.

    Returns
    -------
    G : networkx.Graph
        Undirected cell adjacency graph. Nodes are 0..C-1 (contiguous).
        Edge (i,j) exists if any face has vertices from both cells i and j.
    """
    G = nx.Graph()

    # only use vertices with valid cell labels
    labels = np.asarray(vertex_cell_labels, dtype=int)  # force int
    labels, _ = remap_labels_to_contiguous(labels)
    C = labels.max() + 1
    G.add_nodes_from(range(C))  # make sure we have 0..C-1 nodes

    for tri in faces:
        labs = labels[tri]
        labs = labs[labs >= 0]  # ignore unlabeled verts
        if len(labs) <= 1:
            continue
        uniq = np.unique(labs)
        if len(uniq) > 1:
            # add edges between all distinct cell labels in this face
            for i in range(len(uniq)):
                for j in range(i+1, len(uniq)):
                    G.add_edge(int(uniq[i]), int(uniq[j]))

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





def plot_cell_graph_3D(centroids, graph, node_color=None, node_size=5, edge_width=1):
    """
    Visualize a cell adjacency graph in 3D using centroids.

    Args
    ----
    centroids : (N,3) ndarray
        Coordinates of each cell (cell centers).
    graph : networkx.Graph
        Graph of cell adjacency (nodes = cell indices 0..N-1)
    node_color : (N,) array, optional
        Color per node (e.g., marker value or type)
    node_size : int, optional
        Marker size for nodes
    edge_width : int, optional
        Line width for edges
    """
    # Node coordinates
    x, y, z = centroids[:,0], centroids[:,1], centroids[:,2]

    if node_color is None:
        node_color = ['blue'] * len(centroids)
    else:
        # if node_color is numeric 0/1, map to colors
        if np.issubdtype(np.array(node_color).dtype, np.number):
            node_color = ['red' if val > 0 else 'blue' for val in node_color]

    # Node trace
    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=node_size, color=node_color, line=dict(width=0)),
        hoverinfo='text'
    )

    # Create edge traces
    edge_x, edge_y, edge_z = [], [], []
    for i, j in graph.edges():
        edge_x += [centroids[i,0], centroids[j,0], None]
        edge_y += [centroids[i,1], centroids[j,1], None]
        edge_z += [centroids[i,2], centroids[j,2], None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=edge_width, color='gray'),
        hoverinfo='none'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'
    ))
    fig.show()


def build_weight_matrix_from_graph(G, row_standardize=False):
    """
    Build NxN weight matrix W from networkx Graph G.
    Nodes in G may be arbitrary, so we create an index mapping.
    Returns (W, node_list) where node_list maps row index -> graph node.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    W = np.zeros((N, N), dtype=float)
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        W[i, j] = 1.0
        W[j, i] = 1.0  # undirected
    if row_standardize:
        row_sums = W.sum(axis=1)
        # avoid divide-by-zero
        nz = row_sums != 0
        W[nz] = (W[nz].T / row_sums[nz]).T
    return W, nodes


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
        k=1 corresponds to direct neighbors (same as build_weight_matrix_from_graph()).
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


def compute_morans_I(X, W):
    """
    Compute global and local Moran's I for all markers.

    Args
    ----
    X : (N, M) ndarray
        Data matrix: N cells x M markers (binary or continuous).
    W : (N, N) ndarray
        Spatial weights matrix (binary adjacency or row-standardized).

    Returns
    -------
    global_I : (M,) ndarray
        Global Moran's I for each marker.
    local_I : (N, M) ndarray
        Local Moran's I values for each cell and marker.
    Z : (N, M) ndarray
        Mean-centered values (x - mean).
    m2 : (M,) ndarray
        Variance denominators for each marker.
    """
    N, M = X.shape
    x_mean = X.mean(axis=0)               # (M,)
    Z = X - x_mean                        # (N, M)
    denom = np.sum(Z**2, axis=0)          # (M,)
    S0 = W.sum()

    # Global Moran’s I
    num = np.sum(Z * (W @ Z), axis=0)     # (M,)
    global_I = (N / S0) * (num / denom)

    # Local Moran’s I
    m2 = denom / N
    WZ = W @ Z                            # (N, M)
    local_I = (Z / m2) * WZ               # (N, M)

    return global_I, local_I, Z, m2


def permutation_test_morans_I(X, W, n_perms=999, seed=None):
    """
    Permutation test for global Moran's I on all markers.

    Args
    ----
    X : (N, M) ndarray
        Data matrix: N cells x M markers (binary or continuous).
    W : (N, N) ndarray
        Spatial weights matrix (binary adjacency or row-standardized).
    n_perms : int
        Number of permutations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    global_I_obs : (M,) ndarray
        Observed global Moran's I values for each marker.
    local_I_obs : (N, M) ndarray
        Observed local Moran's I values for each marker and cell.
    global_I_perm : (n_perms, M) ndarray
        Permuted global Moran's I values.
    p_values : (M,) ndarray
        Empirical p-values for global Moran's I.
    """
    rng = np.random.default_rng(seed)
    N, M = X.shape

    # observed values
    global_I_obs, local_I_obs, Z, m2 = compute_morans_I(X, W)

    # permuted values
    global_I_perm = np.zeros((n_perms, M))
    for p in range(n_perms):
        perm_idx = rng.permutation(N)
        X_perm = X[perm_idx, :]
        global_I_p, _, _, _ = compute_morans_I(X_perm, W)
        global_I_perm[p] = global_I_p

    # empirical p-values (two-sided)
    p_values = np.mean(np.abs(global_I_perm) >= np.abs(global_I_obs), axis=0)

    return global_I_obs, local_I_obs, global_I_perm, p_values



def compute_cross_morans_I(X, Y, W):
    """
    Compute global and local cross-Moran's I for all marker pairs.

    Args
    ----
    X : (N, M) ndarray
        First set of markers (e.g., binary 0/1), shape: N cells x M markers.
    Y : (N, M) ndarray
        Second set of markers, same shape as X.
    W : (N, N) ndarray
        Spatial weights matrix (binary adjacency or row-standardized).

    Returns
    -------
    global_I : (M, M) ndarray
        Global cross-Moran's I between each pair of markers X[:,i] and Y[:,j].
    local_I : (N, M, M) ndarray
        Local cross-Moran's I for each cell i and marker pair (X[:,i], Y[:,j]).
    Z_X : (N, M) ndarray
        Mean-centered X values.
    Z_Y : (N, M) ndarray
        Mean-centered Y values.
    m2_X : (M,) ndarray
        Variance denominators for X markers.
    """
    N, M = X.shape
    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    Z_X = X - x_mean
    Z_Y = Y - y_mean

    m2_X = np.sum(Z_X**2, axis=0) / N  # variance per marker

    S0 = W.sum()

    # Global cross-Moran's I
    # Result shape (M, M) for all X_i vs Y_j pairs
    num = Z_X.T @ W @ Z_Y  # (M, M)
    global_I = (N / S0) * (num / np.sum(Z_X**2, axis=0)[:, None])

    # Local cross-Moran's I
    # WZ_Y: weighted sum of neighbors' Y values
    WZ_Y = W @ Z_Y  # (N, M)
    # Broadcast: (N, M, M) = (N, M_X, M_Y)
    local_I = (Z_X[:, :, None] / m2_X[None, :, None]) * WZ_Y[:, None, :]  

    return global_I, local_I, Z_X, Z_Y, m2_X


def permutation_test_cross_morans_I(X, Y, W, n_perms=999, seed=None):
    """
    Permutation test for global cross-Moran's I on all marker pairs.

    Args
    ----
    X : (N, M) ndarray
        First set of markers.
    Y : (N, M) ndarray
        Second set of markers.
    W : (N, N) ndarray
        Spatial weights matrix.
    n_perms : int
        Number of permutations.
    seed : int, optional
        Random seed.

    Returns
    -------
    global_I_obs : (M, M) ndarray
        Observed global cross-Moran's I for each marker pair.
    local_I_obs : (N, M, M) ndarray
        Observed local cross-Moran's I.
    global_I_perm : (n_perms, M, M) ndarray
        Permuted global I values.
    p_values : (M, M) ndarray
        Empirical p-values for global cross-Moran's I.
    """
    rng = np.random.default_rng(seed)
    N, M = X.shape

    # Observed values
    global_I_obs, local_I_obs, _, _, _ = compute_cross_morans_I(X, Y, W)

    # Permuted values
    global_I_perm = np.zeros((n_perms, M, M))
    for p in range(n_perms):
        perm_idx = rng.permutation(N)
        Y_perm = Y[perm_idx, :]  # shuffle Y values across cells
        global_I_p, _, _, _, _ = compute_cross_morans_I(X, Y_perm, W)
        global_I_perm[p] = global_I_p

    # Empirical two-sided p-values
    p_values = np.mean(np.abs(global_I_perm) >= np.abs(global_I_obs)[None, :, :], axis=0)

    return global_I_obs, local_I_obs, global_I_perm, p_values
