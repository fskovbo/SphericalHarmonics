
import numpy as np
import os
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu
from tqdm import tqdm

try:
    import igl
    HAVE_IGL = True
except Exception:
    HAVE_IGL = False


# --------------------------- Mesh helpers ---------------------------------

def vertex_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute per-vertex Voronoi-like area (one-third of incident face areas).
    Returns array of shape (V,).
    """
    V = v.shape[0]
    areas = np.zeros(V, dtype=np.float64)
    tri = v[f]  # (F,3,3)
    face_areas = 0.5 * np.linalg.norm(np.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0]), axis=1)
    for i in range(3):
        areas[f[:, i]] += face_areas / 3.0
    return areas


def build_cotangent_laplacian_and_mass(v: np.ndarray, f: np.ndarray):
    """Return sparse cotangent Laplacian L and mass matrix M (both VxV).
    L is the positive-semidefinite operator so that the weak form is u^T L u.
    We return L in scipy CSR format and M as diagonal mass matrix.
    Requires igl.
    """
    if not HAVE_IGL:
        raise ImportError("build_cotangent_laplacian_and_mass requires libigl (pyigl).")
    L = -igl.cotmatrix(v, f)  # note: igl returns sparse matrix
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    # ensure scipy sparse formats
    L = sparse.csr_matrix(L)
    M = sparse.csr_matrix(M)
    return L, M


def farthest_point_sampling(v: np.ndarray, k: int, seed: int = 0):
    """
    Farthest-point sampling (FPS) in R^3 to pick k vertex indices with near-uniform coverage.
    Works well on reasonably well-sampled surfaces.

    Args:
        v : (V,3) vertices
        k : number of samples
        seed : RNG seed

    Returns:
        idx : (k,) int array of chosen vertex indices (global indexing)
    """
    rng = np.random.default_rng(seed)
    V = v.shape[0]
    if k >= V:
        return np.arange(V, dtype=np.int32)

    # pick a random start
    start = int(rng.integers(0, V))
    idx = np.empty(k, dtype=np.int32)
    idx[0] = start

    # maintain min distance to selected set (Euclidean in R^3; cheap and effective)
    dmin = np.linalg.norm(v - v[start], axis=1)

    for i in range(1, k):
        nxt = int(np.argmax(dmin))
        idx[i] = nxt
        # update
        dnew = np.linalg.norm(v - v[nxt], axis=1)
        dmin = np.minimum(dmin, dnew)

    return idx

# ------------------------ Graph distances ---------------------------------

def build_edge_graph(v: np.ndarray, f: np.ndarray) -> sparse.csr_matrix:
    """Build symmetric adjacency (V x V) sparse matrix with Euclidean edge lengths as weights.
    Only edges present in faces are added.
    """
    V = v.shape[0]
    # collect edges
    I = []
    J = []
    W = []
    for tri in f:
        for a, b in ((0,1),(1,2),(2,0)):
            i = tri[a]; j = tri[b]
            if i == j: continue
            I.append(i); J.append(j)
            W.append(np.linalg.norm(v[i]-v[j]))
            # also add reverse
            I.append(j); J.append(i)
            W.append(np.linalg.norm(v[i]-v[j]))
    A = sparse.csr_matrix((W, (I, J)), shape=(V, V))
    # if multiple duplicate edges, keep smallest weight
    A.data[np.isnan(A.data)] = 0.0
    return A

def geodesic_dijkstra_all_sources(edge_adj: sparse.spmatrix, sources: np.ndarray = None):
    """Compute geodesic distances on mesh (edge-graph) from one or more sources.
    Uses SciPy's csgraph.dijkstra. If sources is None, compute all-pairs distances
    (be careful with memory for V>2000).

    Returns: D (S x V) if sources provided, else (V x V)
    """
    if sources is None:
        # all-pairs
        D = csgraph.dijkstra(edge_adj, directed=False)
        return D
    else:
        D = csgraph.dijkstra(edge_adj, directed=False, indices=sources)
        return D


# ------------------------ Diffusion distances -----------------------------

def diffusion_distance_matrix(eigvals: np.ndarray, eigvecs: np.ndarray, t: float):
    """Compute diffusion distance matrix at time t using spectral formula:

    d_t(i,j)^2 = sum_m exp(-2 lambda_m t) * (phi_m(i) - phi_m(j))^2

    Inputs:
      eigvals: (k,) eigenvalues (nonnegative)
      eigvecs: (V, k) eigenvectors (columns are eigenvectors)
    Returns:
      D: (V, V) symmetric distance matrix (float64)
    """
    eigvals = np.asarray(eigvals).ravel()
    Phi = np.asarray(eigvecs)
    V, k = Phi.shape
    if eigvals.shape[0] != k:
        raise ValueError("eigvals length must match eigvecs second dimension")

    w = np.exp(-2.0 * eigvals * t)  # (k,)
    sqrt_w = np.sqrt(w)
    Phi_w = Phi * sqrt_w[None, :]
    G = Phi_w @ Phi_w.T    # (V,V)
    diag = np.sum(Phi_w * Phi_w, axis=1)
    D2 = diag[:, None] + diag[None, :] - 2.0 * G
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    return D


# def diffusion_distance_rows(eigvals: np.ndarray, eigvecs: np.ndarray, t: float,
#                             sources: np.ndarray) -> np.ndarray:
#     """
#     Diffusion distances from a subset of source vertices to all vertices.

#     Returns:
#         D_rows : (S,V) distances, where S = len(sources)
#     """
#     eigvals = np.asarray(eigvals).ravel()
#     Phi = np.asarray(eigvecs)           # (V,k)
#     V, k = Phi.shape
#     if eigvals.shape[0] != k:
#         raise ValueError("eigvals length must match eigvecs second dimension")

#     w = np.exp(-2.0 * eigvals * t)      # (k,)
#     Phi_w = Phi * np.sqrt(w)[None, :]   # (V,k)

#     diag = np.sum(Phi_w * Phi_w, axis=1)  # (V,)
#     sources = np.asarray(sources, dtype=np.int32)
#     D_rows = np.empty((sources.size, V), dtype=np.float64)

#     for si, s in enumerate(sources):
#         g = Phi_w @ Phi_w[s, :]         # (V,)
#         D2 = diag + diag[s] - 2.0 * g
#         np.maximum(D2, 0.0, out=D2)
#         np.sqrt(D2, out=D2)
#         D_rows[si, :] = D2
#     return D_rows


def diffusion_distance_rows(eigvals: np.ndarray, eigvecs: np.ndarray, 
                            rows: np.ndarray, t_norm: float = 1.0) -> np.ndarray:
    """
    Compute diffusion distances from selected source vertices to all vertices,
    using the spectral definition with normalized timescale.

        d_t(i,j)^2 = sum_m exp(-2 λ_m t) * (φ_m(i) - φ_m(j))^2

    Args
    ----
    eigvals : (k,) ndarray
        Eigenvalues (≥0) of Laplace-Beltrami operator.
    eigvecs : (V,k) ndarray
        Eigenvectors (columns are eigenfunctions, normalized wrt mass matrix).
    rows : (S,) ndarray
        Indices of source vertices for which to compute distances.
    t_norm : float
        Normalized diffusion time. The actual time is set as
            t = t_norm / λ_max
        where λ_max = max(eigvals[1:]) (skipping the zero mode).

    Returns
    -------
    D : (S, V) ndarray
        Diffusion distances from each source in rows to all vertices.
    """
    eigvals = np.asarray(eigvals).ravel()
    Phi = np.asarray(eigvecs)

    V, k = Phi.shape
    if eigvals.shape[0] != k:
        raise ValueError("eigvals length must match eigvecs second dimension")

    # Skip the zero eigenfunction (first mode, constant)
    eigvals = eigvals[1:]
    Phi = Phi[:, 1:]

    # Normalized diffusion time
    lam_max = np.max(eigvals)
    t = t_norm / lam_max

    # Compute weighted eigenvectors
    w = np.exp(-2.0 * eigvals * t)   # (k-1,)
    sqrt_w = np.sqrt(w)
    Phi_w = Phi * sqrt_w[None, :]    # (V, k-1)

    # Precompute diagonal term (per vertex)
    diag = np.sum(Phi_w * Phi_w, axis=1)   # (V,)

    D_rows = np.zeros((len(rows), V), dtype=np.float64)

    # Compute distances row by row
    for si, s in enumerate(rows):
        diff = Phi_w[s, None, :] - Phi_w    # (V, k-1)
        D_rows[si] = np.sqrt(np.sum(diff * diff, axis=1))

    return D_rows


def normalize_eigenvectors_euclidean(eigvecs, M):
    """
    Convert eigenvectors from M-orthonormal basis to standard Euclidean normalization.
    
    Args:
        eigvecs : (V, k) ndarray, M-orthonormal eigenvectors
        M       : (V, V) sparse mass matrix
    Returns:
        eigvecs_normed : (V, k) ndarray, Euclidean-orthonormalized
    """
    M_diag = M.diagonal()   # Voronoi mass lumping, diagonal
    norms = np.sqrt(np.sum((eigvecs**2) * M_diag[:,None], axis=0))
    return eigvecs / norms



# ----------------------- Heat method (Crane) ------------------------------

def build_G_face(v, f):
    """
    Construct per-face gradients of barycentric basis functions.

    Parameters
    ----------
    v : (V,3) array
        Vertex coordinates
    f : (F,3) int array
        Triangle indices

    Returns
    -------
    G_face : (F,3,3) array
        G_face[k,a,:] = gradient (3,) of basis function φ_a on face k
    """
    tri = v[f]  # (F,3,3)

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    n = np.cross(e1, e2)          # face normals (unnormalized)
    dblA = np.linalg.norm(n, axis=1)  # 2 * area
    n_unit = n / (dblA[:, None] + 1e-20)

    F = f.shape[0]
    G_face = np.zeros((F, 3, 3), dtype=np.float64)

    # grad φ_a = (n × edge_opposite) / (2A)
    for a in range(3):
        i1, i2 = (a + 1) % 3, (a + 2) % 3
        edge = tri[:, i2] - tri[:, i1]    # edge opposite vertex a
        grad = np.cross(n_unit, edge) / (dblA[:, None] + 1e-20)
        G_face[:, a, :] = grad

    return G_face


def heat_geodesic_all_sources(v: np.ndarray,
                              f: np.ndarray,
                              L: sparse.spmatrix,
                              M: sparse.spmatrix,
                              t: float = None,
                              sources: np.ndarray = None):
    """Compute approximate geodesic distances using the Heat Method (Crane et al. 2013).

    Parameters
    ----------
    v : (V,3) array
        Mesh vertices
    f : (F,3) array
        Mesh faces (triangles)
    L : (V,V) sparse
        Cotangent Laplacian
    M : (V,V) sparse
        Lumped mass matrix
    t : float, optional
        Heat time. If None, set ~ mean_edge_length^2
    sources : array of int, optional
        Indices of source vertices. If None, compute for all vertices (O(V^2)).

    Returns
    -------
    D_out : (S,V) array
        Geodesic distances from each source to all vertices
    """
    V = v.shape[0]
    F = f.shape[0]

    # build per-face barycentric gradients
    G_face = build_G_face(v, f)

    # per-face areas
    tri = v[f]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    face_areas = 0.5 * np.linalg.norm(face_normals, axis=1)

    # regularize Laplacian for stability
    reg = 1e-12
    L_reg = L + reg * sparse.eye(V)
    L_factor = splu(L_reg.tocsc())

    # heat solve system: (M + tL) u = M δ
    if t is None:
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        mean_edge = np.mean(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1))
        t = mean_edge ** 2

    A = (M + t * L).tocsc()
    A_factor = splu(A)

    # define sources
    if sources is None:
        sources = np.arange(V, dtype=np.int32)
    else:
        sources = np.asarray(sources, dtype=np.int32)

    S = len(sources)
    D_out = np.zeros((S, V), dtype=np.float64)

    for si, s in enumerate(tqdm(sources, desc='heat sources')):
        # RHS: delta at source
        delta = np.zeros(V, dtype=np.float64)
        delta[s] = 1.0
        rhs = M @ delta

        # solve heat equation
        u = A_factor.solve(rhs)

        # grad u per face: sum_a u[v_a] * grad φ_a
        grad_u = np.einsum('ka,kai->ki', u[f], G_face)  # (F,3)

        # normalized vector field X
        norms = np.linalg.norm(grad_u, axis=1)
        norms[norms == 0] = 1e-12
        X_face = -(grad_u.T / norms).T  # (F,3)

        # divergence at vertices: sum over faces
        contrib = np.einsum('ki,kai->ka', X_face, G_face)  # (F,3)
        contrib *= face_areas[:, None]

        div = np.zeros(V, dtype=np.float64)
        for a in range(3):
            np.add.at(div, f[:, a], contrib[:, a])

        # solve Poisson: L φ = div
        phi = L_factor.solve(div)

        # normalize: distance zero at source
        phi -= phi[s]

        D_out[si, :] = phi

    return D_out



# ------------------------- Pair binning ----------------------------------

def pair_bins_from_distances(D: np.ndarray,
                             areas: np.ndarray,
                             bin_edges: np.ndarray,
                             use_area_weights: bool = True):
    """
    Construct pair bins (i,j) from a precomputed distance matrix D.

    Args
    ----
    D : (N,N) ndarray
        Symmetric distance matrix for the chosen discretization (vertices, cells, etc).
    areas : (N,) ndarray
        Weights per element (vertex areas, cell patch areas, etc).
    bin_edges : (B+1,) ndarray
        Distance bin edges.
    use_area_weights : bool
        If True, pair weights = area[i]*area[j].
        If False, weights = 1 (lattice approximation).

    Returns
    -------
    bins_dict : dict
        For each bin b:
          - I_b : indices of first elements in pairs
          - J_b : indices of second elements in pairs
          - W_b : weights for each pair
        Plus 'bin_edges'.
    """
    N = D.shape[0]
    B = len(bin_edges) - 1
    r_max = bin_edges[-1]

    I_bins = [[] for _ in range(B)]
    J_bins = [[] for _ in range(B)]
    W_bins = [[] for _ in range(B)]

    for i in range(N):
        ds = D[i, i+1:]
        js = np.arange(i+1, N)

        mask = ds <= r_max
        if not np.any(mask):
            continue

        ds2 = ds[mask]
        js = js[mask]
        bins = np.searchsorted(bin_edges, ds2, side='right') - 1
        valid = bins >= 0
        bins = bins[valid]
        js = js[valid]

        if use_area_weights:
            w = areas[i] * areas[js]
        else:
            w = np.ones_like(js, dtype=np.float32)

        for b, j, wb in zip(bins, js, w):
            I_bins[b].append(i)
            J_bins[b].append(int(j))
            W_bins[b].append(float(wb))

    out = {'bin_edges': np.asarray(bin_edges)}
    for b in range(B):
        out[f'I_{b}'] = np.asarray(I_bins[b], dtype=np.int32)
        out[f'J_{b}'] = np.asarray(J_bins[b], dtype=np.int32)
        out[f'W_{b}'] = np.asarray(W_bins[b], dtype=np.float32)
    return out


def save_pair_bins(outdict, filename):
    np.savez_compressed(filename, **outdict)


def load_pair_bins(filename):
    data = np.load(filename)
    return dict(data)


# -------------------- Correlation accumulation ---------------------------

def accumulate_correlations_from_bins(bins_dict, X: np.ndarray):
    """
    Compute two-point auto- and cross-correlations for fields on a discretized surface.

    Args
    ----
    bins_dict : dict
        Output of pair_bins_from_distances().
        Contains for each distance bin b:
          - I_b, J_b : indices of pairs
          - W_b : weights for each pair
        Plus 'bin_edges'.
    X : (N, F) ndarray
        Field values at each discretization element (vertex, cell, etc).
        - N = number of elements
        - F = number of fields

    Returns
    -------
    Cnorm : (B, F, F) ndarray
        Correlation matrices per bin.
    Ns : (B,) ndarray
        Normalization weights per bin.


    Notes
    -----
    - This computes correlations of the form

          C_b^{kl} = (1 / N_b) * sum_{(i,j) in bin b} W_ij * (F_i^k * F_j^l + F_j^k * F_i^l)

      where F_i^k is the value of field k at vertex i,
      W_ij = area[i] * area[j] (ensures unbiased integration over surface),
      and N_b = 2 * sum_{(i,j) in bin b} W_ij.

    - The factor of 2 comes from symmetrizing: (i,j) and (j,i) should contribute equally,
      but since only i < j pairs are stored, we explicitly add the symmetric term.

    - For F = 1, this reduces to the standard weighted auto-correlation function:

          g^{(2)}(r_b) = sum_{(i,j) in bin} W_ij * F_i * F_j  /  sum_{(i,j) in bin} W_ij

    - For F > 1, the result is a matrix of all auto- and cross-correlations.
    """

    # Ensure X is 2D
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]   # (N,) -> (N,1)

    bin_edges = bins_dict['bin_edges']
    B = len(bin_edges) - 1
    N, F = X.shape

    Cs = np.zeros((B, F, F), dtype=np.float64)
    Ns = np.zeros(B, dtype=np.float64)

    for b in range(B):
        I = bins_dict.get(f'I_{b}', np.zeros(0, dtype=np.int32))
        J = bins_dict.get(f'J_{b}', np.zeros(0, dtype=np.int32))
        W = bins_dict.get(f'W_{b}', np.zeros(0, dtype=np.float32))
        if I.size == 0:
            Cs[b, :, :] = np.nan
            continue

        A = X[I, :]
        Bv = X[J, :]
        w = W[:, None]

        Cs[b] += A.T @ (w * Bv) + Bv.T @ (w * A)
        Ns[b] += float(np.sum(W)) * 2.0

    for b in range(B):
        if Ns[b] > 0:
            Cs[b] /= Ns[b]
        else:
            Cs[b, :, :] = np.nan

    return Cs, Ns


def binary_seed_enrichment(bins_dict: dict,
                           binary_labels: np.ndarray):
    """
    Compute seed-anchored enrichment of binary-labeled cells.

    Args
    ----
    bins_dict : dict
        Output from pair_bins_from_sources. Must contain:
        - I_b, J_b, W_b for each bin
        - 'bin_edges'
    binary_labels : (N,) or (N,M) array of int or bool
        0/1 array indicating whether each cell expresses the marker(s).
        If 2D, each column is treated as a separate marker.

    Returns
    -------
    enrichment : (B,M) array
        Enrichment values E(r) = f_local(r) / f_global for each distance bin.
    f_local : (B,M) array
        Weighted local fraction of positives per bin.
    f_global : (M,) array
        Weighted global fraction of positives across all cells.
    bin_centers : (B,) array
        Midpoints of the distance bins.
    Ns : (B,) array
        Effective weighted counts (sum of weights) per bin.
    """
    bin_edges = bins_dict['bin_edges']
    B = len(bin_edges) - 1

    # Ensure labels are 2D: (N, M)
    if binary_labels.ndim == 1:
        binary_labels = binary_labels[:, None]
    N, M = binary_labels.shape

    f_local = np.zeros((B, M), dtype=np.float64)
    Ns = np.zeros(B, dtype=np.float64)

    # Compute local fractions in each bin
    for b in range(B):
        J_b = bins_dict[f'J_{b}']
        W_b = bins_dict[f'W_{b}']
        if len(J_b) == 0:
            continue

        lbls = binary_labels[J_b, :]   # (len(J_b), M)
        w = W_b[:, None]               # (len(J_b), 1)

        Ns[b] = np.sum(W_b)
        if Ns[b] > 0:
            f_local[b, :] = np.sum(w * lbls, axis=0) / Ns[b]

    # Global fractions, weighted by areas (since areas are in weights)
    total_w = np.zeros((N,), dtype=np.float64)
    for b in range(B):
        J_b = bins_dict[f'J_{b}']
        W_b = bins_dict[f'W_{b}']
        for j, w in zip(J_b, W_b):
            total_w[j] += w

    denom = np.sum(total_w)
    if denom > 0:
        f_global = (total_w[:, None] * binary_labels).sum(axis=0) / denom
    else:
        f_global = np.full(M, np.nan)

    # Enrichment
    enrichment = np.divide(f_local, f_global[None, :],
                           out=np.full_like(f_local, np.nan),
                           where=f_global[None, :] > 0)

    # Bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return enrichment, f_local, f_global, bin_centers, Ns


# ------------- Coarse graining methods -----------------------

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


def per_cell_stats_from_vertex_labels(v, f, vertex_areas, vertex_cell_labels, vertex_fields):
    """
    Compute per-cell statistics (centroid, area, averaged fields) 
    from vertex-level data on a mesh.

    This function aggregates vertex-level quantities (positions, areas, fields)
    into cell-level statistics, where cells are defined by contiguous vertex labels.

    Parameters
    ----------
    v : (V,3) ndarray
        Vertex positions of the mesh.
    f : (V,3) ndarray
        Mesh faces (indices into v). Currently not used in computation,
        but kept for consistency with mesh-related workflows.
    vertex_areas : (V,) ndarray
        Area associated with each vertex (e.g. Voronoi area).
    vertex_cell_labels : (V,) ndarray of int
        Cell label per vertex. Labels may be non-contiguous or contain -1
        for vertices not assigned to any cell.
    vertex_fields : (V,) ndarray or (V,F) ndarray
        Scalar or multi-dimensional field values defined per vertex.

    Returns
    -------
    centroids : (C,3) ndarray
        Area-weighted centroid of each cell surface in 3D space.
    cell_area_v : (C,) ndarray
        Total area associated with each cell (sum of vertex areas).
    fields : (C,F) ndarray
        Area-weighted average of field values per cell.
    valid : (C,) boolean ndarray
        Mask indicating which cells had nonzero area (True = valid).
    labels : (V,) ndarray
        Contiguous cell labels per vertex, remapped from input labels.
        Useful for indexing and correspondence.
    """

    # remap labels first
    labels, mapping = remap_labels_to_contiguous(vertex_cell_labels)
    V = v.shape[0]
    vf = np.asarray(vertex_fields)
    if vf.ndim == 1:
        vf = vf[:, None]
    C = int(labels.max() + 1) if (labels >= 0).any() else 0

    weighted_pos = np.zeros((C, 3), dtype=float)
    weighted_field = np.zeros((C, vf.shape[1]), dtype=float)
    cell_area_v = np.zeros(C, dtype=float)
    counts = np.zeros(C, dtype=int)

    for idx in range(V):
        lab = int(labels[idx])
        if lab < 0:
            continue
        a = float(vertex_areas[idx])
        weighted_pos[lab] += a * v[idx]
        weighted_field[lab] += a * vf[idx]
        cell_area_v[lab] += a
        counts[lab] += 1

    valid = cell_area_v > 0
    centroids = np.zeros((C, 3), dtype=float)
    fields = np.zeros((C, vf.shape[1]), dtype=float)
    centroids[valid] = weighted_pos[valid] / cell_area_v[valid, None]
    fields[valid] = weighted_field[valid] / cell_area_v[valid, None]

    return centroids, cell_area_v, fields, valid, labels


def get_patch_centers(vertices, labels):
    """
    For each patch (unique label), compute the 'center' vertex index:
      - compute centroid of all vertices in patch
      - pick vertex in patch closest to this centroid
    
    Args:
        vertices: (V,3) ndarray of vertex coordinates
        labels: (V,) ndarray of integer patch labels
    
    Returns:
        unique_labels: (P,) ndarray of sorted unique labels
        centers_idx: (P,) ndarray of vertex indices (into vertices)
    """
    unique_labels = np.unique(labels)
    centers_idx = np.empty(len(unique_labels), dtype=int)

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        patch_vertices = vertices[mask]

        # centroid of patch vertices
        centroid = patch_vertices.mean(axis=0)

        # indices of patch vertices in global array
        patch_indices = np.nonzero(mask)[0]

        # find vertex closest to centroid
        diffs = vertices[patch_indices] - centroid
        dists = np.einsum("ij,ij->i", diffs, diffs)  # squared distances
        best_idx = patch_indices[np.argmin(dists)]

        centers_idx[i] = best_idx

    return unique_labels, centers_idx



def mesh_volume_from_triangles(v: np.ndarray, f: np.ndarray):
    """
    Compute mesh volume by summing signed tetrahedron volumes from a reference point.

    Args
    ----
    v : (V,3) array of vertex coordinates
    f : (F,3) array of triangle indices (integers)
    
    Returns
    -------
    volume : scalar (float) -- absolute volume of the mesh
    """
    
    # triangle vertex coordinates 
    tri = v[f]      # shape (F, 3, 3)
    a = tri[:, 0, :]   # (F,3)
    b = tri[:, 1, :]
    c = tri[:, 2, :]

    # vectorized cross and dot
    cross_ab = np.cross(a, b)          # (F,3)
    tet_signed = np.einsum('ij,ij->i', cross_ab, c) / 6.0   # (F,)

    volume = np.abs(np.sum(tet_signed))

    return volume


def pair_bins_from_sources(D: np.ndarray,
                           areas: np.ndarray,
                           bin_edges: np.ndarray,
                           sources: np.ndarray,
                           use_area_weights: bool = True):
    """
    Construct pair bins (i,j) where i is restricted to a source set.

    Args
    ----
    D : (N,N) ndarray
        Symmetric distance matrix (vertex/cell geodesic distances).
    areas : (N,) ndarray
        Weights per element (vertex areas, cell patch areas, etc).
    bin_edges : (B+1,) ndarray
        Distance bin edges.
    sources : (S,) ndarray of int
        Indices of source elements (e.g. serotonin+ cells).
    use_area_weights : bool
        If True, pair weights = area[i]*area[j].
        If False, weights = 1.

    Returns
    -------
    bins_dict : dict
        For each bin b:
          - I_b : indices of sources in pairs
          - J_b : indices of partner elements
          - W_b : weights for each pair
        Plus 'bin_edges'.
    """
    N = D.shape[0]
    B = len(bin_edges) - 1
    r_max = bin_edges[-1]

    I_bins = [[] for _ in range(B)]
    J_bins = [[] for _ in range(B)]
    W_bins = [[] for _ in range(B)]

    for i in sources:
        ds = D[i, :]
        js = np.arange(N)

        mask = (ds <= r_max) & (js != i)  # exclude self
        if not np.any(mask):
            continue

        ds2 = ds[mask]
        js = js[mask]
        bins = np.searchsorted(bin_edges, ds2, side='right') - 1
        valid = (bins >= 0) & (bins < B)
        bins = bins[valid]
        js = js[valid]

        if use_area_weights:
            w = areas[i] * areas[js]
        else:
            w = np.ones_like(js, dtype=np.float32)

        for b, j, wb in zip(bins, js, w):
            I_bins[b].append(i)
            J_bins[b].append(int(j))
            W_bins[b].append(float(wb))

    out = {'bin_edges': np.asarray(bin_edges)}
    for b in range(B):
        out[f'I_{b}'] = np.asarray(I_bins[b], dtype=np.int32)
        out[f'J_{b}'] = np.asarray(J_bins[b], dtype=np.int32)
        out[f'W_{b}'] = np.asarray(W_bins[b], dtype=np.float32)
    return out


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


def anchored_radial_profile(bins_out: dict, field: np.ndarray):
    """
    Compute anchored radial profile of a continuous field
    (e.g., curvature) around sources.

    Parameters
    ----------
    bins_out : dict
        Output from pair_bins_from_sources().
    field : (N,) ndarray
        Continuous field values (per vertex or per cell).

    Returns
    -------
    bin_centers : (B,) ndarray
        Midpoints of distance bins.
    mean_profile : (B,) ndarray
        Weighted average of field[j] per bin.
    stderr_profile : (B,) ndarray
        Weighted standard error per bin.
    """
    bin_edges = bins_out['bin_edges']
    B = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean_profile = np.full(B, np.nan)
    stderr_profile = np.full(B, np.nan)

    for b in range(B):
        J_b = bins_out[f'J_{b}']
        W_b = bins_out[f'W_{b}']
        if len(J_b) == 0:
            continue
        values = field[J_b]
        weights = W_b
        mean_val = np.average(values, weights=weights)
        # weighted standard error
        var_val = np.average((values - mean_val)**2, weights=weights)
        stderr = np.sqrt(var_val / len(values))

        mean_profile[b] = mean_val
        stderr_profile[b] = stderr

    return bin_centers, mean_profile, stderr_profile


def connected_and_normalized_correlations(Corr: np.ndarray, N: np.ndarray, X: np.ndarray):
    """
    Compute connected and normalized correlation functions.

    Args
    ----
    Corr : ndarray, shape (B, F, F)
        Raw correlation matrices per distance bin (output of accumulate_correlations_from_bins).
    N : ndarray, shape (B,)
        Normalization weights per distance bin (from accumulate_correlations_from_bins).
    X : ndarray, shape (V, F)
        Field values on vertices/cell centers (used to compute means and stds).

    Returns
    -------
    Corr_conn : ndarray, shape (B, F, F)
        Connected correlations: <F_i F_j> - <F_i><F_j>.
    Corr_norm : ndarray, shape (B, F, F)
        Connected + normalized correlations:
            ( <F_i F_j> - <F_i><F_j> ) / (\sigma_i \sigma_j).
    """
    F = Corr.shape[1]

    # --- global statistics for normalization ---
    means = np.mean(X, axis=0)       # (F,)
    vars_ = np.var(X, axis=0)        # (F,)
    stds = np.sqrt(vars_ + 1e-12)    # avoid div-by-zero

    # --- connected correlations ---
    Corr_conn = np.copy(Corr)
    for b in range(Corr.shape[0]):
        if N[b] > 0:
            Corr_conn[b] -= np.outer(means, means)
        else:
            Corr_conn[b] = np.full((F, F), np.nan)

    # --- normalized connected correlations ---
    Corr_norm = np.copy(Corr_conn)
    for b in range(Corr.shape[0]):
        if N[b] > 0:
            Corr_norm[b] /= (np.outer(means, means) + 1e-12)
        else:
            Corr_norm[b] = np.full((F, F), np.nan)

    return Corr_conn, Corr_norm


import numpy as np
import cupy as cp
from scipy.sparse.linalg import splu
from scipy import sparse

def heat_geodesic_all_sources_gpu(v, f, L, M, t=None, sources=None):
    """
    Fast GPU-accelerated Heat Method using CPU sparse solves and batched GPU dense computations.

    Parameters
    ----------
    v : (V,3) array
        Mesh vertices
    f : (F,3) array
        Mesh faces (triangles)
    L : (V,V) sparse
        Cotangent Laplacian
    M : (V,V) sparse
        Lumped mass matrix
    t : float, optional
        Heat time. If None, set ~ mean_edge_length^2
    sources : array of int, optional
        Indices of source vertices. If None, compute for all vertices

    Returns
    -------
    D_out : (S,V) array
        Geodesic distances from each source to all vertices
    """
    V = v.shape[0]
    F = f.shape[0]

    if sources is None:
        sources = np.arange(V, dtype=np.int32)
    else:
        sources = np.asarray(sources, dtype=np.int32)
    S = len(sources)

    # --- CPU sparse LU decompositions ---
    reg = 1e-12
    L_factor = splu((L + reg * sparse.eye(V, dtype=L.dtype)).tocsc())

    if t is None:
        edges = np.vstack([f[:, [0,1]], f[:, [1,2]], f[:, [2,0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        mean_edge = np.mean(np.linalg.norm(v[edges[:,0]] - v[edges[:,1]], axis=1))
        t = mean_edge**2

    A_factor = splu((M + t*L).tocsc())

    # --- Move dense arrays to GPU ---
    G_face = build_G_face(v, f)           # (F,3,3)
    G_face_gpu = cp.asarray(G_face)

    tri = v[f]                             # (F,3,3)
    face_normals = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    face_areas = 0.5 * np.linalg.norm(face_normals, axis=1)
    face_areas_gpu = cp.asarray(face_areas)

    # --- Build RHS for all sources ---
    delta = np.zeros((V, S), dtype=np.float64)
    delta[sources, np.arange(S)] = 1.0
    rhs = M @ delta  # (V,S)

    # --- Solve heat equation per source (CPU) ---
    U = np.zeros_like(rhs)
    for i in range(S):
        U[:, i] = A_factor.solve(rhs[:, i])

    # --- Move batch to GPU ---
    U_gpu = cp.asarray(U.T)  # (S,V)

    # --- Compute grad_u per face for all sources at once ---
    U_f = U_gpu[:, f]  # (S,F,3)
    grad_u = cp.einsum('sfa,fai->sfi', U_f, G_face_gpu)  # (S,F,3)

    # --- Normalize vector field ---
    norms = cp.linalg.norm(grad_u, axis=2)
    norms = cp.where(norms==0, 1e-12, norms)
    X_face = -(grad_u / norms[:,:,None])

    # --- Divergence accumulation ---
    contrib = cp.einsum('sfi,fai->sfa', X_face, G_face_gpu)
    contrib *= face_areas_gpu[None,:,None]

    # Accumulate divergence to vertices on CPU
    D_out = np.zeros((S, V), dtype=np.float64)
    contrib_cpu = cp.asnumpy(contrib)

    for s_idx in range(S):
        div = np.zeros(V, dtype=np.float64)
        for a in range(3):
            np.add.at(div, f[:, a], contrib_cpu[s_idx, :, a])
        # --- Solve Poisson (CPU) ---
        phi = L_factor.solve(div)
        phi -= phi[sources[s_idx]]
        D_out[s_idx, :] = phi

    return D_out