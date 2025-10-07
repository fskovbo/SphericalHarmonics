
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



# # ----------------------- Heat method (Crane) ------------------------------

# def build_G_face(v, f):
#     """
#     Construct per-face gradients of barycentric basis functions.

#     Parameters
#     ----------
#     v : (V,3) array
#         Vertex coordinates
#     f : (F,3) int array
#         Triangle indices

#     Returns
#     -------
#     G_face : (F,3,3) array
#         G_face[k,a,:] = gradient (3,) of basis function φ_a on face k
#     """
#     tri = v[f]  # (F,3,3)

#     e1 = tri[:, 1] - tri[:, 0]
#     e2 = tri[:, 2] - tri[:, 0]
#     n = np.cross(e1, e2)          # face normals (unnormalized)
#     dblA = np.linalg.norm(n, axis=1)  # 2 * area
#     n_unit = n / (dblA[:, None] + 1e-20)

#     F = f.shape[0]
#     G_face = np.zeros((F, 3, 3), dtype=np.float64)

#     # grad φ_a = (n × edge_opposite) / (2A)
#     for a in range(3):
#         i1, i2 = (a + 1) % 3, (a + 2) % 3
#         edge = tri[:, i2] - tri[:, i1]    # edge opposite vertex a
#         grad = np.cross(n_unit, edge) / (dblA[:, None] + 1e-20)
#         G_face[:, a, :] = grad

#     return G_face


# def heat_geodesic_all_sources(v: np.ndarray,
#                               f: np.ndarray,
#                               L: sparse.spmatrix,
#                               M: sparse.spmatrix,
#                               t: float = None,
#                               sources: np.ndarray = None):
#     """Compute approximate geodesic distances using the Heat Method (Crane et al. 2013).

#     Parameters
#     ----------
#     v : (V,3) array
#         Mesh vertices
#     f : (F,3) array
#         Mesh faces (triangles)
#     L : (V,V) sparse
#         Cotangent Laplacian
#     M : (V,V) sparse
#         Lumped mass matrix
#     t : float, optional
#         Heat time. If None, set ~ mean_edge_length^2
#     sources : array of int, optional
#         Indices of source vertices. If None, compute for all vertices (O(V^2)).

#     Returns
#     -------
#     D_out : (S,V) array
#         Geodesic distances from each source to all vertices
#     """
#     V = v.shape[0]
#     F = f.shape[0]

#     # build per-face barycentric gradients
#     G_face = build_G_face(v, f)

#     # per-face areas
#     tri = v[f]
#     face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
#     face_areas = 0.5 * np.linalg.norm(face_normals, axis=1)

#     # regularize Laplacian for stability
#     reg = 1e-12
#     L_reg = L + reg * sparse.eye(V)
#     L_factor = splu(L_reg.tocsc())

#     # heat solve system: (M + tL) u = M δ
#     if t is None:
#         edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
#         edges = np.unique(np.sort(edges, axis=1), axis=0)
#         mean_edge = np.mean(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1))
#         t = mean_edge ** 2

#     A = (M + t * L).tocsc()
#     A_factor = splu(A)

#     # define sources
#     if sources is None:
#         sources = np.arange(V, dtype=np.int32)
#     else:
#         sources = np.asarray(sources, dtype=np.int32)

#     S = len(sources)
#     D_out = np.zeros((S, V), dtype=np.float64)

#     for si, s in enumerate(tqdm(sources, desc='heat sources')):
#         # RHS: delta at source
#         delta = np.zeros(V, dtype=np.float64)
#         delta[s] = 1.0
#         rhs = M @ delta

#         # solve heat equation
#         u = A_factor.solve(rhs)

#         # grad u per face: sum_a u[v_a] * grad φ_a
#         grad_u = np.einsum('ka,kai->ki', u[f], G_face)  # (F,3)

#         # normalized vector field X
#         norms = np.linalg.norm(grad_u, axis=1)
#         norms[norms == 0] = 1e-12
#         X_face = -(grad_u.T / norms).T  # (F,3)

#         # divergence at vertices: sum over faces
#         contrib = np.einsum('ki,kai->ka', X_face, G_face)  # (F,3)
#         contrib *= face_areas[:, None]

#         div = np.zeros(V, dtype=np.float64)
#         for a in range(3):
#             np.add.at(div, f[:, a], contrib[:, a])

#         # solve Poisson: L φ = div
#         phi = L_factor.solve(div)

#         # normalize: distance zero at source
#         phi -= phi[s]

#         D_out[si, :] = phi

#     return D_out



# ------------------------- Pair binning ----------------------------------

def pair_bins_by_distance(D: np.ndarray,
                          areas: np.ndarray,
                          bin_edges: np.ndarray,
                          sources: np.ndarray | None = None,
                          use_area_weights: bool = True):
    """
    Construct distance-binned index pairs (i,j) from a precomputed distance matrix.

    Can operate in two modes:
      1. Global: if `sources` is None, pairs all vertices/cells (i<j).
      2. Source-restricted: if `sources` is provided, only uses i ∈ sources.

    Args
    ----
    D : (N,N) ndarray
        Symmetric distance matrix for vertices, cells, etc.
    areas : (N,) ndarray
        Area (or other weight) per vertex/cell.
    bin_edges : (B+1,) ndarray
        Edges defining the distance bins.
    sources : (S,) ndarray of int, optional
        Indices of source vertices/cells to restrict i.
        If None, all vertices are used.
    use_area_weights : bool
        If True, weight each pair as area[i]*area[j].
        If False, weight = 1.

    Returns
    -------
    bins_dict : dict
        For each bin b:
          - 'I_b': array of source indices
          - 'J_b': array of partner indices
          - 'W_b': array of pair weights
        Also includes 'bin_edges'.
    """
    N = D.shape[0]
    B = len(bin_edges) - 1
    r_max = bin_edges[-1]

    # Determine sources
    if sources is None:
        sources = np.arange(N)
        symmetric = True
    else:
        sources = np.asarray(sources, dtype=int)
        symmetric = False

    I_bins = [[] for _ in range(B)]
    J_bins = [[] for _ in range(B)]
    W_bins = [[] for _ in range(B)]

    for i in sources:
        if symmetric:
            ds = D[i, i+1:]
            js = np.arange(i+1, N)
        else:
            ds = D[i, :]
            js = np.arange(N)

        mask = (ds <= r_max)
        if not symmetric:
            mask &= (js != i)
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



def save_pair_bins(outdict, filename):
    np.savez_compressed(filename, **outdict)


def load_pair_bins(filename):
    data = np.load(filename)
    return dict(data)


# -------------------- Correlation accumulation ---------------------------

def compute_two_point_correlations(bins_dict: dict, X: np.ndarray):
    """
    Compute two-point auto- and cross-correlations for fields on a discretized surface,
    along with connected and normalized (Pearson-like) correlation functions.

    Parameters
    ----------
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
    Corr_raw : (B, F, F) ndarray
        Raw (uncentered) correlation matrices per bin, ⟨F_i F_j⟩.
    Corr_conn : (B, F, F) ndarray
        Connected correlations: ⟨F_i F_j⟩ − ⟨F_i⟩⟨F_j⟩.
    Corr_norm : (B, F, F) ndarray
        Connected + normalized correlations (Pearson-style):
        (⟨F_i F_j⟩ − ⟨F_i⟩⟨F_j⟩) / (σ_i σ_j).
    Ns : (B,) ndarray
        Normalization weights per distance bin.

    Notes
    -----
    - This computes correlations of the form

          C_b^{kl} = (1 / N_b) * sum_{(i,j) in bin b} W_ij * (F_i^k * F_j^l + F_j^k * F_i^l)

      where F_i^k is the value of field k at vertex i,
      W_ij = area[i] * area[j],
      and N_b = 2 * sum_{(i,j) in bin b} W_ij.

    - Connected correlations remove the global mean field contributions.
    - Normalized correlations are dimensionless and comparable across datasets.
    """

    # --- Input validation ---
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]   # (N,) → (N,1)

    bin_edges = bins_dict['bin_edges']
    B = len(bin_edges) - 1
    N, F = X.shape

    Cs = np.zeros((B, F, F), dtype=np.float64)
    Ns = np.zeros(B, dtype=np.float64)

    # --- Compute raw two-point correlations ---
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

        # symmetric correlation accumulation
        Cs[b] += A.T @ (w * Bv) + Bv.T @ (w * A)
        Ns[b] += float(np.sum(W)) * 2.0

    # --- Normalize each bin ---
    for b in range(B):
        if Ns[b] > 0:
            Cs[b] /= Ns[b]
        else:
            Cs[b, :, :] = np.nan

    # --- Compute connected and normalized correlations ---
    means = np.mean(X, axis=0)       # (F,)
    vars_ = np.var(X, axis=0)
    stds = np.sqrt(vars_ + 1e-12)

    Corr_conn = np.copy(Cs)
    Corr_norm = np.copy(Cs)

    for b in range(B):
        if Ns[b] > 0:
            Corr_conn[b] -= np.outer(means, means)
            Corr_norm[b] = Corr_conn[b] / (np.outer(stds, stds) + 1e-12)
        else:
            Corr_conn[b][:] = np.nan
            Corr_norm[b][:] = np.nan

    return Cs, Corr_conn, Corr_norm, Ns



def compute_anchored_enrichment(bins_dict: dict,
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


def compute_anchored_radial_profile(bins_out: dict, field: np.ndarray):
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
