
import numpy as np
import os
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu
from tqdm import tqdm


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
        Output of pair_bins_by_distance().
        Contains for each distance bin b:
          - 'I_b': array of source indices
          - 'J_b': array of partner indices
          - 'W_b': array of pair weights
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
        Connected + normalized correlations (Pearson-style).
    Ns : (B,) ndarray
        Normalization weights per distance bin.

    Notes
    -----
    - Global means and variances are computed using the **same weights**
      as the pair binning: each vertex contributes proportionally to the
      sum of its pair weights across all bins.
    """

    # Ensure X is 2D
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]   # (N,) → (N,1)
    N, F = X.shape

    bin_edges = bins_dict['bin_edges']
    B = len(bin_edges) - 1

    # --- Precompute raw correlations ---
    Corr_raw = np.zeros((B, F, F), dtype=np.float64)
    Ns = np.zeros(B, dtype=np.float64)

    # Also accumulate weights per vertex for global mean calculation
    vertex_weight_sum = np.zeros(N, dtype=np.float64)

    for b in range(B):
        I = bins_dict.get(f'I_{b}', np.zeros(0, dtype=np.int32))
        J = bins_dict.get(f'J_{b}', np.zeros(0, dtype=np.int32))
        W = bins_dict.get(f'W_{b}', np.zeros(0, dtype=np.float32))

        if I.size == 0:
            Corr_raw[b, :, :] = np.nan
            continue

        A = X[I, :]
        Bv = X[J, :]
        w = W[:, None]

        # symmetric accumulation
        Corr_raw[b] += A.T @ (w * Bv) + Bv.T @ (w * A)
        Ns[b] += float(np.sum(W)) * 2.0

        # accumulate vertex weights for global statistics
        np.add.at(vertex_weight_sum, I, W)
        np.add.at(vertex_weight_sum, J, W)

    for b in range(B):
        if Ns[b] > 0:
            Corr_raw[b] /= Ns[b]
        else:
            Corr_raw[b][:] = np.nan

    # --- Compute global weighted means and variances ---
    total_weight = np.sum(vertex_weight_sum)
    if total_weight == 0:
        raise ValueError("Total weight is zero — check bins_dict or fields.")

    means = np.average(X, axis=0, weights=vertex_weight_sum)
    vars_ = np.average((X - means)**2, axis=0, weights=vertex_weight_sum)
    stds = np.sqrt(vars_ + 1e-12)

    # --- Connected and normalized correlations ---
    Corr_conn = np.copy(Corr_raw)
    Corr_norm = np.copy(Corr_raw)

    outer_means = np.outer(means, means)
    outer_stds = np.outer(stds, stds) + 1e-12

    for b in range(B):
        if Ns[b] > 0:
            Corr_conn[b] -= outer_means
            Corr_norm[b] = Corr_conn[b] / outer_stds
        else:
            Corr_conn[b][:] = np.nan
            Corr_norm[b][:] = np.nan

    return Corr_raw, Corr_conn, Corr_norm, Ns



def compute_anchored_enrichment(bins_dict: dict, binary_labels: np.ndarray):
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

    return enrichment, f_local, f_global, Ns


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
