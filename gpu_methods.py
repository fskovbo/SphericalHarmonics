
import numpy as np
import cupy as cp
from scipy.sparse.linalg import splu
from scipy import sparse

from mesh_analysis import build_G_face

def compute_geodesics_GPU(mesh, t=None, sources=None):
    """
    Fast GPU-accelerated Heat Method for geodesic distances on a mesh using
    CPU sparse solves for linear systems and batched GPU dense operations.

    Parameters
    ----------
    mesh : OrganoidMesh
        Must provide:
            - v : (V,3) vertex coordinates
            - f : (F,3) triangle faces
            - laplacian : (V,V) sparse cotangent Laplacian
            - mass_matrix : (V,V) sparse lumped mass matrix
    t : float, optional
        Heat diffusion time. If None, set ~ mean_edge_length^2.
    sources : array of int, optional
        Indices of source vertices. If None, compute for all vertices (O(V²)).

    Returns
    -------
    D_out : (S,V) ndarray
        Geodesic distances from each source to all vertices.
    """
    V = mesh.v.shape[0]
    F = mesh.f.shape[0]

    # --- Define sources ---
    if sources is None:
        sources = np.arange(V, dtype=np.int32)
    else:
        sources = np.asarray(sources, dtype=np.int32)
    S = len(sources)

    # --- CPU sparse LU decompositions ---
    reg = 1e-12
    L_factor = splu((mesh.laplacian + reg * sparse.eye(V, dtype=mesh.laplacian.dtype)).tocsc())
    if t is None:
        edges = np.vstack([mesh.f[:, [0,1]], mesh.f[:, [1,2]], mesh.f[:, [2,0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        mean_edge = np.mean(np.linalg.norm(mesh.v[edges[:,0]] - mesh.v[edges[:,1]], axis=1))
        t = mean_edge**2
    A_factor = splu((mesh.mass_matrix + t*mesh.laplacian).tocsc())

    # --- GPU arrays ---
    G_face = build_G_face(mesh.v, mesh.f)           # (F,3,3)
    G_face_gpu = cp.asarray(G_face)
    face_areas_gpu = cp.asarray(mesh.calc_face_areas())

    # --- Build RHS for all sources ---
    delta = np.zeros((V, S), dtype=np.float64)
    delta[sources, np.arange(S)] = 1.0
    rhs = mesh.mass_matrix @ delta  # (V,S)

    # --- Solve heat equation per source on CPU ---
    U = np.zeros_like(rhs)
    for i in range(S):
        U[:, i] = A_factor.solve(rhs[:, i])

    # --- Move batch to GPU ---
    U_gpu = cp.asarray(U.T)  # (S,V)

    # --- Compute grad_u per face for all sources ---
    U_f = U_gpu[:, mesh.f]                # (S,F,3)
    grad_u = cp.einsum('sfa,fai->sfi', U_f, G_face_gpu)  # (S,F,3)

    # --- Normalize vector field ---
    norms = cp.linalg.norm(grad_u, axis=2)
    norms = cp.where(norms==0, 1e-12, norms)
    X_face = -(grad_u / norms[:,:,None])

    # --- Divergence accumulation ---
    contrib = cp.einsum('sfi,fai->sfa', X_face, G_face_gpu)
    contrib *= face_areas_gpu[None,:,None]

    # --- Move back to CPU and accumulate to vertices ---
    contrib_cpu = cp.asnumpy(contrib)
    D_out = np.zeros((S, V), dtype=np.float64)
    for s_idx in range(S):
        div = np.zeros(V, dtype=np.float64)
        for a in range(3):
            np.add.at(div, mesh.f[:, a], contrib_cpu[s_idx, :, a])
        # --- Solve Poisson: L φ = div (CPU) ---
        phi = L_factor.solve(div)
        phi -= phi[sources[s_idx]]
        D_out[s_idx, :] = phi

    return D_out
