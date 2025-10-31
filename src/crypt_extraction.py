import numpy as np
import networkx as nx
import plotly.graph_objects as go

"""
Cleaned utilities to detect necks and crypts on an organoid graph using HKS.
- Short helpers appear first, then core methods in call order.
- Diagnostics-only utilities were removed; verbose printing and plotting remain.
"""


# ===========================
# Short helpers
# ===========================

def init_patch_labels(n_nodes: int, regions: list[set[int]]):
    """Initialize labels and patch sets from region seed sets."""
    labels = -1 * np.ones(n_nodes, dtype=int)
    patch_sets = []
    for pid, reg in enumerate(regions):
        s = set(reg)
        patch_sets.append(s)
        for u in s:
            labels[u] = pid
    return labels, patch_sets


def count_neighbors_in_set(G: nx.Graph, node: int, node_set: set[int]) -> int:
    """Count how many neighbors of 'node' are in 'node_set'."""
    c = 0
    for v in G.neighbors(node):
        if v in node_set:
            c += 1
    return c


def merge_touching_patches(G: nx.Graph, labels: np.ndarray, patch_sets: list[set[int]]):
    """Union-Find merge for patches that now touch through an edge.
    Returns (merged: bool, new_labels, new_patch_sets).
    """
    if len(patch_sets) <= 1:
        return False, labels, patch_sets

    parent = {pid: pid for pid in range(len(patch_sets))}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in G.edges():
        lu, lv = labels[u], labels[v]
        if lu >= 0 and lv >= 0 and lu != lv:
            union(lu, lv)

    reps = {pid: find(pid) for pid in range(len(patch_sets))}
    if all(pid == reps[pid] for pid in reps):
        return False, labels, patch_sets

    groups: dict[int, list[int]] = {}
    for old_id, rep in reps.items():
        groups.setdefault(rep, []).append(old_id)

    new_sets: list[set[int]] = []
    for _new_pid, rep in enumerate(groups.keys()):
        merged_set: set[int] = set()
        for old_id in groups[rep]:
            merged_set |= patch_sets[old_id]
        new_sets.append(merged_set)

    new_labels = -1 * np.ones_like(labels)
    for new_pid, s in enumerate(new_sets):
        for u in s:
            new_labels[u] = new_pid

    return True, new_labels, new_sets


def frontier_nodes_blocked(G: nx.Graph, patch_nodes: set[int], labels: np.ndarray, blocked_labels: np.ndarray | None = None) -> set[int]:
    """Unlabeled neighbors of patch_nodes, excluding nodes with blocked_labels>=0 (if provided)."""
    cand: set[int] = set()
    for u in patch_nodes:
        for v in G.neighbors(u):
            if labels[v] == -1:
                if blocked_labels is not None and blocked_labels[v] >= 0:
                    continue
                cand.add(v)
    return cand


def labels_to_regions(G: nx.Graph, labels: np.ndarray):
    """Connected components from 'labels>=0'. Returns (regions, remapped_labels 0..K-1)."""
    N = G.number_of_nodes()
    nodes = [n for n in G.nodes() if labels[n] >= 0]
    sub = G.subgraph(nodes)
    comps = [set(c) for c in nx.connected_components(sub)]
    new_labels = -1 * np.ones(N, dtype=int)
    for rid, comp in enumerate(comps):
        for u in comp:
            new_labels[u] = rid
    return comps, new_labels


def find_regions_from_mask(G: nx.Graph, mask: np.ndarray, min_region_size: int = 5):
    """Connected components on mask==True; keep only components with size>=min_region_size.
    Returns (regions: list[set[int]], labels: np.ndarray of shape (N,)).
    """
    if G.number_of_nodes() != mask.shape[0]:
        raise ValueError("G and mask size mismatch.")

    nodes = [n for n in G.nodes if mask[n]]
    sub = G.subgraph(nodes)
    comps = [set(c) for c in nx.connected_components(sub)]
    regions = [c for c in comps if len(c) >= min_region_size]

    labels = -1 * np.ones(G.number_of_nodes(), dtype=int)
    for rid, reg in enumerate(regions):
        for u in reg:
            labels[u] = rid

    return regions, labels


# ===========================
# Growth (build patches from seeds)
# ===========================

def grow_one_step_for_patch_generic_with_pid(
    G: nx.Graph,
    patch_sets: list[set[int]],
    pid: int,
    labels: np.ndarray,
    HKS_t0: np.ndarray,
    thr_t0: float,
    sign: int = +1,  # +1: HKS >= thr ; -1: HKS <= thr
    min_neighbors_in_patch: int = 2,
    blocked_labels: np.ndarray | None = None,
):
    """Single growth step for patch pid (crypt if sign=+1, neck if sign=-1).
    Adds at most one frontier node meeting threshold & connectivity; returns the added node or None.
    """
    patch = patch_sets[pid]
    cand = list(frontier_nodes_blocked(G, patch, labels, blocked_labels))
    if not cand:
        return None

    S = (lambda u: sign * HKS_t0[u])
    S_thr = sign * thr_t0
    cand.sort(key=lambda u: S(u), reverse=True)

    for u in cand:
        if S(u) >= S_thr and count_neighbors_in_set(G, u, patch) >= min_neighbors_in_patch:
            patch.add(u)
            labels[u] = pid
            return u
    return None


def grow_patches_until_stable(
    G: nx.Graph,
    regions: list[set[int]],
    HKS: np.ndarray,
    thr_t0: float,
    sign: int = +1,  # +1 crypts, -1 necks
    min_neighbors_in_patch: int = 2,
    max_outer_passes: int = 100,
    max_inner_steps_per_patch: int | None = None,
    verbose: bool = True,
    blocked_labels: np.ndarray | None = None,
):
    """Iteratively grow all patches seeded in 'regions' using HKS(:,0).
    Returns (labels, patch_sets) with labels>=0 for grown patches.
    Growth respects 'blocked_labels' (nodes already claimed by the other family).
    """
    n = G.number_of_nodes()
    HKS_t0 = HKS[:, 0]
    labels, patch_sets = init_patch_labels(n, regions)

    for outer in range(max_outer_passes):
        any_added_in_pass = False
        any_merged_in_pass = False

        pid = 0
        while pid < len(patch_sets):
            patch = patch_sets[pid]
            if not patch:
                pid += 1
                continue

            added_this_patch = 0
            while True:
                added = grow_one_step_for_patch_generic_with_pid(
                    G=G,
                    patch_sets=patch_sets,
                    pid=pid,
                    labels=labels,
                    HKS_t0=HKS_t0,
                    thr_t0=thr_t0,
                    sign=sign,
                    min_neighbors_in_patch=min_neighbors_in_patch,
                    blocked_labels=blocked_labels,
                )
                if added is None:
                    break
                added_this_patch += 1
                any_added_in_pass = True
                if max_inner_steps_per_patch is not None and added_this_patch >= max_inner_steps_per_patch:
                    break

            merged, labels, patch_sets = merge_touching_patches(G, labels, patch_sets)
            if merged:
                any_merged_in_pass = True
                pid = 0
                if verbose:
                    print(f"[outer {outer+1}] merged; restart. num_patches={len(patch_sets)}")
                continue

            if verbose:
                print(f"[outer {outer+1}] patch {pid}: +{added_this_patch} nodes; size={len(patch_sets[pid])}")
            pid += 1

        if verbose:
            total_size = sum(len(s) for s in patch_sets)
            print(
                f"== End outer pass {outer+1}: any_added={any_added_in_pass}, "
                f"any_merged={any_merged_in_pass}, patches={len(patch_sets)}, "
                f"nodes_in_patches={total_size}"
            )

        if not any_added_in_pass and not any_merged_in_pass:
            if verbose:
                print("Growth stabilized (generic patch-outer).")
            break

    return labels, patch_sets


# ===========================
# Loop detection on neck boundaries
# ===========================

def find_neck_loops(
    G: nx.Graph,
    neck_labels: np.ndarray,  # -1 not neck, >=0 neck region id
    crypt_labels: np.ndarray,  # -1 not crypt, >=0 crypt region id
    min_loop_len: int = 5,
    max_loop_len: int = 20,
    allow_gap: bool = False,
):
    """Identify one simple cycle per boundary-to-crypt neck component.
    Returns (loops: list[np.ndarray], boundary_regions: list[set[int]]).
    If allow_gap, permit one crypt node as a connector between boundary nodes.
    """
    N = G.number_of_nodes()
    if neck_labels.shape[0] != N or crypt_labels.shape[0] != N:
        raise ValueError("Label arrays must match number of graph nodes.")

    is_neck = (neck_labels >= 0)
    is_crypt = (crypt_labels >= 0)

    # boundary neck nodes: neck nodes adjacent to a crypt node
    boundary: set[int] = set()
    for u in G.nodes():
        if not is_neck[u]:
            continue
        for v in G.neighbors(u):
            if is_crypt[v]:
                boundary.add(u)
                break

    if not boundary:
        return [], []

    # boundary graph
    H = nx.Graph()
    H.add_nodes_from(boundary)

    # direct neck–neck edges on boundary
    for u, v in G.edges():
        if u in boundary and v in boundary:
            H.add_edge(u, v, gap=0)

    # optional: allow one crypt node as a gap
    if allow_gap:
        for u in list(boundary):
            for w in G.neighbors(u):
                if not is_crypt[w]:
                    continue
                for v in G.neighbors(w):
                    if v != u and v in boundary:
                        if H.has_edge(u, v):
                            if H[u][v].get("gap", 0) > 0:
                                H[u][v]["gap"] = 0
                        else:
                            H.add_edge(u, v, gap=1)

    boundary_regions = [set(c) for c in nx.connected_components(H)]

    # choose at most one loop per boundary region, prefer the longest (< max_loop_len)
    loops: list[np.ndarray] = []
    for reg in boundary_regions:
        if len(reg) < min_loop_len:
            continue
        subH = H.subgraph(reg).copy()
        cycles = nx.cycle_basis(subH)
        cand = [c for c in cycles if len(c) >= min_loop_len and len(c) < max_loop_len]
        if not cand:
            continue
        best = max(cand, key=lambda c: len(c))
        loops.append(np.asarray(best, dtype=int))

    return loops, boundary_regions


# ===========================
# Shrinking helpers
# ===========================

def _neck_boundary_nodes_crypt_only(G: nx.Graph, neck_labels: np.ndarray, crypt_labels: np.ndarray) -> set[int]:
    """Neck nodes that are adjacent to at least one crypt node."""
    is_neck = (neck_labels >= 0)
    is_crypt = (crypt_labels >= 0)
    boundary: set[int] = set()
    for u in G.nodes():
        if not is_neck[u]:
            continue
        for v in G.neighbors(u):
            if is_crypt[v]:
                boundary.add(u)
                break
    return boundary


def _sum_loop_lengths(
    G: nx.Graph,
    neck_labels: np.ndarray,
    crypt_labels: np.ndarray,
    min_loop_len: int = 5,
    max_loop_len: int = 20,
    allow_gap: bool = False,
):
    """Utility for shrink logic: compute total boundary loop length and list of loops."""
    loops, boundary_regions = find_neck_loops(
        G=G,
        neck_labels=neck_labels,
        crypt_labels=crypt_labels,
        min_loop_len=min_loop_len,
        max_loop_len=max_loop_len,
        allow_gap=allow_gap,
    )
    total_len = sum(len(c) for c in loops)
    return total_len, len(loops), loops, boundary_regions


def _would_merge_distinct_crypts(G: nx.Graph, u: int, crypt_labels: np.ndarray) -> bool:
    """Check if assigning u to crypt would connect two distinct crypt components."""
    nbr_labels = {crypt_labels[v] for v in G.neighbors(u) if crypt_labels[v] >= 0}
    return len(nbr_labels) >= 2


def _modal_adjacent_crypt_label(G: nx.Graph, u: int, crypt_labels: np.ndarray) -> int | None:
    """Most frequent adjacent crypt label for node u (None if no adjacent crypt)."""
    counts: dict[int, int] = {}
    for v in G.neighbors(u):
        lab = crypt_labels[v]
        if lab >= 0:
            counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ===========================
# Shrink necks while preserving topology
# ===========================

def shrink_neck_once(
    G: nx.Graph,
    HKS: np.ndarray,  # (N,T) uses column 0
    neck_labels: np.ndarray,  # -1 not neck, >=0 neck id
    crypt_labels: np.ndarray,  # -1 not crypt, >=0 crypt id
    thr_core_keep: float,  # protect nodes with HKS_t0 <= thr_core_keep
    min_loop_len: int = 5,
    max_loop_len: int = 20,
    allow_gap: bool = False,
):
    """Remove at most one boundary neck node (highest HKS first) if it doesn't
    increase total loop length or break all loops; may reassign the node to crypt.
    Returns (changed: bool, new_neck_labels, new_crypt_labels, info: dict).
    """
    HKS_t0 = HKS[:, 0]

    pre_total_len, pre_num_loops, pre_loops, pre_bregions = _sum_loop_lengths(
        G, neck_labels, crypt_labels, min_loop_len, max_loop_len, allow_gap
    )

    # candidates: boundary neck nodes sorted by descending HKS
    boundary = _neck_boundary_nodes_crypt_only(G, neck_labels, crypt_labels)
    cand = sorted(boundary, key=lambda u: HKS_t0[u], reverse=True)

    for u in cand:
        # Rule 1: protect core
        if HKS_t0[u] <= thr_core_keep:
            continue

        nbrs = list(G.neighbors(u))
        nbr_crypt = sum(1 for v in nbrs if crypt_labels[v] >= 0)
        nbr_unlab = sum(1 for v in nbrs if crypt_labels[v] < 0 and neck_labels[v] < 0)
        prefer_crypt = (nbr_crypt > nbr_unlab)  # tie -> unlabeled

        if prefer_crypt and _would_merge_distinct_crypts(G, u, crypt_labels):
            continue  # would connect distinct crypt regions

        new_neck = neck_labels.copy()
        new_crypt = crypt_labels.copy()

        # remove from neck
        old_neck_id = new_neck[u]
        new_neck[u] = -1

        # optionally assign to crypt by majority
        if prefer_crypt:
            lab = _modal_adjacent_crypt_label(G, u, new_crypt)
            if lab is not None:
                new_crypt[u] = lab

        post_total_len, post_num_loops, post_loops, post_bregions = _sum_loop_lengths(
            G, new_neck, new_crypt, min_loop_len, max_loop_len, allow_gap
        )

        if post_num_loops == 0 and pre_num_loops > 0:
            continue  # broke all loops
        if post_total_len > pre_total_len:
            continue  # loop length grew

        info = {
            "removed_node": int(u),
            "removed_node_hks": float(HKS_t0[u]),
            "prefer_crypt": bool(prefer_crypt),
            "pre_total_loop_len": int(pre_total_len),
            "post_total_loop_len": int(post_total_len),
            "pre_num_loops": int(pre_num_loops),
            "post_num_loops": int(post_num_loops),
            "old_neck_id": int(old_neck_id) if old_neck_id >= 0 else -1,
        }
        return True, new_neck, new_crypt, info

    # nothing could be removed
    return False, neck_labels, crypt_labels, {
        "removed_node": None,
        "reason": "no_valid_candidate",
        "pre_total_loop_len": int(pre_total_len),
        "pre_num_loops": int(pre_num_loops),
    }


def shrink_neck_until_stable(
    G: nx.Graph,
    HKS: np.ndarray,
    neck_labels: np.ndarray,
    crypt_labels: np.ndarray,
    thr_core_keep: float,
    min_loop_len: int = 5,
    max_loop_len: int = 20,
    allow_gap: bool = False,
    max_steps: int = 10,
    verbose: bool = True,
):
    """Iteratively call shrink_neck_once until no valid removal or max_steps.
    Returns (neck_labels, crypt_labels, neck_patches, crypt_patches, info_list).
    """
    steps = 0
    info_list: list[dict] = []
    cur_neck = neck_labels.copy()
    cur_crypt = crypt_labels.copy()

    while steps < max_steps:
        changed, new_neck, new_crypt, info = shrink_neck_once(
            G, HKS, cur_neck, cur_crypt, thr_core_keep,
            min_loop_len=min_loop_len, max_loop_len=max_loop_len, allow_gap=allow_gap,
        )
        info_list.append(info)
        if not changed:
            if verbose:
                print("Neck shrinking stabilized (no valid removal).")
            break
        steps += 1
        cur_neck, cur_crypt = new_neck, new_crypt
        if verbose:
            print(
                f"[shrink step {steps}] removed node {info['removed_node']} "
                f"(HKS={info['removed_node_hks']:.4f}), "
                f"prefer_crypt={info['prefer_crypt']}, "
                f"loops {info['pre_total_loop_len']}→{info['post_total_loop_len']}."
            )

    neck_patches, cur_neck_remapped = labels_to_regions(G, cur_neck)
    crypt_patches, cur_crypt_remapped = labels_to_regions(G, cur_crypt)

    return cur_neck_remapped, cur_crypt_remapped, neck_patches, crypt_patches, info_list


# ===========================
# Post-processing: promote unlabeled to crypt
# ===========================

def promote_unlabeled_to_crypt(
    G: nx.Graph,
    crypt_labels: np.ndarray,
    neck_labels: np.ndarray | None = None,
    min_unlabeled_size: int = 11,
):
    """Promote unlabeled connected components (size>=min_unlabeled_size) to crypt patches.
    Returns (new_crypt_labels, crypt_patches).
    """
    N = G.number_of_nodes()
    if crypt_labels.shape[0] != N:
        raise ValueError("crypt_labels length must match number of nodes.")
    if neck_labels is not None and neck_labels.shape[0] != N:
        raise ValueError("neck_labels length must match number of nodes.")

    is_crypt = (crypt_labels >= 0)
    if neck_labels is not None:
        is_neck = (neck_labels >= 0)
        unlabeled = (~is_crypt) & (~is_neck)
    else:
        unlabeled = (~is_crypt)

    unlabeled_nodes = [n for n in G.nodes() if unlabeled[n]]
    sub = G.subgraph(unlabeled_nodes)
    comps = [set(c) for c in nx.connected_components(sub)]

    new_labels = crypt_labels.copy()
    next_id = (int(np.max(new_labels)) + 1) if np.any(new_labels >= 0) else 0

    for comp in comps:
        if len(comp) >= min_unlabeled_size:
            for u in comp:
                new_labels[u] = next_id
            next_id += 1

    crypt_patches, new_labels = labels_to_regions(G, new_labels)
    return new_labels, crypt_patches


# ===========================
# End-to-end: seed detection and growth/shrink pipeline
# ===========================

def compute_curvature_proxy(cell_weighted_hks: np.ndarray,
                            HKS_times: np.ndarray,
                            eps: float = 1e-12) -> np.ndarray:
    """
    Compute a scale-normalized curvature proxy from HKS.

    Why this scaling:
    - Center per time-scale: subtracting the column-wise mean of the cell-weighted HKS
      emphasizes relative deviations within each time scale, which tends to push necks
      (often lower/negative HKS) and crypts (often higher/positive HKS) apart.
    - Time emphasis: multiplying by `HKS_times` lets you weight small-vs-large scales
      (e.g., to bias toward sharper necks or broader crypt bulges).
    - Per-scale z-score: dividing by the per-column std makes thresholds comparable
      across time-scales and across different organoids/datasets.

    Parameters
    ----------
    cell_weighted_hks : (N, T) array-like
        HKS per cell after any desired per-cell weighting; N = cells, T = time-scales.
    HKS_times : (T,) or (N, T) array-like
        Multiplicative weights per time-scale (or per cell & time-scale). A 1D (T,)
        will broadcast across cells.
    eps : float, default 1e-12
        Small numerical guard to avoid division by zero.

    Returns
    -------
    curvature_proxy : (N, T) ndarray
        Z-scored curvature proxy per cell and time scale.

    Notes
    -----
    - Uses NaN-aware mean/std; NaNs are ignored in the statistics and the result is
      sanitized with zeros for any all-NaN or infinite outcomes.
    """
    X = np.asarray(cell_weighted_hks, dtype=float)
    t = np.asarray(HKS_times, dtype=float)

    if t.ndim == 1:
        t = t[None, :]  # broadcast across cells

    # Center per time-scale (column-wise)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)

    # Apply time weighting
    proxy = t * Xc

    # Z-score per time-scale
    std = np.nanstd(proxy, axis=0, keepdims=True)
    denom = np.where(std > eps, std, 1.0)
    curvature_proxy = proxy / denom

    # Clean up any NaN/inf (e.g., all-NaN columns)
    curvature_proxy = np.nan_to_num(curvature_proxy, nan=0.0, posinf=0.0, neginf=0.0)
    return curvature_proxy


def extract_all_crypts(
    G: nx.Graph,                    # Organoid graph
    HKS: np.ndarray,                # Heat kernel signature at multiple times
    crypt_thresh: float = 1.00,     # Crypt HKS >= crypt_thresh
    neck_thresh: float = -1.00,     # Neck HKS <= neck_thresh
    growth_thresh: float = 0.0,     # Threshold for new nodes to be included when growing regions
    min_region_size: int = 5,       # Minimum number of nodes in regions (crypts, necks)
    verbose = False 
):
    """Main pipeline.
    1) Seed crypts/necks using all-scales HKS thresholds and region size filter.
    2) Grow necks (block crypts), then grow crypts (blocked by necks) using HKS(:,0).
    3) Promote large unlabeled components to crypts (e.g., villus).
    4) Shrink necks while preserving boundary-loop topology.

    Returns dict with:
      - "crypts": list[set[int]] final crypt patches
      - "necks" : list[set[int]] final neck patches
    """
    if HKS.ndim != 2 or HKS.shape[1] < 1:
        raise ValueError("HKS must be a 2D array with at least 1 column.")

    # --- Seeds (all-scales thresholds) ---
    high_mask = np.all(HKS >= crypt_thresh, axis=1)
    low_mask = np.all(HKS <= neck_thresh, axis=1)

    crypt_seed_regions, crypt_seed_labels = find_regions_from_mask(G, high_mask, min_region_size=min_region_size)
    neck_seed_regions, neck_seed_labels = find_regions_from_mask(G, low_mask, min_region_size=min_region_size)

    # --- Grow necks first (so they can block crypt growth) ---
    neck_labels, neck_patches = grow_patches_until_stable(
        G=G, regions=neck_seed_regions, HKS=HKS,
        thr_t0=growth_thresh,
        sign=-1,  # necks: HKS <= thr
        min_neighbors_in_patch=2,
        blocked_labels=crypt_seed_labels,  # forbid annexing crypt seeds
        verbose=verbose,
    )

    # --- Grow crypts (blocked by grown necks) ---
    crypt_labels, crypt_patches = grow_patches_until_stable(
        G=G, regions=crypt_seed_regions, HKS=HKS,
        thr_t0=growth_thresh,
        sign=+1,  # crypts: HKS >= thr
        min_neighbors_in_patch=2,
        blocked_labels=neck_labels,  # forbid annexing neck cells
        verbose=verbose,
    )

    # --- Promote large unlabeled components to crypts (e.g., villus) ---
    crypt_labels, crypt_patches = promote_unlabeled_to_crypt(
        G=G,
        crypt_labels=crypt_labels,
        neck_labels=neck_labels,
        min_unlabeled_size=11,  # >10 nodes
    )

    # --- Shrink necks while preserving topology (may also adjust crypt labels) ---
    neck_labels, crypt_labels, neck_patches, crypt_patches, _ = shrink_neck_until_stable(
        G=G,
        HKS=HKS,
        neck_labels=neck_labels,
        crypt_labels=crypt_labels,
        thr_core_keep=neck_thresh,  # protect very negative HKS nodes from removal; tune as needed
        min_loop_len=5,
        max_loop_len=100,
        allow_gap=False,
        max_steps=1000,
        verbose=verbose,
    )

    return crypt_patches, neck_patches


