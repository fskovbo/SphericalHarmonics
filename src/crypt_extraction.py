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
# Initial masks
# ===========================

def make_crypt_mask_from_HKS(HKS, crypt_threshold, times=None):
    """
    Crypt mask based on:
      1) HKS_t - mean_over_nodes(HKS_t) > crypt_threshold  for ALL t
      2) d/dt [ t * HKS_t ] > 0  for ALL t  (discrete forward diff)
    
    Parameters
    ----------
    HKS : np.ndarray, shape (N, T)
        Heat Kernel Signature per node (rows) and time-scale (cols).
        To restrict to first K times, pass HKS[:, :K].
    crypt_threshold : float
        Threshold for centered HKS.
    times : array-like or None
        Time values for each column. If None, uses 1..T.
    
    Returns
    -------
    mask : np.ndarray, shape (N,), dtype=bool
        True for nodes classified as crypt.
    """
    if HKS.ndim != 2 or HKS.shape[1] < 1:
        raise ValueError("HKS must be a 2D array with at least 1 column.")
    N, T = HKS.shape
    if times is None:
        times = np.arange(1, T + 1, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.shape[0] != T:
        raise ValueError("len(times) must equal HKS.shape[1].")

    # Center across nodes at each timepoint
    mu_t = np.nanmean(HKS, axis=0, keepdims=True)
    centered = HKS - mu_t  # (N,T)

    # cond_center = np.all(centered[:,0] > crypt_threshold, axis=1)
    # cond_first = centered[:, 0] > crypt_threshold
    cond_first = HKS[:, 0] > crypt_threshold

    # Monotonic increase of t * HKS_t (discrete derivative positive)
    scaled = HKS * times[np.newaxis, :]               # (N,T)
    if T < 2:
        raise ValueError("Need at least 2 time-scales to compute a derivative.")
    d_scaled = np.diff(scaled, axis=1)                # (N,T-1)
    cond_deriv = np.all(d_scaled > 0, axis=1)

    # Final mask
    return cond_first & cond_deriv


def make_neck_mask_from_HKS(HKS, neck_threshold, times=None, first_index=0):
    """
    Neck mask based on:
      1) HKS_t - mean_over_nodes(HKS_t) < neck_threshold  at a chosen first time (default column 0)
      2) d/dt [ HKS_t - mean_over_nodes(HKS_t) ] > 0  for ALL t (discrete forward diff)
    
    Parameters
    ----------
    HKS : np.ndarray, shape (N, T)
        Heat Kernel Signature per node (rows) and time-scale (cols).
        To restrict to first K times, pass HKS[:, :K].
    neck_threshold : float
        Threshold for centered HKS at the first time.
    times : array-like or None
        (Not used in the neck rule, included for symmetry/extension.)
    first_index : int
        Column index to use for the "t == 1" check (default 0 = first column).
    
    Returns
    -------
    mask : np.ndarray, shape (N,), dtype=bool
        True for nodes classified as neck.
    """
    if HKS.ndim != 2 or HKS.shape[1] < 1:
        raise ValueError("HKS must be a 2D array with at least 1 column.")
    N, T = HKS.shape
    if not (0 <= first_index < T):
        raise ValueError("first_index out of range.")
    if T < 2:
        raise ValueError("Need at least 2 time-scales to compute a derivative.")

    # Center across nodes at each timepoint
    mu_t = np.nanmean(HKS, axis=0, keepdims=True)
    centered = HKS - mu_t  # (N,T)

    # cond_first = centered[:, first_index] < neck_threshold
    cond_first = HKS[:, first_index] < neck_threshold


    # Monotonic increase of centered HKS over time
    d_centered = np.diff(centered, axis=1)   # (N,T-1)
    cond_deriv = np.all(d_centered > 0, axis=1)

    return cond_first & cond_deriv


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
    G,
    regions,
    HKS,
    thr_t0,
    sign=+1,                      # +1 crypts, -1 necks
    min_neighbors_in_patch=2,
    max_outer_passes=100,
    max_inner_steps_per_patch=None,
    verbose=True,
    blocked_labels=None,
    enforce_delta_pos=False       # NEW: for necks, require (HKS-mean)_last - (HKS-mean)_first > 0
):
    """
    Iteratively grow all patches seeded in 'regions' using HKS(:,0).

    If enforce_delta_pos is True *and* sign == -1 (necks), we only allow adding nodes
    where, letting f_t = HKS_t - mean_over_nodes(HKS_t), the quantity
        f_last - f_first  >  0
    i.e. the final deviation from the mean is greater than the initial.
    Nodes failing this test are treated as 'blocked' and never considered.

    Returns (labels, patch_sets) with labels>=0 for grown patches.
    Growth respects 'blocked_labels' (nodes already claimed by the other family).
    """
    n = G.number_of_nodes()
    HKS_t0 = HKS[:, 0]
    labels, patch_sets = init_patch_labels(n, regions)

    # --- Build an effective blocked mask if careful neck growth is requested ---
    eff_blocked = blocked_labels
    if enforce_delta_pos and sign == -1:
        # center HKS across nodes at each time (columns)
        mu_t = np.nanmean(HKS, axis=0, keepdims=True)
        centered = HKS - mu_t                       # (N,T)
        if centered.shape[1] < 2:
            raise ValueError("Need at least 2 time-scales to enforce delta condition.")
        delta = centered[:, -1] - centered[:, 0]    # (N,)
        allow = delta > 0
        # turn this into a "labels-like" blocked array: -1 means free, >=0 means blocked
        blocked_delta = np.where(allow, -1, 0).astype(int)

        if eff_blocked is None:
            eff_blocked = blocked_delta
        else:
            # combine: if either blocks, block
            if eff_blocked.shape[0] != blocked_delta.shape[0]:
                raise ValueError("blocked_labels length mismatch.")
            # any >=0 in either -> blocked (set to 0), else -1
            eff_blocked = np.where((eff_blocked >= 0) | (blocked_delta >= 0), 0, -1).astype(int)

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
                    blocked_labels=eff_blocked  # use effective blocked (includes delta rule if enabled)
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
    min_unlabeled_size: int = 5,
):
    """
    Promote unlabeled components to crypts, but *only within components of G \ neck*.
    This prevents promotions from bridging across a (supposed) neck barrier.

    Returns
    -------
    new_crypt_labels : np.ndarray (N,)  # remapped to 0..K-1
    crypt_patches    : list[set[int]]   # connected components after promotion
    """
    N = G.number_of_nodes()
    if crypt_labels.shape[0] != N:
        raise ValueError("crypt_labels length must match number of nodes.")
    if neck_labels is not None and neck_labels.shape[0] != N:
        raise ValueError("neck_labels length must match number of nodes.")

    # Masks
    is_crypt = (crypt_labels >= 0)
    is_neck  = (neck_labels >= 0) if neck_labels is not None else np.zeros(N, dtype=bool)
    unlabeled = (~is_crypt) & (~is_neck)

    # Work in the graph with neck nodes removed
    keep_nodes = [n for n in G.nodes() if not is_neck[n]]
    G_wo_neck = G.subgraph(keep_nodes)

    new_labels = crypt_labels.copy()
    next_id = (int(np.max(new_labels)) + 1) if np.any(new_labels >= 0) else 0

    # Connected components of G \ neck
    for CC in nx.connected_components(G_wo_neck):
        CC = set(CC)

        # If CC contains no crypt, skip (we only promote unlabeled *toward* existing crypt)
        if not any((new_labels[u] >= 0) for u in CC):
            continue

        # Unlabeled nodes within this CC
        unl_cc = [u for u in CC if unlabeled[u]]
        if not unl_cc:
            continue

        # Unlabeled components within this CC
        sub_unl = G.subgraph(unl_cc)
        comps = [set(c) for c in nx.connected_components(sub_unl)]

        for comp in comps:
            if len(comp) >= min_unlabeled_size:
                # Assign a temporary id; final merge happens below
                for u in comp:
                    new_labels[u] = next_id
                next_id += 1

    # === merge step: recompute components over all crypt-labeled nodes ===
    nodes_crypt = [n for n in G.nodes() if new_labels[n] >= 0]
    sub_crypt = G.subgraph(nodes_crypt)
    crypt_patches = [set(c) for c in nx.connected_components(sub_crypt)]

    remapped = -1 * np.ones(N, dtype=int)
    for rid, comp in enumerate(crypt_patches):
        for u in comp:
            remapped[u] = rid

    return remapped, crypt_patches


# ===========================
# Neck analysis
# ===========================


def _patches_to_labels(G, patches):
    """labels[u] = component id (0..K-1) if u in any patch, else -1."""
    N = G.number_of_nodes()
    lab = -1 * np.ones(N, dtype=int)
    for i, s in enumerate(patches):
        for u in s:
            lab[u] = i
    return lab

def _centered_hks_t0(HKS):
    mu = np.nanmean(HKS[:, 0])
    return HKS[:, 0] - mu

def _neck_boundary_nodes_to_crypt(G, neck_labels, crypt_labels, nodes_subset=None):
    """Neck nodes that have at least one crypt neighbor; optionally restrict to nodes_subset."""
    out = set()
    for u in (nodes_subset if nodes_subset is not None else G.nodes()):
        if neck_labels[u] < 0:
            continue
        for v in G.neighbors(u):
            if crypt_labels[v] >= 0:
                out.add(u); break
    return out

def _build_boundary_graph(G, boundary_nodes, allow_gap=False, is_crypt=None):
    """Aux graph over boundary neck nodes; optional 1-hop gap via a CRYPT intermediary."""
    H = nx.Graph()
    H.add_nodes_from(boundary_nodes)
    # direct boundary edges
    for u, v in G.edges():
        if u in boundary_nodes and v in boundary_nodes:
            H.add_edge(u, v, gap=0)
    if allow_gap:
        # 1-gap only through CRYPT:
        for u in list(boundary_nodes):
            for w in G.neighbors(u):
                if is_crypt is not None and not is_crypt[w]:
                    continue
                for v in G.neighbors(w):
                    if v != u and v in boundary_nodes:
                        if not H.has_edge(u, v):
                            H.add_edge(u, v, gap=1)
    return H

def _outside_neighbors_by_family(G, loop_nodes, neck_labels, crypt_labels, villus_labels=None):
    """
    Return:
      - crypt_neighbors_by_label: {crypt_label -> {nodes}}
      - villus_neighbors: set(nodes)    (if villus_labels provided)
    Only considers neighbors that are NOT neck and NOT in loop_nodes.
    """
    loop = set(loop_nodes)
    crypt_map = {}
    villus_set = set()
    for u in loop:
        for v in G.neighbors(u):
            if neck_labels[v] >= 0:     # still neck
                continue
            if v in loop:
                continue
            cl = crypt_labels[v]
            if cl >= 0:
                crypt_map.setdefault(cl, set()).add(v)
            elif villus_labels is not None and villus_labels[v] >= 0:
                villus_set.add(v)
    return crypt_map, villus_set

def _disconnects_groups_when_removed(G, loop_nodes, groups):
    """
    Check that removing loop_nodes disconnects all pairs of groups.
    groups: list of representative node sets (non-empty disjoint sets).
    Returns True if for every pair (A,B) there is NO path between any a in A and b in B in G \ loop.
    """
    removed = set(loop_nodes)
    # build component labels on residual graph
    keep_nodes = [u for u in G.nodes() if u not in removed]
    H = G.subgraph(keep_nodes)
    # map node -> cc id
    cc_id = {}
    for cid, comp in enumerate(nx.connected_components(H)):
        for u in comp:
            cc_id[u] = cid
    # For each group, collect its cc ids
    cc_sets = []
    for S in groups:
        ids = {cc_id[u] for u in S if u in cc_id}
        cc_sets.append(ids)
    # If any group is empty in residual (all its nodes were loop or neck), we treat it as disconnected
    # Otherwise, require disjoint component sets pairwise
    for i in range(len(cc_sets)):
        for j in range(i+1, len(cc_sets)):
            if cc_sets[i] & cc_sets[j]:
                return False
    return True

def identify_neck_circumferences(
    G,
    HKS,                      # (N, T), we use column 0 for curvature/score
    neck_patches,             # list[set[int]]
    crypt_patches,            # list[set[int]]
    villus_patches=None,      # list[set[int]] or None
    min_loop_len=5,
    max_loop_len=60,
    allow_gap=False
):
    """
    For each neck patch, pick a single circumference loop that:
      (i) lies on the neck–crypt boundary,
      (ii) is short (thin) and has strongly negative centered HKS at t0,
      (iii) actually separates outside groups (crypt–crypt or crypt–villus) when removed.

    Returns
    -------
    loops : list[np.ndarray]   # one per neck patch (may be empty array if none pass)
    scores: list[tuple]        # (length, mean_centered_hks) for the chosen loop (or None)
    """
    N = G.number_of_nodes()
    if HKS.ndim != 2 or HKS.shape[1] < 1:
        raise ValueError("HKS must be (N,T) with T>=1")

    # Build labels for quick lookup
    neck_labels  = _patches_to_labels(G, neck_patches)
    crypt_labels = _patches_to_labels(G, crypt_patches)
    villus_labels = _patches_to_labels(G, villus_patches) if villus_patches is not None else None
    is_crypt = (crypt_labels >= 0)
    z0_centered = _centered_hks_t0(HKS)

    loops_out = []
    scores_out = []

    for pid, patch in enumerate(neck_patches):
        # Boundary neck nodes (to CRYPT only) for this patch
        boundary_all = _neck_boundary_nodes_to_crypt(G, neck_labels, crypt_labels)
        boundary = boundary_all & set(patch)
        if len(boundary) < min_loop_len:
            loops_out.append(np.array([], dtype=int))
            scores_out.append(None)
            continue

        # Aux boundary graph + cycles
        H = _build_boundary_graph(G, boundary, allow_gap=allow_gap, is_crypt=is_crypt)
        if H.number_of_edges() == 0:
            loops_out.append(np.array([], dtype=int))
            scores_out.append(None)
            continue

        cycles = nx.cycle_basis(H)  # list of simple cycles (lists of nodes)
        # filter by length window
        cand = [c for c in cycles if (len(c) >= min_loop_len and len(c) < max_loop_len)]
        if not cand:
            loops_out.append(np.array([], dtype=int))
            scores_out.append(None)
            continue

        # Score + validate each candidate
        best = None
        best_score = None

        for c in cand:
            c_arr = np.asarray(c, dtype=int)
            L = len(c_arr)
            mean_neg = float(np.nanmean(z0_centered[c_arr]))  # more negative better
            score = (L, mean_neg)  # lexicographic: shortest, then most negative

            # Outside neighbors grouped
            crypt_neighbors_by_label, villus_neighbors = _outside_neighbors_by_family(
                G, c_arr, neck_labels, crypt_labels, villus_labels
            )

            groups = []
            if len(crypt_neighbors_by_label) >= 2:
                # crypt–crypt separation expected: pick one small sample per crypt group
                for s in crypt_neighbors_by_label.values():
                    if s:
                        # limit size to keep it cheap
                        groups.append(set(list(s)[: min(10, len(s))]))
            elif len(crypt_neighbors_by_label) == 1 and villus_labels is not None and len(villus_neighbors) > 0:
                # crypt–villus separation expected
                cset = next(iter(crypt_neighbors_by_label.values()))
                groups = [set(list(cset)[: min(10, len(cset))]),
                          set(list(villus_neighbors)[: min(10, len(villus_neighbors))])]
            else:
                # Not enough outside structure to validate — accept cautiously but still prefer shorter/more negative
                pass

            # If we have at least two groups, enforce barrier check
            if len(groups) >= 2:
                if not _disconnects_groups_when_removed(G, c_arr, groups):
                    continue  # reject side-rings that don't separate

            # pick best by score (shorter first; tie → more negative)
            if (best is None) or (score < best_score) or (score == best_score and L < best.shape[0]):
                best = c_arr
                best_score = score

        if best is None:
            loops_out.append(np.array([], dtype=int))
            scores_out.append(None)
        else:
            loops_out.append(best)
            scores_out.append(best_score)

    return loops_out, scores_out



# ===========================
# End-to-end: seed detection and growth/shrink pipeline
# ===========================


def split_villus_from_crypts(G, crypt_patches):
    """
    Pick villus patches per disconnected organoid:
      - For each connected component of G, select the largest crypt patch
        within that component as a 'villus'.
      - Return all selected villi and the remaining crypt patches.

    Parameters
    ----------
    G : networkx.Graph
        Full organoid graph (may have multiple connected components).
    crypt_patches : list[set[int]]
        Connected crypt patches (each a set of node indices).

    Returns
    -------
    villi : list[set[int]]
        One villus patch per graph component that contains crypts (may be 0, 1, or many).
    crypts_wo_villus : list[set[int]]
        All crypt patches excluding those chosen as villi.
    """
    if not crypt_patches:
        return [], []

    # Map each node -> graph component id
    comp_id = {}
    for i, comp_nodes in enumerate(nx.connected_components(G)):
        for u in comp_nodes:
            comp_id[u] = i

    # Group crypt patches by the graph component they lie in
    by_cc = {}
    for p in crypt_patches:
        # all nodes in a patch are connected, so they share the same G-component id
        any_node = next(iter(p))
        cid = comp_id.get(any_node, None)
        if cid is None:
            # node not in G (shouldn't happen); skip safely
            continue
        by_cc.setdefault(cid, []).append(p)

    villi = []
    chosen_ids = set()  # to exclude later

    # For each graph component: pick the largest crypt patch as villus
    for cid, patches in by_cc.items():
        if not patches:
            continue
        villus_patch = max(patches, key=len)
        villi.append(villus_patch)
        # remember identity by object id (safe since sets are the same objects we return)
        chosen_ids.add(id(villus_patch))

    # Remaining crypt patches = all not chosen as villi
    crypts_wo_villus = [p for p in crypt_patches if id(p) not in chosen_ids]

    return villi, crypts_wo_villus


def split_villus_from_crypts_by_hks(G, crypt_patches, HKS_col0, *, top_frac=0.5, min_cells_in_top=1):
    """
    Select villus patches per connected component using HKS:
      - For each patch: take the top `top_frac` of nodes by HKS_col0 (descending),
        compute the mean of those top values.
      - In each connected component of G, the patch with the *lowest* such mean is the villus.
      - Return those villi and the remaining patches as crypts.

    Parameters
    ----------
    G : networkx.Graph
        Full organoid graph (may have multiple connected components).
    crypt_patches : list[set[int]]
        Connected patches (each a set of node indices) that currently all count as “crypts”.
    HKS_col0 : array-like of shape (n_cells,)
        First HKS column for *all* cells in the organoid (same indexing as graph nodes).
    top_frac : float in (0,1]
        Fraction of highest-HKS nodes to keep per patch before averaging (default 0.5).
    min_cells_in_top : int >= 1
        Minimum number of nodes to average per patch (after NaN removal).

    Returns
    -------
    villi : list[set[int]]
        One villus patch per graph component that contains patches.
    crypts_wo_villus : list[set[int]]
        All patches excluding those chosen as villi.
    """
    if not crypt_patches:
        return [], []

    HKS_col0 = np.asarray(HKS_col0, dtype=float).reshape(-1)

    # Map each node -> graph component id
    comp_id = {}
    for cid, comp_nodes in enumerate(nx.connected_components(G)):
        for u in comp_nodes:
            comp_id[u] = cid

    # Group patches by the graph component they lie in
    by_cc = {}
    for p in crypt_patches:
        if not p:
            continue
        any_node = next(iter(p))
        cid = comp_id.get(any_node, None)
        if cid is None:
            # Node not present in G; skip safely
            continue
        by_cc.setdefault(cid, []).append(p)

    villi = []
    chosen_ids = set()

    for cid, patches in by_cc.items():
        if not patches:
            continue

        best_patch = None
        best_score = np.inf

        for p in patches:
            # restrict to nodes that exist in G (defensive)
            idx = np.array([u for u in p if u in comp_id], dtype=int)
            if idx.size == 0:
                continue

            vals = HKS_col0[idx]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                score = np.inf
            else:
                k = max(min_cells_in_top, int(np.ceil(top_frac * vals.size)))
                k = min(k, vals.size)
                # take top-k by value (descending) without fully sorting
                if k < vals.size:
                    topk = vals[np.argpartition(-vals, k-1)[:k]]
                else:
                    topk = vals
                score = float(np.mean(topk))

            if score < best_score:
                best_score = score
                best_patch = p

        if best_patch is not None:
            villi.append(best_patch)
            chosen_ids.add(id(best_patch))

    # Remaining patches are crypts
    crypts_wo_villus = [p for p in crypt_patches if id(p) not in chosen_ids]
    return villi, crypts_wo_villus



def segment_organoid(
    G: nx.Graph,                    # Organoid graph
    HKS: np.ndarray,                # Heat kernel signature at multiple times
    HKS_times: np.ndarray,          # Time-scales for HKS
    crypt_thresh: float = 1.00,     # Crypt HKS >= crypt_thresh
    neck_thresh: float = -1.00,     # Neck HKS <= neck_thresh
    growth_thresh: float = 0,       # Threshold for new nodes to be included when growing necks
    shrink_thresh: float = 0,       # Threshold for new nodes to be kept when shrinking necks
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
    crypt_mask = make_crypt_mask_from_HKS(HKS, crypt_thresh, HKS_times)
    neck_mask = make_neck_mask_from_HKS(HKS, neck_thresh, HKS_times)


    crypt_regions, crypt_labels = find_regions_from_mask(G, crypt_mask, min_region_size=min_region_size)
    neck_regions, neck_labels = find_regions_from_mask(G, neck_mask, min_region_size=min_region_size)

    # --- Grow necks first (so they can block crypt growth) ---
    neck_labels, neck_regions = grow_patches_until_stable(
        G=G,
        regions=neck_regions,
        HKS=HKS,                 # or a sliced early-time HKS
        thr_t0=neck_thresh + growth_thresh,  # your low-time threshold
        sign=-1,                 # necks
        min_neighbors_in_patch=2,
        blocked_labels=crypt_labels,   # keep blocking crypt
        enforce_delta_pos=True,        # << turn on the new constraint
        verbose=verbose
    )


    # --- Promote large unlabeled components to crypts (e.g., villus) ---
    crypt_labels, crypt_regions = promote_unlabeled_to_crypt(
        G=G,
        crypt_labels=crypt_labels,
        neck_labels=neck_labels,
        min_unlabeled_size=min_region_size,  # >5 nodes
    )

    # --- Shrink necks while preserving topology (may also adjust crypt labels) ---
    neck_labels, crypt_labels, neck_regions, crypt_regions, _ = shrink_neck_until_stable(
        G=G,
        HKS=HKS,
        neck_labels=neck_labels,
        crypt_labels=crypt_labels,
        thr_core_keep=neck_thresh+shrink_thresh,  # protect very negative HKS nodes from removal; tune as needed
        min_loop_len=5,
        max_loop_len=100,
        allow_gap=False,
        max_steps=1000,
        verbose=verbose,
    )

    # villus_regions, crypt_regions, = split_villus_from_crypts(G, crypt_regions)
    villus_regions, crypt_regions = split_villus_from_crypts_by_hks(G, crypt_regions, HKS[:,0])

    return crypt_regions, neck_regions, villus_regions


