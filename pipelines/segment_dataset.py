# pipelines/segment_dataset.py
import os
import numpy as np
from tqdm import tqdm

from io_utils.config import validate_cfg
from io_utils.index import iter_graph_records
from io_utils.segmentation_io import save_segmentation_npz, patches_to_ll
from io_utils.features_io import save_features_npz

def run_segmentation_dataset(
    cfg,
    *,
    timepoints=None,
    bin_centers=None,
    crypt_vocab_idx=None,
    neck_vocab_idx=None,
    segment_kwargs=None,
    save_features=True,
    debug=False,
    verbose=True,
    load_cell_graph_fn=None, # TODO: why is this general?
    mesh_loader_fn=None, # TODO: why is this general?
    graph_get_fn=None, # TODO: why is this general?
    segment_crypts_organoid_fn=None, 
):
    cfg = validate_cfg(cfg)
    blacklist = set(cfg.get("blacklist", []) or [])

    if bin_centers is None:
        raise ValueError("Provide bin_centers.")
    if crypt_vocab_idx is None:
        raise ValueError("Provide crypt_vocab_idx.")
    if segment_kwargs is None:
        segment_kwargs = {}

    for fn, name in [
        (load_cell_graph_fn, "load_cell_graph_fn"),
        (mesh_loader_fn, "mesh_loader_fn"),
        (graph_get_fn, "graph_get_fn"),
        (segment_crypts_organoid_fn, "segment_crypts_organoid_fn"),
    ]:
        if fn is None:
            raise ValueError(f"Provide {name}.")

    graphs_dir = cfg["graphs_dir"]
    seg_dir = cfg["seg_dir"]
    features_dir = cfg.get("features_dir", None)

    recs = list(iter_graph_records(graphs_dir, timepoints=timepoints))
    it = tqdm(recs, desc="segment") if verbose else recs

    for rec in it:
        tp = rec["timepoint"]
        gpath = rec["graph_path"]

        try:
            G = load_cell_graph_fn(gpath)
        except Exception as e:
            if verbose: print(f"[{tp}] graph load failed: {gpath} ({e})")
            continue

        label_uid = G.graph.get("label_uid", None)
        if label_uid is None:
            if verbose: print(f"[{tp}] graph missing label_uid: {gpath}")
            continue
        if label_uid in blacklist:
            continue

        mesh_path = cfg["mesh_path_from_label_uid"](label_uid)
        if not os.path.exists(mesh_path):
            if verbose: print(f"[{tp}] missing mesh for {label_uid}: {mesh_path}")
            continue

        try:
            mesh = mesh_loader_fn(mesh_path)
        except Exception as e:
            if verbose: print(f"[{tp}] mesh load failed for {label_uid}: {e}")
            continue

        # Optional: do your centering/rescale here consistently for *this dataset*
        # If you want it, add cfg["mesh_normalize_fn"] and call it here.

        try:
            out = segment_crypts_organoid_fn(
                G=G,
                mesh=mesh,
                bin_centers=bin_centers,
                crypt_vocab_idx=crypt_vocab_idx,
                neck_vocab_idx=neck_vocab_idx,
                debug=debug,
                **segment_kwargs,
            )
        except Exception as e:
            if verbose: print(f"[{tp}] segmentation failed for {label_uid}: {e}")
            continue

        if debug:
            crypts, villi, dnorm_v, L_crypt, Circ, dbg = out
        else:
            crypts, villi, dnorm_v, L_crypt, Circ = out
            dbg = None

        # compute dnorm_c for storage (cheap)
        try:
            proj_vertex_ids = graph_get_fn(G, "proj_vertex", dtype=np.int32)
        except Exception:
            proj_vertex_ids = None

        tp_seg_dir = os.path.join(seg_dir, tp) if tp is not None else seg_dir
        os.makedirs(tp_seg_dir, exist_ok=True)
        seg_path = os.path.join(tp_seg_dir, f"{label_uid}.npz")

        save_segmentation_npz(
            seg_path,
            label_uid=label_uid,
            crypts_ll=patches_to_ll(crypts),
            villi_ll=patches_to_ll(villi),
            bin_centers=bin_centers,
            dnorm_v=dnorm_v,
            L_crypt=L_crypt,
            Circ=Circ,
            extra={"debug": dbg} if dbg is not None else None,
        )

        if save_features and features_dir is not None:
            tp_feat_dir = os.path.join(features_dir, tp) if tp is not None else features_dir
            os.makedirs(tp_feat_dir, exist_ok=True)
            feat_path = os.path.join(tp_feat_dir, f"{label_uid}.npz")

            # Store only the arrays you actually reuse in downstream analysis
            try:
                markers_bin = graph_get_fn(G, "markers_bin")
            except Exception:
                markers_bin = None
            try:
                centroids = graph_get_fn(G, "centroid", dtype=float)
            except Exception:
                centroids = None

            save_features_npz(
                feat_path,
                label_uid=label_uid,
                markers_bin=markers_bin,
                proj_vertex_ids=proj_vertex_ids,
                centroids=centroids,
                extra={"graph_path": gpath, "mesh_path": mesh_path},
            )
