import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pickle

from mesh.OrganoidMesh import OrganoidMesh





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





