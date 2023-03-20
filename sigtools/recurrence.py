import numpy as np
import pandas as pd

# import pykeops.torch as ktorch
from tsprofiles.functions import *
from scipy.sparse.csgraph import connected_components


def find_next_recurrence(X, t, horizon=500, exclusion=5, eps=1):
    curr_state = X[[t], :]
    assert t + exclusion < X.shape[0]
    state_diff = np.linalg.norm(
        X[t + exclusion : min(X.shape[0], t + horizon)] - curr_state, ord=2, axis=-1
    )
    (recurr_points,) = np.nonzero(state_diff < eps)
    return recurr_points + exclusion


def find_max_recurrence(X, k, exclusion, eps):
    D, I = exclusion_knn_search(X, k, exclusion)
    G = knn_entries_to_sparse_dists(D, I)
    G = remove_temporal_drift_from_knn(D, I, exclusion, k)
    skd = find_local_density(D, k)
    G = sparse_distance_to_membership(G, skd, True)
    G = sparse_make_symmetric(G)
    G = purge_zeros_in_sparse_tensor(G, eps)
    G_sp = to_scipy_sparse(G)
    n_components, labels = connected_components(G_sp, directed=False)
    counts = pd.Series(labels).value_counts()
    return D, I, G, n_components, labels, counts
