import numpy as np

# import pykeops.torch as ktorch
from ..ts_search.functions import *


def find_next_recurrence(X, t, horizon=500, exclusion=5, eps=1):
    curr_state = X[[t], :]
    assert t + exclusion < X.shape[0]
    state_diff = np.linalg.norm(
        X[t + exclusion : min(X.shape[0], t + horizon)] - curr_state, ord=2, axis=-1
    )
    (recurr_points,) = np.nonzero(state_diff < eps)
    return recurr_points + exclusion