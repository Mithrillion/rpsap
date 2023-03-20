import numpy as np
import torch
import torch.nn.functional as F
import typing
import pykeops.torch as ktorch
from pykeops.torch.cluster import from_matrix
from scipy.sparse import coo_matrix, eye, diags

#############################################
# Data manipulation
#############################################


def extract_td_embeddings(
    arr: typing.Union[np.ndarray, torch.Tensor],
    delta_t: int,
    embedding_dim: int,
    skip_step: int,
    dim_order: str = "dpt",  # [(per-time) dim, (OG seq) position, (subseq) time]
) -> torch.tensor:
    source_tensor = torch.tensor(arr) if isinstance(arr, np.ndarray) else arr
    td_embedding = F.unfold(
        source_tensor.T.view(1, source_tensor.shape[1], 1, source_tensor.shape[0]),
        (1, embedding_dim),
        dilation=delta_t,
        stride=skip_step,
    ).view(source_tensor.shape[1], embedding_dim, -1)
    if dim_order == "dpt":
        td_embedding = td_embedding.permute(0, 2, 1)
    elif dim_order == "dtp":
        pass
    elif dim_order == "ptd":
        td_embedding = td_embedding.permute(2, 1, 0)
    elif dim_order == "p_td":
        td_embedding = td_embedding.permute(2, 1, 0).flatten(-2, -1)
    else:
        raise ValueError("Invalid dim_order string!")
    return td_embedding


def purge_zeros_in_sparse_tensor(X: torch.Tensor, tolerance=1e-20) -> torch.Tensor:
    if not X.is_coalesced():
        X = X.coalesce()
    indices, values = X.indices(), X.values()
    zero_ind = values <= tolerance
    return torch.sparse_coo_tensor(
        indices[:, ~zero_ind], values[~zero_ind], size=X.shape, device=X.device
    ).coalesce()


def purge_large_in_sparse_tensor(X: torch.Tensor, tolerance=1) -> torch.Tensor:
    if not X.is_coalesced():
        X = X.coalesce()
    indices, values = X.indices(), X.values()
    large_ind = values >= tolerance
    return torch.sparse_coo_tensor(
        indices[:, ~large_ind], values[~large_ind], size=X.shape, device=X.device
    ).coalesce()


def to_scipy_sparse(X: torch.Tensor) -> coo_matrix:
    if not X.is_coalesced():
        X = X.coalesce()
    indices, values = X.indices().numpy(), X.values().numpy()
    return coo_matrix((values, (indices[0, :], indices[1, :])), shape=X.shape)


def to_torch_sparse(X: coo_matrix, dtype=torch.float) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.stack([torch.tensor(X.row), torch.tensor(X.col)], dim=0),
        X.data,
        dtype=dtype,
    ).coalesce()


def zero_rows_and_cols(M, idx):
    diag = eye(M.shape[1]).tolil()
    for i in idx:
        diag[i, i] = 0
    res = diag.dot(M).dot(diag)
    res.eliminate_zeros()
    return res


def to_torch_sparse(X: coo_matrix, dtype=torch.float) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.stack([torch.tensor(X.row), torch.tensor(X.col)], dim=0),
        X.data,
        X.shape,
        dtype=dtype,
    ).coalesce()


def make_batches(data_size, batch_size, drop_last=False, stride=None):
    if stride is None:
        stride = batch_size
    s = np.arange(0, data_size - batch_size + stride, stride)
    e = s + batch_size
    if drop_last:
        s, e = s[e < data_size], e[e < data_size]
    else:
        s, e = s[s < data_size], e[s < data_size]
        e[-1] = data_size
    return list(zip(s, e))


#############################################
# kNN searches
#############################################


def knn_search(
    feats: torch.Tensor, k: int, dist: str = "ed"
) -> typing.Tuple[torch.tensor, torch.tensor]:
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
        D *= -1
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def exclusion_knn_search(
    feats: torch.Tensor, k: int, excl: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    Diag_ij = float(excl) - (I_i - I_j).abs()
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij.ifelse(np.inf, D_ij)
    D, I = D_ij.Kmin_argKmin(K=k, dim=1)
    if dist == "ed":
        D = torch.sqrt(D)
    return D, I


def prune_by_exclusion(
    D: torch.Tensor, I: torch.Tensor, k: int, excl: int
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError()


def radius_exclusion_knn_search(
    feats: torch.Tensor, k: int, band: typing.Tuple[int, int], dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    excl, radius = band
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    index_dists = (I_i - I_j).abs()
    Diag_ij_excl = float(excl) - index_dists
    Diag_ij_radius = index_dists - float(radius)
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij_excl.ifelse(np.inf, D_ij)
    D_ij = Diag_ij_radius.ifelse(np.inf, D_ij)
    D, I = D_ij.Kmin_argKmin(K=k, dim=1)
    if dist == "ed":
        D = torch.sqrt(D)
    return D, I


def _search_with_ranges(feats, k, ij_ranges, dist):
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = D_ij.Kmin_argKmin(k, 1, ranges=ij_ranges)
        D = torch.sqrt(D)
    elif dist == "ed_max":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(k, 1, ranges=ij_ranges)
        D = torch.sqrt(torch.abs(D))
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(k, 1, ranges=ij_ranges)
        D *= -1
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def band_knn_search(
    feats: torch.Tensor,
    k: int,
    band: typing.Tuple[int, int],
    block_size: int = 1,
    dist: str = "ed",
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    excl, radius = band
    x_ranges = torch.tensor(make_batches(len(feats), block_size)).int()
    if radius == np.inf:
        radius = len(x_ranges)
    pos_diags = np.arange(excl, min(excl + radius, len(x_ranges)))
    keep = torch.tensor(
        diags(
            [1] * len(pos_diags),
            pos_diags,
            (len(x_ranges), len(x_ranges)),
            dtype=bool,
        ).toarray()
    )
    keep = keep + keep.t()
    ij_ranges = from_matrix(x_ranges, x_ranges, keep)
    D, I = _search_with_ranges(feats, k, ij_ranges, dist)
    return D, I


# TODO: implement block-band knn search (more efficient)


def trivial_kfn_search(
    feats: torch.tensor, k: int, incl: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Inverted kNN search within the 'inclusion zone' to find maximum
    distance changes w.r.t. small phase drift

    Args:
        feats (torch.tensor): _description_
        k (int): _description_
        incl (int): _description_

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    Diag_ij = (I_i - I_j).abs() - float(incl)
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij.ifelse(0, D_ij)
    D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
    D *= -1
    if dist == "ed":
        D = torch.sqrt(D)
    return D, I


def mean_dist_profile(
    feats: torch.tensor, k: int, incl: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    Diag_ij = (I_i - I_j).abs() - float(incl)
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij.ifelse(0, D_ij)
    if dist == "ed":
        D = D_ij.sqrt().sum(dim=1) / len(feats)
    else:
        D = D_ij.sum(dim=1) / len(feats)
    return D


def cross_knn_search(
    A: torch.Tensor, B: torch.Tensor, k: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(A[:, None, :])
    X_j = ktorch.LazyTensor(B[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
        D *= -1
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def cross_knn_search_index_only(
    A: torch.Tensor, B: torch.Tensor, k: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(A[:, None, :])
    X_j = ktorch.LazyTensor(B[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        I = D_ij.argKmin(K=k, dim=1)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        I = (-D_ij).argKmin(K=k, dim=1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return I


#############################################
# triangular and banded-triangular kNN searches
#############################################


def tri_knn_search(
    TQ: torch.Tensor,
    T: torch.Tensor,
    k: int,
    dist: str = "ed",
    min_win=0,
    max_win=None,
    triangle="lower",
):
    X_i = ktorch.LazyTensor(TQ[:, None, :])
    X_j = ktorch.LazyTensor(T[None, :, :])
    indices = torch.arange(len(T), device=T.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    if max_win == np.inf:
        max_win = len(T)
    if triangle == "lower":
        diff = I_i - I_j
    elif triangle == "upper":
        diff = I_j - I_i
    else:
        raise ValueError(
            f"triangle value must be 'upper' or 'lower', but got {triangle}!"
        )
    win_LB = diff - min_win
    win_UB = max_win - diff
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D_ij = win_LB.ifelse(D_ij, np.inf)
        D_ij = win_UB.ifelse(D_ij, np.inf)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D_ij = win_LB.ifelse(D_ij, -np.inf)
        D_ij = win_UB.ifelse(D_ij, -np.inf)
        D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
        D *= -1
    elif dist == "normal":
        NX_i = X_i.normalize()
        NX_j = X_j.normalize()
        S_ij = (NX_i * NX_j).sum(-1)
        D_ij = X_j.sqnorm2() * (1 - S_ij.square())
        D_ij = win_LB.ifelse(D_ij, np.inf)
        D_ij = win_UB.ifelse(D_ij, np.inf)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def tri_knn_search_index_only(
    TQ: torch.Tensor,
    T: torch.Tensor,
    k: int,
    dist: str = "ed",
    min_win=0,
    max_win=None,
    triangle="lower",
):
    X_i = ktorch.LazyTensor(TQ[:, None, :])
    X_j = ktorch.LazyTensor(T[None, :, :])
    indices = torch.arange(len(T), device=T.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    if max_win == np.inf:
        max_win = len(T)
    if triangle == "lower":
        diff = I_i - I_j
    elif triangle == "upper":
        diff = I_j - I_i
    else:
        raise ValueError(
            f"triangle value must be 'upper' or 'lower', but got {triangle}!"
        )
    win_LB = diff - min_win
    win_UB = max_win - diff
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D_ij = win_LB.ifelse(D_ij, np.inf)
        D_ij = win_UB.ifelse(D_ij, np.inf)
        I = D_ij.argKmin(K=k, dim=1)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D_ij = win_LB.ifelse(D_ij, -np.inf)
        D_ij = win_UB.ifelse(D_ij, -np.inf)
        I = (-D_ij).argKmin(K=k, dim=1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return I