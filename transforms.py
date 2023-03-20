import numpy as np
from scipy.interpolate import splrep, splev
import torch
import math
from sklearn.decomposition import PCA


def to_td(X, tau=1, d=2):
    # last dimension assumed to be time
    if d < 2:
        raise ValueError("TD dimension must be at least 2!")
    return torch.tensor(
        np.stack([X[..., i * tau : -(d - i) * tau] for i in range(d)], axis=-1)
    ).float()


def znorm(x, dim=-1):
    return (x - x.mean(dim, keepdim=True)) / (1e-8 + x.std(dim, keepdim=True))


def unorm(x, dim=-1):
    return x / (1e-8 + torch.norm(x, dim=dim, keepdim=True))


def to_td_pca(X, dims=2, tau=1, d=10, pretrained_pca=None):
    X_td = to_td(X, tau, d)
    batch_dims = X_td.shape[:-1]
    if pretrained_pca:
        pca = pretrained_pca
        Z = pca.transform(X_td.reshape(-1, X_td.shape[-1]))
    else:
        pca = PCA(n_components=dims)
        Z = pca.fit_transform(X_td.reshape(-1, X_td.shape[-1]))
    return torch.tensor(Z).float().view(*batch_dims, -1), pca


def to_steps(X, bins, cp=None):
    cuts = np.linspace(0, 1, bins)
    if cp is None:
        cut_points = np.quantile(X.reshape(-1), cuts, interpolation="linear")
        # cut_points = np.linspace(0, 1, bins)
    else:
        cut_points = cp
    steps = np.digitize(X, cut_points)
    ext_cut_points = np.concatenate([cut_points[[0]], cut_points, cut_points[[-1]]])
    return ext_cut_points[steps], cut_points


def to_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, (channel)]
    X_repeat = X.repeat_interleave(2, dim=1)

    # Split out lead and lag
    lead = X_repeat[:, 1:, :]
    lag = X_repeat[:, :-1, :]

    # Combine
    X_leadlag = torch.cat((lead, lag), 2)

    return X_leadlag


def to_nd_leadlag(X):
    if not torch.is_tensor(X):
        X = torch.tensor(X).float()
    # [batch, time, channel]
    n_chs = X.shape[-1]
    X_repeat = X.repeat_interleave(n_chs * 2, dim=1)
    streams = []
    for i in range(2):
        for j in range(n_chs):
            streams += [
                X_repeat[
                    :, 2 * n_chs - 2 * i - j - 1 : (X_repeat.shape[1] - 2 * i - j), j
                ]
            ]
    X_leadlag = torch.stack(streams, 2)
    return X_leadlag


def resample(X, resample, smooth=0, k=3):
    # TODO: batch optimisation?
    spls = [
        splrep(np.linspace(0, 1, X.shape[-1]), X[i, :], k=k, s=smooth)
        for i in range(X.shape[0])
    ]
    X_rs = [splev(np.linspace(0, 1, resample), spl) for spl in spls]
    return np.stack(X_rs, axis=0)


def rescale_path(path, depth):
    coeff = math.factorial(depth) ** (1 / depth)
    return coeff * path
