import torch
from sklearn.decomposition import PCA

from tsprofiles.functions import extract_td_embeddings
from tsprofiles.functions import exclusion_knn_search


def tdpca_fnn(X, k, excl, d_pca, d_max):
    X_td = extract_td_embeddings(X, 1, d_pca, 1, "p_td").squeeze()
    pca = PCA()
    X_pc = torch.tensor(pca.fit_transform(X_td)).float()
    D_mean = torch.zeros(d_max - 1)
    for i, d in enumerate(range(1, d_max)):
        X_emb = X_pc[:, :d]
        D, I = exclusion_knn_search(X_emb.contiguous(), k, excl)
        D_mean[i] = torch.mean(D)
    return D_mean