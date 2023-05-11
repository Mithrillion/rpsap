# import pwlf
import numpy as np
import pandas as pd
import signatory
import torch
import torch.nn.functional as F

# from sklearn.decomposition import PCA
from .transforms import unorm
from collections import namedtuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from ts_search.functions import (
    trivial_kfn_search,
    radius_exclusion_knn_search,
    tri_knn_search
)


##################################################################
# helper functions
##################################################################


def compute_signed_areas(X):
    # NOTE: assuming X has a "batch" dimension (which should always be 1 here)
    logsigs = signatory.logsignature(
        X,
        2,
        True,
        basepoint=X[:, 0, :],
        mode="brackets",
    )
    signed_areas = logsigs[0, :, X.shape[-1] :]
    return signed_areas


def compute_separable_logsigs(X):
    logsigs = signatory.logsignature(
        X,
        4,
        True,
        basepoint=X[:, 0, :],
        mode="brackets",
    )
    depth_1 = X.shape[-1]
    depth_2 = signatory.logsignature_channels(X.shape[-1], 2)
    signed_areas = logsigs[0, :, depth_1:depth_2]
    return signed_areas, logsigs
    # TODO: segmentwise version


def _make_batches(data_size, batch_size, drop_last=False, stride=None):
    if stride is None:
        stride = batch_size
    s = np.arange(0, data_size - batch_size + stride, stride, dtype=int)
    e = s + batch_size
    if drop_last:
        s, e = s[e < data_size], e[e < data_size]
    else:
        s, e = s[s < data_size], e[s < data_size]
        e[-1] = data_size
    return list(zip(s, e))


def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


##################################################################
# new persistence methods
##################################################################


RPResult = namedtuple(
    "RPResult",
    [
        "consec_cossim",
        "D_state",
        "I_state",
        "selected_SAs",
    ],
)

DDResult = namedtuple("DDResult", ["detection", "SA_PCs", "SA_profile"])


def _find_recurrence(
    X, recurrence_radius, recurrence_neighbours, recurrence_dist_measure, side="both"
):
    assert recurrence_dist_measure in ["ed", "znorm_ed"]
    assert side in ["both", "left", "right"]
    w, W = recurrence_radius
    if recurrence_dist_measure == "znorm_ed":
        X_input = X / torch.norm(X, 2, -1, keepdim=True)
    else:
        X_input = X
    if side == "both":
        D_state, I_state = radius_exclusion_knn_search(
            X_input, recurrence_neighbours, (w, W), "ed"
        )
    elif side == "left":
        D_state, I_state = tri_knn_search(
            X_input, X_input, recurrence_neighbours, "ed", w, W, "lower"
        )
    elif side == "right":
        D_state, I_state = tri_knn_search(
            X_input, X_input, recurrence_neighbours, "ed", w, W, "upper"
        )
    return D_state, I_state


def recurrence_persistence(
    X,
    recurrence_radius,
    recurrence_neighbours=3,
    recurrence_dist_measure="ed",
    persistence_comparison_gap=5,
    find_signed_areas=False,
    side="both",
):
    D_state, I_state = _find_recurrence(
        X, recurrence_radius, recurrence_neighbours, recurrence_dist_measure, side
    )
    SA = compute_signed_areas(X[None, ...])
    SA_inc = (SA[:, None, :] - SA[I_state]) * (
        (-1) ** (I_state > torch.arange(len(I_state))[:, None]).float()
    )[..., None]
    # similarities = torch.abs(
    #     F.cosine_similarity(
    #         SA_inc[:-persistence_comparison_gap, :, None, :],
    #         SA_inc[persistence_comparison_gap:, None, :, :],
    #         dim=-1,
    #     )
    # )
    similarities = F.cosine_similarity(
        SA_inc[:-persistence_comparison_gap, :, None, :],
        SA_inc[persistence_comparison_gap:, None, :, :],
        dim=-1,
    )
    consec_cossim = torch.amax(
        similarities,
        dim=(1, 2),
    )
    if not find_signed_areas:
        selected_SAs = None
    else:
        match_locations = torch.argmax(torch.amax(similarities, -1), -1)
        selected_SAs = SA_inc[: len(match_locations), ...][
            torch.arange(len(match_locations)), match_locations, :
        ]
    return RPResult(consec_cossim, D_state, I_state, selected_SAs)


def segmentwise_recurrence_persistence(
    X,
    recurrence_radius,
    seg_len,
    recurrence_neighbours=3,
    recurrence_dist_measure="ed",
    persistence_comparison_gap=5,
    find_signed_areas=False,
    side="both",
):
    # enlarge the buffer to include one radius plus buffer for logsigs
    seg_start_buffer = 2 * recurrence_radius[1]
    seg_end_buffer = recurrence_radius[1]
    D_state, I_state = _find_recurrence(
        X, recurrence_radius, recurrence_neighbours, recurrence_dist_measure, side
    )
    # pre-allocate aggregated results
    # NOTE: some of the allocated values may not be used
    consec_cossim = torch.zeros(len(X), dtype=X.dtype, device=X.device)
    if find_signed_areas:
        selected_SAs = torch.zeros(
            len(X),
            signatory.logsignature_channels(X.shape[-1], 2) - X.shape[-1],
            dtype=X.dtype,
            device=X.device,
        )
    else:
        selected_SAs = None
    batches = _make_batches(len(X), seg_len)
    # if batches[-1][1] - batches[-1][0] <= persistence_comparison_gap:
    #     batches = batches[:-1]
    for s, e in batches:
        # we need a buffer zone to include possible match positions for points in the segment
        # also need the buffer to give logsig enough "warmup"
        buffer_zone_start = max(0, s - seg_start_buffer)
        # we also include the gap as a suffix in the actual segment we take
        _e = e + persistence_comparison_gap if e < len(consec_cossim) else e
        buffer_zone_end = min(len(X), _e + seg_end_buffer)
        segment_SA = compute_signed_areas(X[None, buffer_zone_start:buffer_zone_end, :])
        # we need to ensure (I_state[s:e] - buffer_start) contains valid indices for the segement
        # and in the next step, exclude the increments from buffer zones
        segment_SA_inc = (
            segment_SA[
                s - buffer_zone_start : len(segment_SA) - (buffer_zone_end - _e),
                None,
                :,
            ]
            - segment_SA[I_state[s:_e] - buffer_zone_start]
        )
        # but now the problem is that the comparison gap shortens the similarity seq by the gap for
        # every segment! to fix this, we make each segment slightly longer as above
        segment_similarities = torch.abs(
            F.cosine_similarity(
                segment_SA_inc[:-persistence_comparison_gap, :, None, :],
                segment_SA_inc[persistence_comparison_gap:, None, :, :],
                dim=-1,
            )
        )
        # this way, we cancel out the extra "persistence_comparison_gap" we add to the segment end,
        # except for the last segment
        segment_consec_cossim = torch.amax(
            segment_similarities,
            dim=(1, 2),
        )
        consec_cossim[
            s : min(e, s + len(segment_consec_cossim))
        ] = segment_consec_cossim
        if find_signed_areas:
            match_locations = torch.argmax(torch.amax(segment_similarities, -1), -1)
            segment_selected_SAs = segment_SA_inc[: len(match_locations), ...][
                torch.arange(len(match_locations)), match_locations, :
            ]
            selected_SAs[
                s : min(e, s + len(segment_consec_cossim))
            ] = segment_selected_SAs
    return RPResult(consec_cossim, D_state, I_state, selected_SAs)


def cluster_directions(directions, k, **kwargs):
    clust = KMeans(n_clusters=k, **kwargs)
    labels = clust.fit_predict(directions.numpy())
    return labels, clust


def find_labels_from_cossim(
    consec_cossim, w, W, mean_threshold=0.99, min_threshold=0.6, enforce_min=False
):
    rp_left_mean = (
        pd.Series(consec_cossim)[::-1]
        .rolling(W // 2, 1, False, closed="both")
        .mean()
        .values[::-1]
    )
    rp_right_mean = (
        pd.Series(consec_cossim).rolling(W // 2, 1, False, closed="both").mean().values
    )
    rp_min = pd.Series(consec_cossim).rolling(w, 1, True, closed="both").min().values
    if enforce_min:
        rp_scores = (np.maximum(rp_left_mean, rp_right_mean) > mean_threshold) & (
            rp_min > min_threshold
        )
    else:
        rp_scores = np.maximum(rp_left_mean, rp_right_mean) > mean_threshold
    rp_left_running = (
        pd.Series(rp_scores)[::-1]
        .rolling(W, 1, False, closed="both")
        .sum()
        .values[::-1]
    )
    rp_right_running = (
        pd.Series(rp_scores).rolling(W, 1, False, closed="both").sum().values
    )
    rp_detection = (
        rp_scores
        & ((rp_left_running + rp_right_running) > W)
        & (rp_min > min_threshold)
    )
    return rp_detection


def find_labels_from_directions(SAs, w, W, threshold, pc_dims, pca_training_size=None):
    if not pca_training_size:
        SA_PCs = PCA(n_components=pc_dims).fit_transform(unorm(SAs, -1))
    else:
        pca = PCA(n_components=pc_dims).fit(unorm(SAs[:pca_training_size], -1))
        SA_PCs = pca.transform(unorm(SAs, -1))

    SA_rolling = pd.DataFrame(SA_PCs).rolling(w, 1, False, closed="both")
    SA_inv_rolling = pd.DataFrame(SA_PCs)[::-1].rolling(w, 1, False, closed="both")
    SA_profile_left = (
        SA_inv_rolling.quantile(0.95).values[::-1]
        - SA_inv_rolling.quantile(0.05).values[::-1]
    ).max(-1)
    SA_profile_right = (
        SA_rolling.quantile(0.95).values - SA_rolling.quantile(0.05).values
    ).max(-1)

    SA_profile = np.minimum(SA_profile_left, SA_profile_right)

    # SA_profile, _ = trivial_kfn_search(
    #     unorm(torch.tensor(SA_PCs).float(), -1), 1, w, "dot"
    # )
    # SA_profile = SA_profile[:, 0]
    rp_scores = SA_profile < threshold

    rp_left_running = (
        pd.Series(rp_scores)[::-1]
        .rolling(W, 1, False, closed="both")
        .sum()
        .values[::-1]
    )
    rp_right_running = (
        pd.Series(rp_scores).rolling(W, 1, False, closed="both").sum().values
    )
    rp_detection = rp_scores & ((rp_left_running + rp_right_running) > W)
    # rp_detection = (rp_left_running + rp_right_running) > win_len
    return DDResult(rp_detection, SA_PCs, SA_profile)


def get_trivial_mask(X, W, threshold=0.01):
    return pd.DataFrame(X).rolling(W, 1, True).std().values.max(-1) < threshold


##################################################################
# heuristic methods
##################################################################


def moving_deviation(mean_directions, alpha=0.8):
    last = mean_directions[0, :]
    dev = torch.zeros(
        len(mean_directions),
        dtype=mean_directions.dtype,
        device=mean_directions.device,
    )
    for i in range(1, len(mean_directions)):
        current = mean_directions[i, :]
        dev[i] = torch.dot(current, last)
        last = unorm(alpha * last + current, -1)
    return dev


def mean_SA_persistence(X, win_len, gap=1, return_directions=True):
    signed_areas = compute_signed_areas(X)
    W = torch.ones(1, 1, win_len) / win_len
    mean_SA = F.conv1d(
        signed_areas[:, None, :],
        W,
        padding="same",
    )[:, 0, :]
    mSA_diffs = mean_SA[gap:] - mean_SA[:-gap]
    mSA_directions = unorm(mSA_diffs, -1)
    consec_cossim = torch.sum(mSA_directions[gap:] * mSA_directions[:-gap], -1)
    returns = [consec_cossim]
    if return_directions:
        returns += [mSA_directions]
    return returns


def get_score_annotation(
    P,
    M,
    k,
    sim_win,
    outlier_win,
    cluster_count_win=None,
    avg_threshold=0.99,
    outlier_threshold=0.5,
    pure_state_threshold=2,
    return_clusters=True,
    return_running_cluster_counts=True,
    **kwargs
):
    cluster_count_win = cluster_count_win if cluster_count_win is not None else sim_win
    cluster_labels, _ = cluster_directions(M, k, **kwargs)
    cc = (
        pd.Series(cluster_labels)
        .rolling(cluster_count_win, 1, True)
        .apply(lambda x: np.unique(x).shape[0])
        .values
    )
    avg = pd.Series(P.numpy()).rolling(sim_win, 1, True).mean().values
    lb = pd.Series(P.numpy()).rolling(outlier_win, 1, True).min().values

    # high persistence condition
    p_ind = avg > avg_threshold
    # not outliers
    o_ind = lb > outlier_threshold
    # pure states
    c_ind = cc[: len(p_ind)] <= pure_state_threshold

    joint = p_ind * o_ind * c_ind
    # merged = (pd.Series(joint).rolling(merge_win, 1, True).mean() > 0.5).values
    returns = [joint]
    if return_clusters:
        returns += [cluster_labels]
    if return_running_cluster_counts:
        returns += [cc]
    return returns


def get_kfn_profile(M, neighbour_range=50, profile_win=50):
    D, I = trivial_kfn_search(M, 1, neighbour_range, dist="dot")
    kfn_profile = pd.Series(D[:, 0].numpy()).rolling(profile_win, 1, True).max().values
    return kfn_profile, D