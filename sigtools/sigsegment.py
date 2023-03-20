from sigtools.signed_areas import *
from sigtools.sigconv import *
from sigtools.transforms import *
from scipy.signal import find_peaks


def nsmall(x, k):
    if len(x) < k:
        return np.max(x)
    else:
        return np.partition(x, k - 1)[k - 1]


def sigsegment(
    X,
    depth,
    burnin,
    r,
    excl,
    w,
    W,
    sig_inc_threshold,
    sig_excl_threshold,
    sig_plat_threshold,
    n_small_counts,
    k=1,
):
    head_logsigs = signatory.logsignature(X[:, :burnin, :], depth, True, mode="expand")
    C = exclusion_count_rnn_search(unorm(head_logsigs[0, ...], -1), r, excl, "dot")
    max_val, max_pos = torch.max(C[:, 0], dim=0)

    S, I = sig_left_k_conv(
        X[:, :max_pos, :],
        X,
        depth,
        min_win=w,
        max_win=W,
        k=k,
        rescale=True,
        lookahead_mode="logsig_cosine",
        dist_mode="logsig",
        logsig_mode="brackets",
        dist_measure="cosine",
    )
    S = 1 - S

    S_right_max = pd.Series(S.numpy()).rolling(10, 1, False).max().values
    sig_plateau_threshold = np.maximum(S_right_max - 0.05, sig_plat_threshold)
    sig_matches = find_peaks(S, height=sig_inc_threshold, distance=16)
    highs = np.where(
        (S.numpy() > sig_plateau_threshold) & (S.numpy() > sig_plateau_threshold)
    )
    sig_matches = np.unique(
        np.sort(np.concatenate([sig_matches[0], highs[0]]), kind="mergesort")
    )
    sig_final_matches = []
    for p1, p2 in zip(sig_matches[:-1], sig_matches[1:]):
        # if np.min(S[p1:p2].numpy()) > sig_excl_threshold:
        if nsmall(S[p1:p2].numpy(), n_small_counts) > sig_excl_threshold:
            continue
        sig_final_matches += [p1]
    sig_final_matches = np.array(sig_final_matches)
    sig_final_matches = sig_final_matches[sig_final_matches > burnin]

    return sig_final_matches, max_pos, S, I
