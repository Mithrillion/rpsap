import numpy as np
import signatory
import torch
from iisignature import prepare, logsigtosig

# import pykeops.torch as ktorch
from ts_search.functions import *
from .transforms import rescale_path, unorm, rescale_signature, rectify_signature
from .tensor_product import tensor_exp, prepare_tensor_product


def sig_pow_iisig(sig, alpha, channels, depth):
    logsig = signatory.signature_to_logsignature(sig, channels, depth, mode="brackets")
    s = prepare(channels, depth, "S2")
    pow = torch.tensor(
        logsigtosig((alpha * logsig).cpu().numpy(), s),
        dtype=sig.dtype,
        device=sig.device,
    )
    return pow


def sig_pow(sig: torch.Tensor, alpha: float, channels: int, depth: int) -> torch.Tensor:
    """Given a signature sig and a power alpha, raise the signature to the alpha-th power using the log and exp map.
    S^alpha = exp(alpha * log(S))

    Parameters
    ----------
    sig : torch.Tensor
        _description_
    alpha : float
        _description_
    channels : int
        _description_
    depth : int
        _description_

    Returns
    -------
    torch.Tensor
        _description_
    """
    with torch.no_grad():
        logsig = signatory.signature_to_logsignature(
            sig, channels, depth, mode="expand"
        )
        pow = tensor_exp(alpha * logsig, channels, depth)
    return pow


def sig_conv(
    Q: torch.Tensor,
    T: torch.Tensor,
    depth: int,
    min_win: int,
    max_win: int,
    rescale=False,
    logsig_mode="rackets",
    norm_p=2,
):
    assert Q.shape[0] == 1 and T.shape[0] == 1
    window_range = max_win - min_win
    output_len = T.shape[1] - max_win
    with torch.no_grad():
        if rescale:
            T = rescale_path(T, depth)
            Q = rescale_path(Q, depth)
        # [batch, time, sig_terms]
        T_sigs = signatory.signature(T, depth, stream=True, basepoint=T[:, 0, :])
        T_invsigs = signatory.signature(
            T, depth, stream=True, basepoint=T[:, 0, :], inverse=True
        )
        T_sigs, T_invsigs = T_sigs.flatten(0, 1), T_invsigs.flatten(
            0, 1
        )  # [time, sig_terms]
        # Q_invs = signatory.signature(Q, depth, inverse=True).expand(output_len, -1)
        Q_logsigs = signatory.logsignature(Q, depth, mode=logsig_mode)
        all_dists_list = []
        for i in range(window_range):
            current_window_diffs = signatory.multi_signature_combine(
                [
                    T_invsigs[:output_len, :],
                    T_sigs[min_win + i : min_win + i + output_len, :],
                    # Q_invs,
                ],
                T.shape[-1],
                depth,
            )
            current_window_diffs = signatory.signature_to_logsignature(
                current_window_diffs, T.shape[-1], depth, mode=logsig_mode
            )
            # current_window_dists = torch.norm(current_window_diffs, p=norm_p, dim=-1)
            current_window_dists = 1 - torch.cosine_similarity(
                current_window_diffs, Q_logsigs, dim=-1
            )
            all_dists_list += [current_window_dists]
        all_dists = torch.stack(all_dists_list, dim=-1)
        D, I = torch.min(all_dists, dim=-1)
    return D, I


# TODO: might be more efficient if first calculate all segment sigs in parallel, then concatenate on expanding windows
def rolling_sig(
    X, depth, step, basepoint=None, inverse=False, alpha=1, exp_mode="pytorch"
):
    assert exp_mode in ["iisignature", "pytorch"]
    with torch.no_grad():
        # X assumed to be [batch, length, dims]
        breaks = torch.arange(0, X.shape[1], step)

        sig_dims = signatory.signature_channels(X.shape[-1], depth)
        sig_array = torch.zeros(
            X.shape[0], len(breaks), sig_dims, dtype=torch.float32, device=X.device
        )
        sig_map = signatory.Signature(depth, inverse=inverse)
        for idx in range(len(breaks)):
            if idx == 0:
                if step == 1 and basepoint is None:
                    sig_array[:, idx, :] = sig_map(
                        X[:, : breaks[idx + 1], :], basepoint=X[:, 0, :]
                    )
                else:
                    sig_array[:, idx, :] = sig_map(
                        X[:, : breaks[idx + 1], :], basepoint=basepoint
                    )
            else:
                last = sig_array[:, idx - 1, :]
                if alpha < 1:
                    # TODO: check alpha range
                    if exp_mode == "iisignature":
                        last = sig_pow_iisig(last, alpha, X.shape[-1], depth)
                    else:
                        last = sig_pow(last, alpha, X.shape[-1], depth)

                if idx == len(breaks) - 1:
                    sig_array[:, idx, :] = sig_map(
                        X[:, breaks[idx] :, :],
                        basepoint=X[:, breaks[idx] - 1, :],
                        initial=last,
                    )
                else:
                    sig_array[:, idx, :] = sig_map(
                        X[:, breaks[idx] : breaks[idx + 1], :],
                        basepoint=X[:, breaks[idx] - 1, :],
                        initial=last,
                    )
    return sig_array


def rolling_sig_bidirectional(X, depth, step, basepoint=None):
    # X assumed to be [batch (1), length, dims]
    with torch.no_grad():
        breaks = torch.arange(0, X.shape[1], step)

        sig_dims = signatory.signature_channels(X.shape[-1], depth)
        sig_array = torch.zeros(
            X.shape[0], len(breaks), sig_dims, dtype=torch.float32, device=X.device
        )
        invsig_array = torch.zeros(
            X.shape[0], len(breaks), sig_dims, dtype=torch.float32, device=X.device
        )
        sig_map = signatory.Signature(depth)
        invsig_map = signatory.Signature(depth, inverse=True)
        for idx in range(len(breaks)):
            if idx == 0:
                if step == 1 and basepoint is None:
                    sig_array[:, idx, :] = sig_map(
                        X[:, : breaks[idx + 1], :], basepoint=X[:, 0, :]
                    )
                    invsig_array[:, idx, :] = invsig_map(
                        X[:, : breaks[idx + 1], :], basepoint=X[:, 0, :]
                    )
                else:
                    sig_array[:, idx, :] = sig_map(
                        X[:, : breaks[idx + 1], :], basepoint=basepoint
                    )
                    invsig_array[:, idx, :] = invsig_map(
                        X[:, : breaks[idx + 1], :], basepoint=basepoint
                    )
            else:
                if idx == len(breaks) - 1:
                    sig_array[:, idx, :] = sig_map(
                        X[:, breaks[idx] :, :],
                        basepoint=X[:, breaks[idx] - 1, :],
                        initial=sig_array[:, idx - 1, :],
                    )
                    invsig_array[:, idx, :] = invsig_map(
                        X[:, breaks[idx] :, :],
                        basepoint=X[:, breaks[idx] - 1, :],
                        initial=invsig_array[:, idx - 1, :],
                    )
                else:
                    sig_array[:, idx, :] = sig_map(
                        X[:, breaks[idx] : breaks[idx + 1], :],
                        basepoint=X[:, breaks[idx] - 1, :],
                        initial=sig_array[:, idx - 1, :],
                    )
                    invsig_array[:, idx, :] = invsig_map(
                        X[:, breaks[idx] : breaks[idx + 1], :],
                        basepoint=X[:, breaks[idx] - 1, :],
                        initial=invsig_array[:, idx - 1, :],
                    )
    return sig_array, invsig_array


def rolling_logsig_profile(
    T,
    depth,
    step,
    basepoint=None,
    min_win=0,
    max_win=np.inf,
    dist_measure="cosine",
    logsig_mode="words",
    alpha=1,
    bidirection=False,
    rescale=False,
    exp_mode="pytorch",
):
    assert dist_measure in ["cosine", "ed", "normal"]
    assert exp_mode in ["pytorch", "iisignature"]
    if rescale:
        T = rescale_path(T, depth)
    if step == 1 and alpha == 1:
        temp_basepoint = basepoint if basepoint is not None else T[:, 0, :]
        T_logsigs = signatory.logsignature(
            T, depth, stream=True, basepoint=temp_basepoint, mode=logsig_mode
        )[0, ...]
    else:
        T_sigs = rolling_sig(
            T, depth, step, basepoint=basepoint, alpha=alpha, exp_mode=exp_mode
        ).flatten(0, 1)
        T_logsigs = signatory.signature_to_logsignature(
            T_sigs, T.shape[-1], depth, mode=logsig_mode
        )  # [len, ch]
    if dist_measure == "cosine":
        normed_sigs = unorm(T_logsigs)
        if bidirection:
            D, I = band_knn_search(
                normed_sigs,
                1,
                band=(min_win, max_win),
                block_size=1,
                dist="dot",
            )
        else:
            D, I = tri_knn_search(
                normed_sigs,
                normed_sigs,
                1,
                dist="dot",
                min_win=min_win,
                max_win=max_win,
            )  # [len, 1]
    elif dist_measure == "ed":
        if bidirection:
            D, I = band_knn_search(
                T_logsigs,
                1,
                band=(min_win, max_win),
                block_size=1,
                dist="ed",
            )
        else:
            D, I = tri_knn_search(
                T_logsigs,
                T_logsigs,
                1,
                dist="ed",
                min_win=min_win,
                max_win=max_win,
            )  # [len, 1]
    elif dist_measure == "normal":
        if bidirection:
            raise NotImplementedError()
            # D, I = band_knn_search(
            #     T_logsigs,
            #     1,
            #     band=(min_win, max_win),
            #     block_size=1,
            #     dist="normal",
            # )
        else:
            D, I = tri_knn_search(
                T_logsigs,
                T_logsigs,
                1,
                dist="normal",
                min_win=min_win,
                max_win=max_win,
            )  # [len, 1]
    return D, I


###############################################################
# old (but probably more flexible) methods
###############################################################


def sig_left_conv(
    Q,
    T,
    depth,
    step,
    basepoint=None,
    min_win=0,
    max_win=None,
    rescale=False,
    lookahead_mode="sig_ed",
    dist_mode="logsig",
    logsig_mode="words",
    dist_measure="ed",
):
    assert lookahead_mode in ["sig_ed", "logsig_cosine"]
    assert dist_mode in ["sig", "logsig"]
    assert dist_measure in ["ed", "cosine", "normal", "inverse"]

    if rescale:
        T = rescale_path(T, depth)
        Q = rescale_path(Q, depth)
    if step == 1:
        temp_basepoint = basepoint if basepoint is not None else T[:, 0, :]
        T_sigs = signatory.signature(T, depth, stream=True, basepoint=temp_basepoint)
        T_invsigs = signatory.signature(
            T, depth, stream=True, basepoint=temp_basepoint, inverse=True
        )
    else:
        T_sigs, T_invsigs = rolling_sig_bidirectional(
            T, depth, step, basepoint=basepoint
        )
    # [1, len, ch]
    Q_sigs = signatory.signature(Q, depth)
    # [1, ch]
    flat_T_sigs, flat_T_invsigs = T_sigs.flatten(0, 1), T_invsigs.flatten(0, 1)
    # [len, ch]
    flat_Q_sigs = torch.repeat_interleave(Q_sigs, len(flat_T_sigs), dim=0)
    # [len, ch]
    TQ = signatory.signature_combine(flat_T_sigs, flat_Q_sigs, T.shape[-1], depth)
    # [len, ch]

    if lookahead_mode == "sig_ed":
        I = tri_knn_search_index_only(
            TQ,
            flat_T_sigs,
            1,
            dist="ed",
            min_win=min_win,
            max_win=max_win,
            triangle="upper",
        )
    else:
        log_map = signatory.SignatureToLogSignature(
            T.shape[-1], depth, False, logsig_mode
        )
        I = tri_knn_search_index_only(
            unorm(log_map(TQ), -1),
            unorm(log_map(flat_T_sigs), -1),
            1,
            dist="dot",
            min_win=min_win,
            max_win=max_win,
            triangle="upper",
        )
    if min_win > 0:
        I = I[:-min_win, :]
        matched = signatory.signature_combine(
            flat_T_invsigs[:-min_win, :],
            flat_T_sigs[I[:, 0]].contiguous(),
            T.shape[-1],
            depth,
        )  # [len, ch]
    else:
        matched = signatory.signature_combine(
            flat_T_invsigs, flat_T_sigs[I[:, 0]].contiguous(), T.shape[-1], depth
        )  # [len, ch]
    if dist_mode == "logsig":
        match_logsigs = signatory.signature_to_logsignature(
            matched, T.shape[-1], depth, mode=logsig_mode
        )
        Q_logsigs = signatory.signature_to_logsignature(
            Q_sigs, T.shape[-1], depth, mode=logsig_mode
        )
        if dist_measure == "cosine":
            C = torch.cosine_similarity(Q_logsigs, match_logsigs)
        elif dist_measure == "normal":
            C = torch.norm(match_logsigs, dim=-1) * torch.sqrt(
                1 - torch.cosine_similarity(Q_logsigs, match_logsigs) ** 2
            )
        elif dist_measure == "inverse":
            raise NotImplementedError("Inverse dist mode is only available for sigs!")
        else:
            C = torch.norm(Q_logsigs - match_logsigs, dim=-1)
    else:
        if dist_measure == "cosine":
            C = torch.cosine_similarity(Q_sigs, matched)
        elif dist_measure == "normal":
            C = torch.norm(matched, dim=-1) * torch.sqrt(
                1 - torch.cosine_similarity(Q_sigs, matched) ** 2
            )
        elif dist_measure == "inverse":
            Q_invsigs = signatory.signature(Q, depth, inverse=True).repeat_interleave(
                len(matched), 0
            )
            TQ_inv = signatory.signature_combine(matched, Q_invsigs, T.shape[-1], depth)
            C = torch.norm(
                signatory.signature_to_logsignature(
                    TQ_inv, T.shape[-1], depth, mode=logsig_mode
                ),
                dim=-1,
            )
        else:
            C = torch.norm(Q_sigs - matched, dim=-1)

    return C, I


def sig_left_k_conv(
    Q,
    T,
    depth,
    basepoint=None,
    min_win=0,
    max_win=None,
    k=3,
    rescale=False,
    lookahead_mode="sig_ed",
    dist_mode="logsig",
    logsig_mode="words",
    dist_measure="ed",
):
    assert lookahead_mode in ["sig_ed", "logsig_cosine"]
    assert dist_mode in ["sig", "logsig"]
    assert dist_measure in ["ed", "cosine"]

    if rescale:
        T = rescale_path(T, depth)
        Q = rescale_path(Q, depth)
    temp_basepoint = basepoint if basepoint is not None else T[:, 0, :]
    T_sigs = signatory.signature(T, depth, stream=True, basepoint=temp_basepoint)
    T_invsigs = signatory.signature(
        T, depth, stream=True, basepoint=temp_basepoint, inverse=True
    )
    # [1, len, ch]
    Q_sigs = signatory.signature(Q, depth)
    # [1, ch]
    flat_T_sigs, flat_T_invsigs = T_sigs.flatten(0, 1), T_invsigs.flatten(0, 1)
    # [len, ch]
    flat_Q_sigs = torch.repeat_interleave(Q_sigs, len(flat_T_sigs), dim=0)
    # [len, ch]
    TQ = signatory.signature_combine(flat_T_sigs, flat_Q_sigs, T.shape[-1], depth)
    # [len, ch]

    if lookahead_mode == "sig_ed":
        I = tri_knn_search_index_only(
            TQ,
            flat_T_sigs,
            k,
            dist="ed",
            min_win=min_win,
            max_win=max_win,
            triangle="upper",
        )
    else:
        log_map = signatory.SignatureToLogSignature(
            T.shape[-1], depth, False, logsig_mode
        )
        I = tri_knn_search_index_only(
            unorm(log_map(TQ), -1),
            unorm(log_map(flat_T_sigs), -1),
            k,
            dist="dot",
            min_win=min_win,
            max_win=max_win,
            triangle="upper",
        )
    # [len, k]
    # TODO: make knn match and max possible
    if min_win > 0:
        I = I[:-min_win, :]
        first = flat_T_invsigs[:-min_win, None, :].expand(-1, k, -1)
    else:
        first = flat_T_invsigs[:, None, :].expand(-1, k, -1)
    second = flat_T_sigs[I, ...].contiguous()  # [len, k, ch]
    matched = signatory.signature_combine(
        first.flatten(0, 1), second.flatten(0, 1), T.shape[-1], depth
    )  # [len * k, ch]

    if dist_mode == "logsig":
        match_logsigs = signatory.signature_to_logsignature(
            matched, T.shape[-1], depth, mode=logsig_mode
        ).view(len(first), k, -1)  # [len, k, ch]
        Q_logsigs = signatory.signature_to_logsignature(
            Q_sigs, T.shape[-1], depth, mode=logsig_mode
        )
        if dist_measure == "cosine":
            C = 1 - torch.cosine_similarity(Q_logsigs[:, None, :], match_logsigs, dim=-1)
        else:
            C = torch.norm(Q_logsigs[:, None, :] - match_logsigs, dim=-1)
    else:
        if dist_measure == "cosine":
            C = 1 - torch.cosine_similarity(Q_sigs[:, None, :], matched.view_as(first), dim=-1)
        else:
            C = torch.norm(Q_sigs[:, None, :] - matched.view_as(first), dim=-1)
    C, MI = torch.min(C, dim=1)
    return C, I


def slow_sig_conv(
    Q: torch.Tensor,
    T: torch.Tensor,
    depth: int,
    min_win: int,
    max_win: int,
    rescale=False,
    logsig_mode="words",
    dist_mode="inverse",
    norm_p=2,
):
    assert Q.shape[0] == 1 and T.shape[0] == 1
    assert dist_mode in ["inverse", "cosine"]
    with torch.no_grad():
        if rescale:
            T = rescale_path(T, depth)
            Q = rescale_path(Q, depth)
        window_range = max_win - min_win
        output_len = T.shape[1] - max_win
        # [batch, time, sig_terms]
        rows = []
        sig_map = signatory.Signature(depth, True)
        for i in range(0, output_len):
            # row = []
            # for j in range(i + min_win, i + max_win):
            #     row += [T_path.signature(i, j)]
            # rows += [torch.cat(row, dim=0)]

            row = sig_map(T[:, i : i + max_win], basepoint=T[:, i, :])[:, min_win:, :]
            rows += [row]
        sig_mat = torch.cat(rows, dim=0)  # [output_len, win_range, sig_terms]

        if dist_mode == "inverse":
            Q_invsig = signatory.signature(Q, depth, inverse=True)
            Q_invs = Q_invsig.expand(output_len * window_range, -1)
            all_diffs = signatory.signature_combine(
                sig_mat.flatten(0, 1), Q_invs, T.shape[-1], depth
            )
            all_log_diffs = signatory.signature_to_logsignature(
                all_diffs, T.shape[-1], depth, mode=logsig_mode
            )
            all_dists = torch.norm(all_log_diffs, p=norm_p, dim=-1).reshape(
                output_len, window_range
            )
        elif dist_mode == "cosine":
            Q_logsigs = signatory.logsignature(Q, depth, mode=logsig_mode)
            logsig_mat = signatory.signature_to_logsignature(
                sig_mat.flatten(0, 1), T.shape[-1], depth, mode=logsig_mode
            )
            all_dists = (
                1 - torch.cosine_similarity(Q_logsigs, logsig_mat, -1)
            ).reshape(output_len, window_range)
        D, I = torch.min(all_dists, dim=-1)
    return D, I
