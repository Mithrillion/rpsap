import signatory
import torch
from itertools import product


# fastest


# exp(x) = 1 + x + x/2(x + x/3(x + x/4(x + ...)))
def tensor_exp(logsig, dims, depth):
    result = torch.zeros_like(logsig, device=logsig.device)
    for level in range(depth - 1, 0, -1):
        term = (
            signatory.signature_combine(logsig, result, dims, depth) - logsig - result
        ) / (level + 1)
        result = logsig + term
    return result


# log(1 + x) = x - x(x/2 - x(x/3 - x(x/4 - ...)))
def tensor_log1p(sig, dims, depth):
    result = torch.zeros_like(sig, device=sig.device)
    max_level = depth - 1
    for level in range(max_level, 0, -1):
        term = signatory.signature_combine(sig, result, dims, depth) - sig - result
        result = sig / level - term
    return result


def lie_bracket(logsig_a, logsig_b, dims, depth):
    mult_by_id = logsig_a + logsig_b
    left = signatory.signature_combine(logsig_a, logsig_b, dims, depth) - mult_by_id
    right = signatory.signature_combine(logsig_b, logsig_a, dims, depth) - mult_by_id
    return left - right


# alternatives


def get_term_level_breaks(dims, depth):
    end = 0
    term_length = 1
    ends = [0, 1]
    for d in range(1, depth + 1):
        start = end
        term_length *= dims
        end = start + term_length
        ends += [end + 1]
    return torch.tensor(ends).long()


def append_scalar_term(sig):
    return torch.cat([torch.ones(len(sig), 1, device=sig.device), sig], dim=-1)


def append_zero_term(sig):
    return torch.cat([torch.zeros(len(sig), 1, device=sig.device), sig], dim=-1)


def get_slots(dims, depth):
    all_words = [()] + signatory.all_words(dims, depth)
    word_order_dict = {word: i for i, word in enumerate(all_words)}
    product_words = [
        word_order_dict[x[0] + x[1]]
        for x in product(all_words, all_words)
        if len(x[0] + x[1]) <= depth
    ]
    return torch.tensor(product_words).long()


def prepare_tensor_product(dims, depth):
    breaks = get_term_level_breaks(dims, depth)
    slots = get_slots(dims, depth)
    return breaks, slots


# NOTE: this is the same as signatory.signature_combine()
def tensor_product_torch(sig_a, sig_b, dims, depth, scalar_term=1):
    sigtensor_channels = sig_a.size(-1)
    if sig_a.shape != sig_b.shape:
        raise ValueError("sig_a and sig_b must have consistent shapes!")
    if signatory.signature_channels(dims, depth) != sigtensor_channels:
        raise ValueError(
            "Given a sigtensor with {} channels, a path with {} channels and a depth of {}, which are "
            "not consistent.".format(sigtensor_channels, dims, depth)
        )
    breaks = get_term_level_breaks(dims, depth)
    if scalar_term == 1:
        sig_a_, sig_b_ = append_scalar_term(sig_a), append_scalar_term(sig_b)
    else:
        sig_a_, sig_b_ = append_zero_term(sig_a), append_zero_term(sig_b)
    flat_prods = []
    for left_level in range(depth + 1):
        left_terms = sig_a_[:, breaks[left_level] : breaks[left_level + 1]]
        right_terms = sig_b_[:, : breaks[-left_level - 1]]
        tensor_prod = left_terms[:, :, None] * right_terms[:, None, :]
        flat_prods += [tensor_prod.flatten(-2, -1)]
    flat_prod = torch.cat(flat_prods, dim=1)
    slots = get_slots(dims, depth)
    # TODO: avoid explicitly using the scalar dimension
    result = torch.zeros(len(sig_a), sigtensor_channels + 1, device=sig_a.device)
    result.index_add_(1, slots, flat_prod)
    return result[:, 1:]


def tensor_product_prepared(sig_a_, sig_b_, depth, breaks, slots):
    flat_prods = torch.zeros(len(sig_a_), len(slots), device=sig_a_.device)
    pos = 0
    for left_level in range(depth + 1):
        left_terms = sig_a_[:, breaks[left_level] : breaks[left_level + 1]]
        right_terms = sig_b_[:, : breaks[-left_level - 1]]
        tensor_prod = left_terms[:, :, None] * right_terms[:, None, :]
        this_prod = tensor_prod.flatten(-2, -1)
        flat_prods[:, pos : pos + this_prod.shape[-1]] = this_prod
        pos = pos + this_prod.shape[-1]
    # TODO: avoid explicitly using the scalar dimension
    result = torch.zeros_like(sig_a_, device=sig_a_.device)
    result.index_add_(1, slots, flat_prods)
    return result


# NOTE: "correct" but slower
def tensor_product_alt(sig_a, sig_b, depth, breaks_w_0):
    breaks = breaks_w_0[1:] - 1
    result = torch.zeros_like(sig_a, device=sig_a.device)
    for level in range(1, depth + 1):
        mult_by_id = (
            sig_a[..., breaks[level - 1] : breaks[level]]
            + sig_b[..., breaks[level - 1] : breaks[level]]
        )
        result[..., breaks[level - 1] : breaks[level]] += mult_by_id

        for left_level in range(1, level):
            right_level = level - left_level
            left = sig_a[..., breaks[left_level - 1] : breaks[left_level]]
            right = sig_b[..., breaks[right_level - 1] : breaks[right_level]]
            prod = (left[:, :, None] * right[:, None, :]).flatten(-2, -1)
            result[..., breaks[level - 1] : breaks[level]] += prod
    return result


def tensor_product_no_id(logsig_a, logsig_b, depth, breaks_w_0):
    breaks = breaks_w_0[1:] - 1
    result = torch.zeros_like(logsig_a, device=logsig_a.device)
    for level in range(2, depth + 1):
        for left_level in range(1, level):
            right_level = level - left_level
            left = logsig_a[..., breaks[left_level - 1] : breaks[left_level]]
            right = logsig_b[..., breaks[right_level - 1] : breaks[right_level]]
            prod = (left[:, :, None] * right[:, None, :]).flatten(-2, -1)
            result[..., breaks[level - 1] : breaks[level]] += prod
    return result


# cf. iisignature (https://github.com/bottler/iisignature/)
def tensor_exp_impl(logsig, dims, depth):
    logsig_channels = logsig.size(-1)
    if signatory.signature_channels(dims, depth) != logsig_channels:
        raise ValueError(
            "Given a logsigtensor with {} channels, a path with {} channels and a depth of {}, which are "
            "not consistent. Check that the logsig is in expanded (tensor algebra) format!".format(
                logsig_channels, dims, depth
            )
        )
    max_level = depth - 1
    breaks, slots = prepare_tensor_product(dims, depth)
    logsig_ = append_zero_term(logsig)
    result_ = torch.zeros_like(logsig_, device=logsig.device)
    for level in range(max_level, 0, -1):
        term = tensor_product_prepared(
            1.0 / (level + 1) * logsig_,
            result_,
            depth,
            breaks,
            slots,
        )
        result_ = logsig_ + term
    return result_[:, 1:]


def tensor_exp_prepared(logsig, depth, breaks, slots):
    max_level = depth - 1
    logsig_ = append_zero_term(logsig)
    result_ = torch.zeros_like(logsig_, device=logsig.device)
    for level in range(max_level, 0, -1):
        term = tensor_product_prepared(
            1.0 / (level + 1) * logsig_,
            result_,
            depth,
            breaks,
            slots,
        )
        result_ = logsig_ + term
    return result_[:, 1:]
