from sigtools.signed_areas import *
from sigtools.sigconv import *
from sigtools.transforms import *


RPSAPResult = namedtuple(
    "RPSAPResult",
    [
        "detection",
        "consec_cossim",
        "D_state",
        "I_state",
        "SAs",
        "SA_PCs",
        "SA_profile",
        "PCA",
        "td_data",
    ],
)


def rpsap(
    X,
    d,
    n_pcs,
    w,
    W,
    k,
    gap,
    agg_win,
    threshold,
    min_agg_win=None,
    label_mode="directions",
    trivial_threshold=None,
):
    assert label_mode in ["persistence", "directions", "union", "intersection"]
    min_agg_win = w if min_agg_win is None else min_agg_win
    td_data, pca = to_td_pca(X, n_pcs, 1, d)
    consec_cossim, D_state, I_state, SAs = recurrence_persistence(
        td_data[0, ...], (w, W), k, "ed", gap, True
    )
    if label_mode in ["persistence", "union", "intersection"]:
        detection_per = find_labels_from_cossim(consec_cossim, w, agg_win)
    if label_mode in ["directions", "union", "intersection"]:
        detection_dir, SA_PCs, SA_profile = find_labels_from_directions(
            SAs, agg_win // 2, agg_win, threshold, 2, None
        )
    match label_mode:
        case "persistence":
            detection = detection_per
            SA_PCs, SA_profile = None, None
        case "directions":
            detection = detection_dir
        case "union":
            detection = detection_per | detection_dir
        case "intersection":
            detection = detection_per & detection_dir
    if trivial_threshold is not None:
        is_trivial = get_trivial_mask(X.flatten(), agg_win, trivial_threshold)
        detection[is_trivial[: len(detection)]] = 0
    return RPSAPResult(
        detection,
        consec_cossim,
        D_state,
        I_state,
        SAs,
        SA_PCs,
        SA_profile,
        pca,
        td_data,
    )


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


def enforce_label_consec_runs(x, min_runs):
    run_values, run_starts, run_lengths = find_runs(x)
    x_filtered = x.copy()
    is_pos_value = run_values == 1
    is_short = is_pos_value & (run_lengths < min_runs)
    is_long = is_pos_value & (run_lengths >= min_runs)
    for s, e in zip(run_starts[is_short], run_starts[is_short] + run_lengths[is_short]):
        x_filtered[s:e] = 0
    return x_filtered, run_starts[is_long], run_lengths[is_long]
