import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# sys.path.append("../external/")
sys.path.append("..")
sys.path.append("../..")
# from STEM import SSPDetector as ssp

from wavetools import lowpass_filtering
from Recofit_commons import *
from sigtools.transforms import *
from sigtools.signed_areas import *

# from NASC import maxNASC
import optuna

# hv.extension("bokeh")
# hv.opts.defaults(hv.opts.Curve(width=800, height=250))

fs = 50


def random_drop(X, label, mask, ratio, p=None):
    indices = np.random.choice(
        np.arange(1, len(X)), int(len(X) * (1 - ratio)), replace=False, p=p
    )
    indices = np.sort(indices)
    return X[indices], label[indices], mask[indices]


random_modification_ratio = 0

# win_len = int(100 * (1 - random_modification_ratio))
# w, W = int(40 * (1 - random_modification_ratio)), int(
#     100 * (1 - random_modification_ratio)
# )
# std_w = 40
# std_threshold = 0.24
# ac_threshold = 0.4
# smoothing_w = 500

# loading data
dat, fs, groupings_map = load_data()
mats, all_activities_df = prepare_data(dat, fs, groupings_map)


def NASC_eval(
    subject_id,
    random_modification_ratio,
    n_pcs,
    d,
    w,
    W,
    k,
    gap,
    min_agg_win,
    agg_win,
    mean_threshold,
    min_threshold,
    trivial_threshold,
    enforce_min,
):
    rep_annot = generate_annotation(mats, all_activities_df, subject_id).astype(bool)
    mask = generate_mask(mats, all_activities_df, subject_id).astype(bool)
    this_mat = mats[subject_id][:, 1:]

    if random_modification_ratio > 0:
        np.random.seed(7777)
        # NOTE: set uneven warping
        p = 2 + np.sin(np.arange(len(this_mat) - 1) / 500)
        p /= np.sum(p)
        this_mat, rep_annot, mask = random_drop(
            this_mat, rep_annot, mask, random_modification_ratio, p
        )
    this_mat = lowpass_filtering(this_mat, 3, fs, 5)

    # applying RecPersistence
    with torch.no_grad():
        # td_data = (
        #     to_td(torch.tensor(this_mat.copy()).T, 3, 2)
        #     .permute(1, 0, 2)
        #     .flatten(-2, -1)[None, ...]
        # )
        td_data, pca = to_td_pca(torch.tensor(this_mat.copy())[None, ...], n_pcs, 1, d)
        consec_cossim, D_state, I_state, SAs = recurrence_persistence(
            td_data[0, ...], (w, W), k, "ed", gap, True
        )
        # consec_cossim, D_state, I_state, SAs = segmentwise_recurrence_persistence(
        #     td_data[0, ...], (w, W), int(0.5e4), k, "ed", gap, find_signed_areas=True
        # )
    detection = find_labels_from_cossim(
        consec_cossim,
        min_agg_win,
        agg_win,
        mean_threshold,
        min_threshold,
        enforce_min=enforce_min,
    )
    is_trivial = get_trivial_mask(this_mat, agg_win, trivial_threshold)
    detection[is_trivial[: len(detection)]] = 0

    # evaluating - naive method
    detection = np.array(detection)
    target = rep_annot[: len(detection)]
    # naive_acc = accuracy_score(target, detection)
    # naive_prec = precision_score(target, detection)
    # naive_rec = recall_score(target, detection)
    # naive_f1 = f1_score(target, detection)

    # evaluating - masking
    detection_masked = detection[~mask[: len(detection)]]
    target_masked = target[~mask[: len(detection)]]
    # masked_acc = accuracy_score(target_masked, detection_masked)
    # masked_prec = precision_score(target_masked, detection_masked)
    # masked_rec = recall_score(target_masked, detection_masked)
    masked_f1 = f1_score(target_masked, detection_masked)

    return masked_f1


def objective(trial):
    n_pcs = trial.suggest_int("n_pcs", 3, 12, step=1)
    # d = trial.suggest_int("d", 15, 50, step=1)
    d = 20
    w = trial.suggest_int("w", 50, 100, step=10)
    W = trial.suggest_int("W", 200, 1000, step=100)
    k = trial.suggest_int("k", 4, 20, step=2)
    gap = trial.suggest_int("gap", 5, 50, step=1)
    min_agg_win = trial.suggest_int("min_agg_win", 30, 70, step=10)
    agg_win = trial.suggest_int("agg_win", 200, 1200, step=100)
    mean_threshold = trial.suggest_float("mean_threshold", 0.950, 0.995, step=0.005)
    min_threshold = trial.suggest_float("min_threshold", 0.00, 0.95, step=0.05)
    trivial_threshold = trial.suggest_float("trivial_threshold", 0.02, 0.20, step=0.02)
    enforce_min = trial.suggest_categorical("enforce_min", [True, False])

    f1_list = []
    for subject_id in np.arange(25):
        f1 = NASC_eval(
            subject_id,
            random_modification_ratio,
            n_pcs,
            d,
            w,
            W,
            k,
            gap,
            min_agg_win,
            agg_win,
            mean_threshold,
            min_threshold,
            trivial_threshold,
            enforce_min,
        )
        f1_list += [f1]

    return -np.mean(f1_list)


sampler = optuna.samplers.TPESampler(seed=7777)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=160, n_jobs=2)

best_f1 = objective(study.best_trial)

# new

last_best = {'n_pcs': 6,
 'w': 50,
 'W': 1000,
 'k': 20,
 'gap': 33,
 'min_agg_win': 30,
 'agg_win': 900,
 'mean_threshold': 0.995,
 'min_threshold': 0.05,
 'trivial_threshold': 0.04,
 'enforce_min': True}

# old

# {'n_pcs': 8,
#  'd': 34,
#  'w': 60,
#  'W': 250,
#  'k': 8,
#  'gap': 33,
#  'agg_win': 850,
#  'mean_threshold': 0.97,
#  'trivial_threshold': 0.14}

# {'w': 75,
#  'W': 800,
#  'k': 10,
#  'gap': 15,
#  'agg_win': 350,
#  'mean_threshold': 0.995,
#  'min_threshold': 0.25,
#  'trivial_threshold': 0.04}

# {'w': 60,
#  'W': 950,
#  'k': 7,
#  'gap': 34,
#  'min_agg_win': 35,
#  'agg_win': 800,
#  'mean_threshold': 0.98,
#  'min_threshold': 0.35000000000000003,
#  'trivial_threshold': 0.02,
#  'enforce_min': True}
