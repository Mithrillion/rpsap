import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from PAMAP_commons import *
from NASC import maxNASC
import optuna

fs = 50


def random_drop(X, label, mask, ratio, p=None):
    indices = np.random.choice(
        np.arange(1, len(X)), int(len(X) * (1 - ratio)), replace=False, p=p
    )
    indices = np.sort(indices)
    return X[indices], label[indices], mask[indices]


data_folder = "../external/PAMAP2_Dataset/Protocol"
acc_features = ["acc_x", "acc_y", "acc_z"]

random_modification_ratio = 0

# win_len = int(100 * (1 - random_modification_ratio))
# w, W = int(40 * (1 - random_modification_ratio)), int(
#     100 * (1 - random_modification_ratio)
# )
# std_w = 40
# std_threshold = 0.24
# ac_threshold = 0.4
# smoothing_w = 500


def NASC_eval(
    data,
    random_modification_ratio,
    win_len,
    w,
    W,
    std_w,
    std_threshold,
    ac_threshold,
    smoothing_w,
):
    this_mat = data[acc_features].values
    rep_annot = data["ilabel"].values
    mask = data.activityID == 0

    if random_modification_ratio > 0:
        np.random.seed(7777)
        # NOTE: set uneven warping
        p = 2 + np.sin(np.arange(len(this_mat) - 1) / 500)
        p /= np.sum(p)
        this_mat, rep_annot, mask = random_drop(
            this_mat, rep_annot, mask, random_modification_ratio, p
        )
    # this_mat = lowpass_filtering(this_mat, 3, fs, 5)

    mag = np.sqrt(np.sum(this_mat**2, -1))
    ac_trace = maxNASC(mag, win_len, (w, W))
    std_trace = pd.Series(mag).rolling(std_w, 1, True).std().values
    label = ((std_trace > std_threshold) & (ac_trace > ac_threshold)).astype(int)
    detection = (pd.Series(label).rolling(smoothing_w, 1, True).mean() > 0.5).values

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


results_df = pd.DataFrame()
full_data = [load_file(i) for i in range(9)]


def objective(trial):
    win_len = trial.suggest_int("win_len", 80, 120, step=5)
    w = trial.suggest_int("w", 30, 50, step=5)
    W = trial.suggest_int("W", 90, 110, step=5)
    std_w = trial.suggest_int("std_w", 30, 50, step=5)
    std_threshold = trial.suggest_float("std_threshold", 0.1, 0.3, step=0.02)
    ac_threshold = trial.suggest_float("ac_threshold", 0.1, 0.9, step=0.1)
    smoothing_w = trial.suggest_int("smoothing_w", 200, 1000, step=100)

    f1_list = []
    for data in full_data:
        f1 = NASC_eval(
            data,
            random_modification_ratio,
            win_len,
            w,
            W,
            std_w,
            std_threshold,
            ac_threshold,
            smoothing_w,
        )
        f1_list += [f1]

    return -np.mean(f1_list)


sampler = optuna.samplers.TPESampler(seed=7777)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

best_f1 = objective(study.best_trial)