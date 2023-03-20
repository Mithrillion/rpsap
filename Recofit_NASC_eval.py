from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from NASC import maxNASC
from Recofit_commons import *


def random_drop(X, label, mask, ratio, p=None):
    indices = np.random.choice(
        np.arange(1, len(X)), int(len(X) * (1 - ratio)), replace=False, p=p
    )
    indices = np.sort(indices)
    return X[indices], label[indices], mask[indices]


random_modification_ratio = 0.5

params = {
    "win_len": 80,
    "w": 70,
    "W": 190,
    "std_w": 70,
    "std_threshold": 0.04,
    "ac_threshold": 0.4,
    "smoothing_w": 900,
}

win_len = int(80 * (1 - random_modification_ratio))
w, W = int(70 * (1 - random_modification_ratio)), int(
    190 * (1 - random_modification_ratio)
)
std_w = int(70 * (1 - random_modification_ratio))
std_threshold = 0.04
ac_threshold = 0.4
smoothing_w = int(900 * (1 - random_modification_ratio))

output_file = Path(
    f"results/recofit_nasc_{win_len}_{w}_{W}_{std_w}_{std_threshold}_{ac_threshold}_{smoothing_w}_rd{random_modification_ratio}.csv"
)

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "subject_id",
            "naive_acc",
            "naive_prec",
            "naive_rec",
            "naive_f1",
            "masked_acc",
            "masked_prec",
            "masked_rec",
            "masked_f1",
        ]
    )
    header.to_csv(output_file.absolute(), index=False)

# loading data
dat, fs, groupings_map = load_data()
mats, all_activities_df = prepare_data(dat, fs, groupings_map)

for subject_id in tqdm(range(len(mats))):
    rep_annot = generate_annotation(mats, all_activities_df, subject_id).astype(bool)
    mask = generate_mask(mats, all_activities_df, subject_id).astype(bool)
    this_mat = mats[subject_id][:, 1:]
    if random_modification_ratio > 0:
        np.random.seed(7777)
        this_mat, rep_annot, mask = random_drop(
            this_mat, rep_annot, mask, random_modification_ratio
        )
    # this_mat = lowpass_filtering(this_mat, 3, fs, 5)

    # applying NASC
    mag = np.sqrt(np.sum(this_mat**2, -1))
    ac_trace = maxNASC(mag, win_len, (w, W))
    std_trace = pd.Series(mag).rolling(std_w, 1, True).std().values
    label = ((std_trace > std_threshold) & (ac_trace > ac_threshold)).astype(int)
    detection = (pd.Series(label).rolling(smoothing_w, 1, True).mean() > 0.5).values

    # evaluating - naive method
    detection = np.array(detection)
    target = rep_annot[: len(detection)]
    naive_acc = accuracy_score(target, detection)
    naive_prec = precision_score(target, detection)
    naive_rec = recall_score(target, detection)
    naive_f1 = f1_score(target, detection)

    # evaluating - masking
    detection_masked = detection[~mask[: len(detection)]]
    target_masked = target[~mask[: len(detection)]]
    masked_acc = accuracy_score(target_masked, detection_masked)
    masked_prec = precision_score(target_masked, detection_masked)
    masked_rec = recall_score(target_masked, detection_masked)
    masked_f1 = f1_score(target_masked, detection_masked)

    row = pd.DataFrame(
        [
            [
                subject_id,
                naive_acc,
                naive_prec,
                naive_rec,
                naive_f1,
                masked_acc,
                masked_prec,
                masked_rec,
                masked_f1,
            ]
        ],
        columns=[
            "subject_id",
            "naive_acc",
            "naive_prec",
            "naive_rec",
            "naive_f1",
            "masked_acc",
            "masked_prec",
            "masked_rec",
            "masked_f1",
        ],
    )
    row.to_csv(output_file.absolute(), index=False, mode="a", header=False)
