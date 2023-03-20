from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from external.STEM import SSPDetector as ssp  # ver. 2
from Recofit_commons import *


def random_drop(X, label, mask, ratio, p=None):
    indices = np.random.choice(
        np.arange(1, len(X)), int(len(X) * (1 - ratio)), replace=False, p=p
    )
    indices = np.sort(indices)
    return X[indices], label[indices], mask[indices]


omega = 0.675
random_modification_ratio = 0.05
l = int(100 * (1 - random_modification_ratio))
output_file = Path(
    f"results/recofit_ssp_v2_omega_{omega}_l{l}_rd{random_modification_ratio}.csv"
)
m = 5 * l
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


def process_subject(subject_id):
    rep_annot = generate_annotation(mats, all_activities_df, subject_id).astype(bool)
    mask = generate_mask(mats, all_activities_df, subject_id).astype(bool)
    this_mat = mats[subject_id][:, 1:]
    if random_modification_ratio > 0:
        np.random.seed(7777)
        this_mat, rep_annot, mask = random_drop(
            this_mat, rep_annot, mask, random_modification_ratio
        )
    # this_mat = lowpass_filtering(this_mat, 3, fs, 5)

    # applying SIMPAD
    detection_table = ssp.SIMPAD(mats[subject_id][:, 1:].T, l, m, omega=omega)
    N = len(this_mat)
    detection = np.zeros(N, dtype=int)
    for i, (s, e, _) in detection_table.iterrows():
        detection[s:e] = 1

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
    # row.to_csv(output_file.absolute(), index=False, mode="a", header=False)
    return row


for row in map(process_subject, np.arange(126)):
    row.to_csv(output_file.absolute(), index=False, mode="a", header=False)
