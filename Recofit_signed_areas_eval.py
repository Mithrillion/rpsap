import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# sys.path.append("../external/mSIMPAD/")
sys.path.append("..")
sys.path.append("../..")
# from SIMPAD import SSPDetector as ssp

from sigtools.transforms import *
from sigtools.signed_areas import *
from wavetools import lowpass_filtering
from Recofit_commons import *


def random_drop(X, label, mask, ratio, p=None):
    indices = np.random.choice(
        np.arange(1, len(X)), int(len(X) * (1 - ratio)), replace=False, p=p
    )
    indices = np.sort(indices)
    return X[indices], label[indices], mask[indices]


params = {
    "n_pcs": 6,
    "w": 50,
    "W": 1000,
    "k": 20,
    "gap": 33,
    "min_agg_win": 30,
    "agg_win": 900,
    "mean_threshold": 0.995,
    "min_threshold": 0.05,
    "trivial_threshold": 0.04,
    "enforce_min": True,
}

random_modification_ratio = 0.5
n_pcs = 6
d = 20
w, W = int(50 * (1 - random_modification_ratio)), int(
    1000 * (1 - random_modification_ratio)
)
k = 20
gap = 33
min_agg_win = 30
agg_win = int(900 * (1 - random_modification_ratio))
mean_threshold = 0.995
min_threshold = 0.05
trivial_threshold = 0.04
enforce_min = True
output_file = Path(
    f"results/recofit_rpsap_{n_pcs}_{d}_{w}_{W}_{k}_{gap}_{min_agg_win}_{agg_win}_{mean_threshold}_{min_threshold}_{trivial_threshold}_{enforce_min}_rd{random_modification_ratio}.csv"
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
