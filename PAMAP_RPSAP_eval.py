from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sigtools.transforms import *
from sigtools.signed_areas import *
from wavetools import lowpass_filtering
from PAMAP_commons import *

hv.extension("bokeh")
hv.opts.defaults(hv.opts.Curve(width=800, height=250))

fs = 50


def random_drop(X, label, mask, ratio, p=None):
    indices = np.random.choice(
        np.arange(1, len(X)), int(len(X) * (1 - ratio)), replace=False, p=p
    )
    indices = np.sort(indices)
    return X[indices], label[indices], mask[indices]


data_folder = "../external/PAMAP2_Dataset/Protocol"
acc_features = ["acc_x", "acc_y", "acc_z"]

output_file = Path("results/pamap_rp_sa_single_10_10_td_6_20_099.csv")
random_modification_ratio = 0

w, W = int(50 * (1 - random_modification_ratio)), int(
    1000 * (1 - random_modification_ratio)
)
gap = int(10 * (1 - random_modification_ratio))
k = 10

rows = []
# if not output_file.is_file():
#     header = pd.DataFrame(
#         columns=[
#             "subject_id",
#             "acc",
#             "prec",
#             "rec",
#             "f1",
#         ]
#     )
#     header.to_csv(output_file.absolute(), index=False)

for i in tqdm(range(9)):
    data = load_file(i)
    this_mat = data[acc_features].values
    rep_annot = data["ilabel"].values
    mask = data.activityID == 0

    if random_modification_ratio > 0:
        np.random.seed(7777)
        this_mat, rep_annot, mask = random_drop(
            this_mat, rep_annot, mask, random_modification_ratio
        )
    this_mat = lowpass_filtering(this_mat, 3, fs, 5)

    with torch.no_grad():
        td_data, pca = to_td_pca(torch.tensor(this_mat.copy())[None, ...], 8, 1, 20)
        # td_data = torch.tensor(this_mat.copy())[None, ...].float()
        consec_cossim, D_state, I_state, SAs = recurrence_persistence(
            td_data[0, ...], (w, W), k, "ed", gap, True
        )
    res = find_labels_from_cossim(consec_cossim, w // 2, W, 0.99, 0)
    detection = np.zeros_like(rep_annot)
    detection[-len(res) :] = res
    is_trivial = get_trivial_mask(this_mat, W, 0.05)
    detection[is_trivial[: len(detection)]] = 0

    # evaluating - naive method
    detection = np.array(detection)
    target = rep_annot[: len(detection)]
    naive_acc = accuracy_score(target, detection)
    naive_prec = precision_score(target, detection)
    naive_rec = recall_score(target, detection)
    naive_f1 = f1_score(target, detection)
    # naive_auc = roc_auc_score(target, )

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
                i,
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
    rows += [row]
output = pd.concat(rows)
print(output)
print(output.mean())
print(output.std())
