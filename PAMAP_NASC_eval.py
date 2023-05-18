from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from wavetools import lowpass_filtering
from PAMAP_commons import *
from NASC import maxNASC

hv.extension("bokeh")
hv.opts.defaults(hv.opts.Curve(width=800, height=250))

fs = 50


def random_drop(X, label, mask, ratio, p=None):
    indices = np.random.choice(
        np.arange(1, len(X)), int(len(X) * (1 - ratio)), replace=False, p=p
    )
    indices = np.sort(indices)
    return X[indices], label[indices], mask[indices]


if __name__ == "main":
    data_folder = "../external/PAMAP2_Dataset/Protocol"
    acc_features = ["acc_x", "acc_y", "acc_z"]

    output_file = Path("results/pamap_nasc_100_40_100_024_04_500.csv")
    random_modification_ratio = 0

    win_len = int(85 * (1 - random_modification_ratio))
    w, W = int(30 * (1 - random_modification_ratio)), int(
        90 * (1 - random_modification_ratio)
    )
    std_w = 50
    std_threshold = 0.18
    ac_threshold = 0.5
    smoothing_w = 1000

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
    # print(output.std())
