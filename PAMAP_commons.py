#%%
import os
import sys
import pandas as pd
import holoviews as hv
import numpy as np
from sklearn.decomposition import PCA

sys.path.append("external/mSIMPAD/")

data_folder = "../external/PAMAP2_Dataset/Protocol"
acc_features = ["acc_x", "acc_y", "acc_z"]
pc_features = ["pc_0", "pc_1"]

repeatActs = [4, 5, 6, 7, 12, 13, 24]
nonRepeatActs = [1, 2, 3, 8, 9, 10]


def safeDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0


def evalPerf(_detection, _label):
    _detection = np.array(_detection)
    _label = np.array(_label)

    if _detection.shape[0] != _label.shape[0]:
        print("Length not match between detection and label.")

    TP = np.sum(_detection & _label)
    TN = np.sum(~_detection & ~_label)
    FP = np.sum(_detection & ~_label)
    FN = np.sum(~_detection & _label)
    errRate = safeDiv(FP + FN, len(_detection)) * 100
    accuracy = safeDiv(TP + TN, len(_detection)) * 100
    precision = safeDiv(TP, TP + FP) * 100
    recall = safeDiv(TP, TP + FN) * 100
    F1 = 2 * safeDiv((precision * recall), (precision + recall))

    return [TP, TN, FP, FN, errRate, accuracy, precision, recall, F1]


files = sorted(os.listdir(data_folder))


def load_file(file_no, newSamplingRate=50, samplingRate=100):

    transRatio = newSamplingRate / samplingRate
    # Reading From Raw Data
    selected_file = files[file_no]
    data = pd.read_csv("%s/%s" % (data_folder, selected_file), sep="\s+", header=None)
    data = data.iloc[:, [0, 1, 38, 39, 40]]  # Using Ankle Accelerometers
    data.columns = ["time", "activityID"] + acc_features
    data.index = pd.to_datetime(data.time, unit="s")
    data = data.drop("time", axis=1)

    # Resample Data
    data = data.resample("%.7fS" % (1 / newSamplingRate)).mean()

    # Interpolate missing data
    data = data.interpolate()

    data.index = range(len(data))

    # Generate Labels
    data.activityID = data.activityID.astype(int)
    data["label"] = data.activityID.isin(repeatActs)
    data["ilabel"] = data["label"].astype(int)
    return data


def compute_pcs(data):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(data)
    data["pc_0"] = pcs[:, 0]
    data["pc_1"] = pcs[:, 1]
    return data


def preview_data(data, mode="acc"):
    if mode == "acc":
        preview = (
            hv.Curve(data, "index", "acc_x").opts(width=600, height=250)
            * hv.Curve(data, "index", "acc_y")
            * hv.Curve(data, "index", "acc_z")
        )
    elif mode == "pc":
        preview = hv.Curve(data, "index", "pc_0").opts(
            width=600, height=250
        ) * hv.Curve(data, "index", "pc_1")
    preview += hv.Curve(data, "index", "ilabel").opts(
        width=600, height=50, xaxis="bare", yaxis="bare"
    )
    return preview.cols(1)


def plot_results(
    data,
    detection,
    P,
    p_mean_profile,
    p_std_profile,
    D,
    kfn_labels,
    kfn_profile,
    merge_win=500,
    mode="acc",
):
    if mode == "acc":
        layout = (
            hv.Curve(data, "index", "acc_x").opts(width=800, height=250)
            * hv.Curve(data, "index", "acc_y")
            * hv.Curve(data, "index", "acc_z")
        )
    elif mode == "pc":
        layout = hv.Curve(data, "index", "pc_0").opts(width=800, height=250) * hv.Curve(
            data, "index", "pc_1"
        )
    layout += hv.Curve(data, "index", "ilabel").opts(
        width=800, height=50, xaxis="bare", yaxis="bare"
    )
    layout += hv.Curve(np.array(detection).astype(int), "index", "detected").opts(
        width=800, height=50, xaxis="bare", yaxis="bare"
    ) * hv.Curve(
        pd.Series(detection).rolling(merge_win, 1, True).mean().values > 0.5,
        "index",
        "detected",
    )
    layout += hv.Curve(kfn_labels, "index", "kfn-label").opts(
        width=800, height=50, xaxis="bare", yaxis="bare"
    ) * hv.Curve(
        pd.Series(kfn_labels).rolling(merge_win, 1, True).mean().values > 0.5,
        "index",
        "kfn-label",
    )
    layout += hv.Curve(P.numpy(), "index", "persistence").opts(
        width=800, height=100, xaxis="bare"
    ) * hv.Curve(p_mean_profile, "index", "persistence").opts(color="red", alpha=0.5)
    layout += hv.Curve(p_std_profile, "index", "pers. std.").opts(
        width=800, height=100, xaxis="bare"
    )
    layout += hv.Curve(D[:, 0].numpy(), "index", "deviation").opts(
        width=800, height=100, xaxis="bare"
    ) * hv.Curve(kfn_profile, "index", "deviation").opts(color="red", alpha=0.5)
    return layout
