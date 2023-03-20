import numpy as np
import pandas as pd
from scipy.io import loadmat
from functools import reduce
import operator


def ex_pd_wrapper(mat, name="value"):
    return pd.melt(
        pd.DataFrame(
            {"time": mat[:, 0], "X": mat[:, 1], "Y": mat[:, 2], "Z": mat[:, 3]}
        ),
        ["time"],
        ["X", "Y", "Z"],
        "axis",
        name,
    )


def activity_annotation_extractor(mat):
    activity_list = []
    for i in range(len(mat)):
        name, start, end, notes, repeats, _, _ = mat[i, :]
        if name == "Non-Exercise" or name == "<Initial Activity>":
            continue
        annot_text = f"{name}" + ("" if repeats == -1 else f" x{repeats}")
        activity_list += [(name, start, end, repeats, annot_text)]
    return pd.DataFrame(
        activity_list, columns=["name", "start", "end", "repeats", "annot"]
    )


def combine_sample_subject_data(data):
    data_list = []
    for subject_data in data:
        if type(subject_data) == np.ndarray:
            data_list += [subject_data]
        else:
            data_list += [np.array([subject_data])]
    return np.concatenate(data_list)


def load_data():
    file_path = "external/ExRecog/exercise_data.50.0000_multionly.mat"
    dat = loadmat(file_path, simplify_cells=True)

    fs = dat["Fs"]

    groupings = dat["exerciseConstants"]["usefulActivityGroupings"]
    groupings_map = [{u: k for u in v} for k, v in groupings]
    groupings_map = reduce(operator.or_, groupings_map)

    return dat, fs, groupings_map


exercise_list = [
    "Repetitive exercise",
    "Repetitive non-arm exercise",
    # "Walk",
    "Machines",
    # "Walk and run",
    "Opposite-hand repetitive exercise",
    # "Legacy activity"
    # "Ambiguous exercise",
]

mask_list = ["Walk", "Walk and run", "Ambiguous exercise", "Ambiguous labeling"]


def generate_annotation(mats, df, id):
    length = mats[id].shape[0]
    annot_vec = np.zeros(length)
    # this_activities = df.query(
    #     "grouping in @exercise_list and id == @id"
    # )
    this_activities = df.query(
        "(grouping in @exercise_list or repeats > 1) and id == @id"
    )
    for _, row in this_activities.iterrows():
        annot_vec[row["start_frame"] : row["end_frame"]] = 1
    return annot_vec


def generate_mask(mats, df, id):
    length = mats[id].shape[0]
    annot_vec = np.zeros(length)
    # this_activities = df.query(
    #     "grouping in @exercise_list and id == @id"
    # )
    this_activities = df.query("grouping in @mask_list and id == @id")
    for _, row in this_activities.iterrows():
        annot_vec[row["start_frame"] : row["end_frame"]] = 1
    return annot_vec


def prepare_data(dat, fs, groupings_map):
    sample_subject_data = combine_sample_subject_data(dat["subject_data"])

    mats = [
        sample_subject_data[i].data.accelDataMatrix
        for i in range(len(sample_subject_data))
    ]

    all_activities = []
    for n in range(len(sample_subject_data)):
        activity_sample = sample_subject_data[n].activityStartMatrix
        activities = activity_annotation_extractor(activity_sample)
        activities["grouping"] = activities["name"].apply(lambda x: groupings_map[x])
        activities["id"] = n
        all_activities += [activities]
    all_activities_df = pd.concat(all_activities)
    all_activities_df["start_frame"] = (all_activities_df["start"] // (1 / fs)).astype(
        int
    )
    all_activities_df["end_frame"] = (all_activities_df["end"] // (1 / fs)).astype(int)

    return mats, all_activities_df


def evaluation_scores():
    pass
