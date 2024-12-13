import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from VideoDataset import VideoDataset


def get_dataloader(dataset, batch_size=8, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle)


def load_dataset(folder_path_train, folder_path_test):
    df_train, df_val, df_test = load_dataframe(folder_path_train, folder_path_test)
    all_label_train = list(df_train["tag"])
    all_label_val = list(df_val["tag"])

    train_paths = [os.path.join(folder_path_train, i) for i in df_train["video_name"]]
    test_paths = [os.path.join(folder_path_test, i) for i in df_test["video_name"]]
    val_paths = [os.path.join(folder_path_train, i) for i in df_val["video_name"]]

    train_dataset = VideoDataset(train_paths, all_labels=all_label_train)
    val_dataset = VideoDataset(val_paths, all_labels=all_label_val)
    test_dataset = VideoDataset(
        test_paths, all_labels=list(df_test["tag"]), is_train=False
    )

    return train_dataset, val_dataset, test_dataset


def load_dataframe(folder_path_train, folder_path_test):
    label_encoder = LabelEncoder()

    train_df, test_df, encode_label = load_metadata(
        folder_path_train, folder_path_test, label_encoder
    )

    df_train = create_dataframe_metadata(train_df, 0, 90, label_encoder=label_encoder)
    df_val = create_dataframe_metadata(train_df, 90, 110, encode_label=encode_label)
    df_test = create_dataframe_metadata(test_df, 0, 50, encode_label=encode_label)

    return df_train, df_val, df_test


def load_metadata(folder_path_train, folder_path_test, label_encoder):
    train_df = pd.read_csv(folder_path_train[:-6] + "train.csv")
    test_df = pd.read_csv(folder_path_test[:-5] + "test.csv")

    df_train = create_dataframe_metadata(train_df, 0, 90, label_encoder=label_encoder)
    decode_label = dict()
    for index, row in df_train.iterrows():
        decode_label.update({row["tag"]: row["origin_tag"]})
    encode_label = {v: k for k, v in decode_label.items()}

    return train_df, test_df, encode_label


def create_dataframe_metadata(df, start, end, label_encoder=None, encode_label=None):
    df_punch = df[df["tag"] == "Punch"]
    df_playCello = df[df["tag"] == "PlayingCello"]
    df_CricketShot = df[df["tag"] == "CricketShot"]
    df_ShavingBeard = df[df["tag"] == "ShavingBeard"]
    df_TennisSwing = df[df["tag"] == "TennisSwing"]
    new_df = pd.concat(
        [
            df_punch[start:end],
            df_playCello[start:end],
            df_CricketShot[start:end],
            df_ShavingBeard[start:end],
            df_TennisSwing[start:end],
        ],
        ignore_index=True,
    )
    new_df["origin_tag"] = new_df["tag"]
    if label_encoder is not None:
        new_df["tag"] = label_encoder.fit_transform(new_df["tag"])
    else:
        new_df["tag"] = new_df["tag"].map(encode_label)
    return new_df
