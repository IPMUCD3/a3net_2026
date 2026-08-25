import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_data(
    feature_columns,
    train_path="data/train.csv",
    test_path="data/test.csv",
):
    train_and_valid = pd.read_csv(
        train_path,
        usecols=feature_columns + ["is_valid"],
    )

    train = (
        train_and_valid.loc[~train_and_valid["is_valid"]]
        .drop(columns="is_valid")
        .reset_index(drop=True)
    )

    valid = (
        train_and_valid.loc[train_and_valid["is_valid"]]
        .drop(columns="is_valid")
        .reset_index(drop=True)
    )

    test_columns = [
        column
        for column in feature_columns
        if column not in {"z_spec", "z_spec_err"}
    ]

    test = pd.read_csv(
        test_path,
        usecols=test_columns,
    )

    return train, valid, test


def print_quality_report(df):
    report = (
        df.describe(
            percentiles=[0.01, 0.50, 0.99]
        )
        .T
        .rename(
            columns={
                "1%": "p01",
                "50%": "median",
                "99%": "p99",
            }
        )
    )

    report.insert(
        0,
        "missing",
        df.isna().sum(),
    )
    report.insert(
        1,
        "unique",
        df.nunique(dropna=True),
    )

    report = report[
        [
            "count",
            "missing",
            "unique",
            "mean",
            "std",
            "min",
            "p01",
            "median",
            "p99",
            "max",
        ]
    ]

    print(
        report.to_string(
            float_format=lambda value: f"{value:.5g}"
        )
    )

    return None


def create_data_loader(
    df,
    feature_columns,
    batch_size,
    shuffle,
):
    features = torch.tensor(
        df[feature_columns].to_numpy(
            dtype=np.float32
        )
    )

    if "z_spec" in df.columns:
        targets = torch.tensor(
            df["z_spec"].to_numpy(
                dtype=np.float32
            )
        )
    else:
        targets = torch.full(
            (len(df),),
            float("nan"),
        )

    source_ids = torch.tensor(
        df["source_id"]
        .to_numpy(dtype=np.int64)
        .copy()
    )

    dataset = TensorDataset(
        features,
        targets,
        source_ids,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )