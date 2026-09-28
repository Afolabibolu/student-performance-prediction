"""Loading, splitting and the shared preprocessing pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_data(path=config.DATA_PATH):
    """Read the semicolon-separated dataset and return (X, y) with all 36 features."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]

    missing = set(config.ALL_FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise KeyError(f"Dataset is missing expected columns: {sorted(missing)}")

    return df[config.ALL_FEATURES], df[config.TARGET]


def split(X, y):
    """Stratified 80/20 train/test split, shared by both stages."""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )


def build_preprocessor(cols):
    """Scale the numeric columns and one-hot encode the categorical ones in `cols`."""
    numeric = [c for c in cols if c in config.NUMERIC_FEATURES]
    categorical = [c for c in cols if c in config.CATEGORICAL_FEATURES]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
