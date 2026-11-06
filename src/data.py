"""Loading, splitting and preprocessing for the network intrusion dataset."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_data(path=config.TRAIN_CSV):
    return pd.read_csv(path)


def split_features_target(df):
    """Return features X and a binary target y (1 = anomaly, 0 = normal)."""
    X = df.drop(columns=[config.TARGET])
    y = (df[config.TARGET] != "normal").astype(int)
    return X, y


def train_test_split_data(X, y):
    """80/20 split with the notebook's seed, so the test set matches the presentation."""
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)


def build_preprocessor(scale=True):
    """One-hot encode protocol_type/service/flag, then optionally standardise.

    This is the notebook's ``pd.get_dummies`` -> ``StandardScaler`` step packaged
    as a fitted scikit-learn object, so the same transform can be applied to new
    traffic. Categories never seen in training (e.g. ``tftp_u`` in
    Test_data.csv) are encoded as all zeros instead of raising an error.

    Tree models get ``scale=False``: they do not need scaling, and unscaled
    thresholds (``src_bytes <= 28.5``) are readable in the tree plot.
    """
    encode = ColumnTransformer(
        [(
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            config.CATEGORICAL_COLS,
        )],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    steps = [("encode", encode)]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)
