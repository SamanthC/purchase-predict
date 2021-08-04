import pandas as pd
import numpy as np

from purchase_predict_2.pipelines.processing.nodes import encode_feature, split_dataset

MIN_SAMPLES = 5000
BALANCE_THRESHOLD = 0.1


def test_encode_features(dataset_not_encoded):
    df = encode_feature(dataset_not_encoded)["features"]
    assert df["purchased"].isin([0, 1]).all()

    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df.dtypes[col])

    assert df.shape[0] > MIN_SAMPLES

    assert (df["purchased"].value_counts() / df.shape[0] > BALANCE_THRESHOLD).all()

    print(df.head())


def test_split_dataset(dataset_encoded, test_ratio):
    X_train, y_train, X_test, y_test = split_dataset(
        dataset_encoded, test_ratio
    ).values()
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == dataset_encoded.shape[0]
    assert np.ceil(dataset_encoded.shape[0] * test_ratio) == X_test.shape[0]
