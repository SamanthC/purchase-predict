import pytest

from purchase_predict_2.pipelines.loading.nodes import load_csv_from_bucket
from purchase_predict_2.pipelines.processing.nodes import encode_feature

from kedro.io import DataCatalog, MemoryDataSet


@pytest.fixture(scope="module")
def project_id():
    return "blent-mleng"


@pytest.fixture(scope="module")
def primary_folder():
    return "ml-eng-blent/primary/data-test.csv"


@pytest.fixture(scope="module")
def dataset_not_encoded(project_id, primary_folder):
    return load_csv_from_bucket(project_id, primary_folder)


@pytest.fixture(scope="module")
def test_ratio():
    return 0.3


@pytest.fixture(scope="module")
def dataset_encoded(dataset_not_encoded):
    return encode_feature(dataset_not_encoded)["features"]


@pytest.fixture(scope="module")
def catalog_test(dataset_not_encoded, test_ratio):
    catalog = DataCatalog(
        {
            "primary": MemoryDataSet(dataset_not_encoded),
            "params:test_ratio": MemoryDataSet(test_ratio),
        }
    )

    return catalog
