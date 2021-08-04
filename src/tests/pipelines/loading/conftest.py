import pytest

from kedro.io import MemoryDataSet, DataCatalog


@pytest.fixture(scope="module")
def project_id():
    return "blent-mleng"


@pytest.fixture(scope="module")
def primary_folder():
    return "ml-eng-blent/primary/data-test.csv"


@pytest.fixture(scope="module")
def catalog_test(project_id, primary_folder):
    catalog = DataCatalog(
        {
            "params:gcp_project_id": MemoryDataSet(project_id),
            "params:gcs_primary_folder": MemoryDataSet(primary_folder),
        }
    )

    return catalog
