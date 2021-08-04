from purchase_predict_2.pipelines.loading.nodes import load_csv_from_bucket
import pandas as pd


def test_load_csv_from_bucket(project_id, primary_folder):
    df = load_csv_from_bucket(project_id, primary_folder)
    assert type(df) == pd.DataFrame
    assert df.shape[1] == 16
    assert "purchased" in df
