from kedro.pipeline import Pipeline, node

from purchase_predict_2.pipelines.deployment.nodes import (
    push_model_to_registry,
    stage_model,
)


def create_pipeline(*kwargs):
    return Pipeline(
        [
            node(
                push_model_to_registry,
                ["params:mlflow_model_registry", "mlflow_run_id"],
                "mlflow_model_version",
            ),
            node(
                stage_model,
                ["params:mlflow_model_registry", "mlflow_model_version"],
                None,
            ),
        ]
    )
