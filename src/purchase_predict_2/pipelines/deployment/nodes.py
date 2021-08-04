import os
import mlflow

from mlflow.tracking import MlflowClient


def push_model_to_registry(registry_name: str, run_id: int):
    """
    Pushes a model's version to the specified registry
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))
    result = mlflow.register_model(
        "runs:/{}/articafts/model".format(run_id), registry_name
    )

    return result.version


def stage_model(registry_name: str, version: int):
    """
    Stages a model version pushed to model registry
    """
    env = os.getenv("ENV")
    if env not in ["production", "staging"]:
        return

    client = MlflowClient()
    client.transition_model_version_stage(
        name=registry_name, version=int(version), stage=env[0].upper() + env[1:]
    )
