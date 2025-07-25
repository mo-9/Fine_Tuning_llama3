import mlflow
import logging

class ExperimentTracker:
    def __init__(self, tracking_uri: str = "./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        self.logger = logging.getLogger(__name__)

    def start_run(self, run_name: str = None):
        self.run = mlflow.start_run(run_name=run_name)
        self.logger.info(f"MLflow run started: {self.run.info.run_id}")
        return self.run

    def end_run(self):
        mlflow.end_run()
        self.logger.info("MLflow run ended.")

    def log_param(self, key: str, value):
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float):
        mlflow.log_metric(key, value)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, artifact_path: str, signature=None, input_example=None):
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example
        )

    def set_tag(self, key: str, value: str):
        mlflow.set_tag(key, value)


