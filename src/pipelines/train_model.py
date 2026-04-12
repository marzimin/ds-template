import importlib
import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.pipelines.pipeline import Pipeline
from src.utils.utils import read_config, read_data, write_data

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainModelPipeline(Pipeline):
    """End-to-end training pipeline."""

    def __init__(self, run_name: Optional[str] = None) -> None:
        """Initialise configuration and placeholders.

        Args:
            run_name: Optional MLflow run name. If not provided, a default is used.

        Raises:
            ValueError: If the configuration file cannot be loaded.
        """
        logger.info("Initializing the TrainModelPipeline.")
        self.config = read_config()
        if not self.config:
            raise ValueError("Configuration file is empty or not found.")

        self.model_name = str(self.config.get("model_name", "xgboost")).lower()
        raw_params = self.config.get("model_params", {}) or {}
        if (
            isinstance(raw_params, dict)
            and self.model_name in raw_params
            and isinstance(raw_params[self.model_name], dict)
        ):
            self.model_params = raw_params[self.model_name]
        else:
            self.model_params = raw_params
        self.run_name = run_name or "Default_Run_Name"
        self.model: Optional[BaseEstimator] = None

    def run(self) -> None:
        """Execute training and log metrics, model, and plots to MLflow."""
        logger.info("Starting the training pipeline.")

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

        input_file = self.config["data"]["input_file"]
        target_col = str(self.config.get("target_column", "TARGET"))

        df = read_data(
            file_name=input_file, suffix="prepared", schema_obj="prepared_data"
        )
        X = df.drop(columns=[target_col])
        y = df[target_col]

        test_size = float(self.config.get("test_size", 0.2))
        random_state = int(self.config.get("random_state", 42))
        stratify = y if bool(self.config.get("stratify", True)) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        self.train(X_train, y_train)
        train_accuracy = self.evaluate(X_train, y_train)
        test_accuracy = self.evaluate(X_test, y_test)

        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_param("model_name", self.model_name)
            if self.model_params:
                mlflow.log_params(self.model_params)
            mlflow.log_metric("train_accuracy", float(train_accuracy))
            logger.info(f"MODEL DRIFT: Train Accuracy = {train_accuracy:.4f}")
            mlflow.log_metric("test_accuracy", float(test_accuracy))
            logger.info(f"MODEL DRIFT: Test Accuracy = {test_accuracy:.4f}")

            self._log_confusion_matrix(X_test, y_test)
            self._log_classification_report(X_test, y_test)
            self._log_roc_curve(X_test, y_test)
            self._log_pr_curve(features=X_test, target=y_test)

            if self.model is not None:
                mlflow.sklearn.log_model(self.model, artifact_path=self.model_name)

            if self.model is not None:
                df_out = df.copy()
                df_out["PREDICTION"] = self.model.predict(X)
                write_data(
                    df_out,
                    file_name=input_file,
                    suffix="trained",
                    schema_obj="output_data",
                )

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Fit the configured classifier.

        Args:
            features: Feature matrix (no target column).
            target: Target series aligned with `features`.
        """
        logger.info("Training the model.")
        self.model = self._build_model()
        self.model.fit(features, target)

    def evaluate(self, features: pd.DataFrame, target: pd.Series) -> float:
        """Compute accuracy on the provided data.

        Args:
            features: Feature matrix for evaluation.
            target: True labels for evaluation.

        Returns:
            Accuracy as a float in [0, 1].

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        logger.info("Evaluating the model.")
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        y_pred = self.model.predict(features)
        return float(accuracy_score(target, y_pred))

    def _build_model(self) -> BaseEstimator:
        """Construct the model based on configuration.

        The model registry is read from ``cfg/config.yaml`` under the
        ``model_registry`` key. Each entry maps a name to a fully qualified
        class path, e.g. ``"sklearn.ensemble.RandomForestClassifier"``.

        Returns:
            An untrained sklearn-compatible estimator.

        Raises:
            ValueError: If model_name is absent from the registry or the
                class path cannot be imported.
        """
        registry: dict[str, str] = self.config.get("model_registry", {})
        class_path = registry.get(self.model_name)
        if class_path is None:
            raise ValueError(
                f"Unsupported model_name '{self.model_name}'. "
                f"Available: {list(registry.keys())}"
            )
        module_name, class_name = class_path.rsplit(".", 1)
        model_cls: type[BaseEstimator] = getattr(
            importlib.import_module(module_name), class_name
        )
        return model_cls(**self.model_params)

    # Model metrics

    def _log_confusion_matrix(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Log a confusion matrix as a .png to MLflow."""
        logger.info("Logging the confusion matrix.")
        if self.model is None:
            logger.warning("Model not available; skipping confusion matrix logging.")
            return

        y_pred = self.model.predict(features)
        cm = confusion_matrix(target, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        plots_dir = Path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        cm_path = plots_dir / "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(cm_path), artifact_path=cm_path.parent.name)
        logger.info("Confusion matrix saved locally to %s and logged to MLflow.")

    def _log_classification_report(
        self, features: pd.DataFrame, target: pd.Series
    ) -> None:
        """Log a classification report as a text artifact to MLflow."""
        logger.info("Logging the classification report.")
        if self.model is None:
            logger.warning(
                "Model not available; skipping classification report logging."
            )
            return

        y_pred = self.model.predict(features)
        report = classification_report(target, y_pred)

        reports_dir = Path(os.getenv("LOCAL_REPORTS_PATH", "outputs/reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(str(report_path), artifact_path=report_path.parent.name)
        logger.info(
            "Classification report saved locally to %s and logged to MLflow.",
            report_path,
        )

    def _log_roc_curve(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Log a ROC curve plot as an artifact to MLflow (binary targets only)."""
        logger.info("Logging the ROC curve.")
        if self.model is None:
            logger.warning("Model not available; skipping ROC logging.")
            return

        if hasattr(self.model, "predict_proba"):
            y_pred_prob = pd.Series(self.model.predict_proba(features)[:, 1])
        elif hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(features)
            if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1:
                y_pred_prob = pd.Series(raw_scores[:, 1])
            else:
                y_pred_prob = pd.Series(raw_scores)
        else:
            logger.warning("Model does not expose predict_proba or decision_function.")
            return

        fpr, tpr, _ = roc_curve(target, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()

        plots_dir = Path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        roc_path = plots_dir / "roc_curve.png"
        plt.savefig(roc_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(roc_path), artifact_path=roc_path.parent.name)
        logger.info("ROC curve saved locally to %s and logged to MLflow.", roc_path)

    def _log_pr_curve(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Log a Precision Recall Curve an an artifact to MLflow (binary targets only)."""
        logger.info(msg="Logging the Precision Recall Curve.")
        if self.model is None:
            logger.warning(msg="Model not available; skipping PRC logging.")
            return

        if hasattr(self.model, "predict_proba"):
            y_pred_prob = pd.Series(self.model.predict_proba(features)[:, 1])
        elif hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(features)
            if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1:
                y_pred_prob = pd.Series(raw_scores[:, 1])
            else:
                y_pred_prob = pd.Series(raw_scores)
        else:
            logger.warning("Model does not expose predict_proba or decision_function.")
            return

        precision, recall, _ = precision_recall_curve(target, y_pred_prob)
        prc_auc = auc(recall, precision)

        prevalence = float(target.mean())

        plt.figure()
        plt.plot(
            recall, precision, label=f"Precision Recall Curve (area = {prc_auc:.2f})"
        )
        plt.axhline(
            y=prevalence,
            linestyle="--",
            color="grey",
            label=f"No skill (prevalence = {prevalence:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")
        plt.legend()
        plt.grid()

        plots_dir = Path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        pr_curve_path = plots_dir / "pr_curve.png"
        plt.savefig(pr_curve_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(pr_curve_path), artifact_path=pr_curve_path.parent.name)
        logger.info(
            "Precision Recall curve saved locally to %s and logged to MLflow.",
            pr_curve_path,
        )
