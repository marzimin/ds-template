"""Model training, evaluation, and MLflow logging."""

import importlib
import logging
import os
from typing import Optional, cast

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.config import read_config, resolve_project_path, target_column
from src.ml.io import read_data, write_data
from src.ml.pipeline import Pipeline
from src.ml.tracking import (
    active_or_new_run,
    build_signature,
    log_model,
    setup_mlflow,
)
from src.schemas import normalise_column_name

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
        self.class_labels: list[object] = []
        # Import path from model_registry; set by _build_model and used to pick
        # the matching MLflow flavor when logging.
        self.class_path: str = ""

    def run(self) -> None:
        """Execute training and log metrics, model, and plots to MLflow."""
        logger.info("Starting the training pipeline.")

        setup_mlflow()

        input_file = self.config["data"]["input_file"]
        target_col = normalise_column_name(target_column(self.config))

        df = read_data(
            file_name=input_file, suffix="prepared", schema_obj="prepared_data"
        )
        X, y = self._validate_training_data(df, target_col)

        # Fallbacks match the values shipped in cfg/config.yaml, so deleting a
        # key does not silently change behaviour.
        test_size = float(self.config.get("test_size", 0.2))
        random_state = int(self.config.get("random_state", 42))
        stratify = y if bool(self.config.get("stratify", False)) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        self.train(X_train, y_train)
        train_accuracy, train_precision, train_recall, train_f1_score = self.evaluate(
            X_train, y_train
        )
        test_accuracy, test_precision, test_recall, test_f1_score = self.evaluate(
            X_test, y_test
        )

        with active_or_new_run(self.run_name):
            mlflow.log_param("model_name", self.model_name)
            if self.model_params:
                mlflow.log_params(self.model_params)

            # log metrics
            metrics = [
                ("accuracy", train_accuracy, test_accuracy),
                ("precision", train_precision, test_precision),
                ("recall", train_recall, test_recall),
                ("f1_score", train_f1_score, test_f1_score),
            ]
            for name, train_val, test_val in metrics:
                mlflow.log_metric(f"train_{name}", train_val)
                logger.info(f"MODEL DRIFT: Train {name.capitalize()} = {train_val:.4f}")
                mlflow.log_metric(f"test_{name}", test_val)
                logger.info(f"MODEL DRIFT: Test {name.capitalize()} = {test_val:.4f}")

            self._log_confusion_matrix(X_test, y_test)
            self._log_classification_report(X_test, y_test)
            self._log_roc_curve(X_test, y_test)
            self._log_pr_curve(features=X_test, target=y_test)

            if self.model is not None:
                # A signature is logged rather than left optional: it is what
                # lets consumers discover the feature names and types a
                # prediction request must supply.
                signature = build_signature(X_train, self.model.predict(X_train))
                model_uri = log_model(
                    self.model,
                    class_path=self.class_path,
                    config=self.config,
                    signature=signature,
                    input_example=X_train.head(5),
                )
                logger.info("Model logged to %s", model_uri)

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

    def evaluate(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[float, float, float, float]:
        """Compute accuracy, precision, recall, and f1-score on the provided data.

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
        positive_label = self.class_labels[-1]
        accuracy = float(accuracy_score(target, y_pred))
        precision = float(
            precision_score(target, y_pred, pos_label=positive_label, zero_division=0)
        )
        recall = float(
            recall_score(target, y_pred, pos_label=positive_label, zero_division=0)
        )
        f1 = float(f1_score(target, y_pred, pos_label=positive_label, zero_division=0))

        return accuracy, precision, recall, f1

    def _validate_training_data(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Validate template assumptions before sklearn raises opaque errors."""
        if target_col not in df.columns:
            raise KeyError(
                f"Configured target_column {target_col!r} was not found in prepared "
                f"data. Available columns: {list(df.columns)}. Column names are "
                "normalised to uppercase with underscores when CSVs are read."
            )

        features = df.drop(columns=[target_col])
        target = df[target_col]
        if features.empty:
            raise ValueError(
                "Training requires at least one feature column besides the target."
            )

        non_numeric_cols = features.select_dtypes(exclude="number").columns.tolist()
        if non_numeric_cols:
            raise ValueError(
                "Default training expects all feature columns to be numeric. "
                f"Encode, drop, or transform these columns in PrepareDataPipeline: "
                f"{non_numeric_cols}."
            )

        null_counts = features.isna().sum()
        null_feature_cols = null_counts[null_counts > 0].index.tolist()
        if null_feature_cols:
            raise ValueError(
                "Default training does not impute missing feature values. "
                "Handle nulls in PrepareDataPipeline before training. Columns with "
                f"nulls: {null_feature_cols}."
            )

        if target.isna().any():
            raise ValueError(
                "Default training requires a non-null target column. Drop or fill "
                "target nulls in PrepareDataPipeline before training."
            )

        observed_labels = target.drop_duplicates().tolist()
        configured_labels = list(self.config.get("target_values") or [])
        class_labels = configured_labels or observed_labels
        if len(observed_labels) != 2:
            raise ValueError(
                "Default metrics and plots are configured for binary "
                f"classification, but target_column {target_col!r} has "
                f"{len(observed_labels)} classes: {observed_labels}. For regression or "
                "multiclass projects, replace the metrics and schema assumptions in "
                "TrainModelPipeline."
            )
        if set(class_labels) != set(observed_labels):
            raise ValueError(
                f"Configured target_values {class_labels} do not match observed "
                f"target values {observed_labels}."
            )
        self.class_labels = class_labels

        return features, target

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
        self.class_path = class_path
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
        cm = confusion_matrix(target, y_pred, labels=self.class_labels)

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Predicted {label}" for label in self.class_labels],
            yticklabels=[f"Actual {label}" for label in self.class_labels],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        plots_dir = resolve_project_path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        cm_path = plots_dir / "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(cm_path), artifact_path=cm_path.parent.name)
        logger.info(
            "Confusion matrix saved locally to %s and logged to MLflow.", cm_path
        )

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
        report = classification_report(target, y_pred, zero_division=0)

        reports_dir = resolve_project_path(
            os.getenv("LOCAL_REPORTS_PATH", "outputs/reports")
        )
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
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
            y_pred_prob = self._positive_class_scores(features)
        elif hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(features)
            if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1:
                y_pred_prob = pd.Series(raw_scores[:, 1])
            else:
                y_pred_prob = pd.Series(raw_scores)
        else:
            logger.warning("Model does not expose predict_proba or decision_function.")
            return

        positive_label = self.class_labels[-1]
        fpr, tpr, _ = roc_curve(target, y_pred_prob, pos_label=positive_label)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()

        plots_dir = resolve_project_path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
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
            y_pred_prob = self._positive_class_scores(features)
        elif hasattr(self.model, "decision_function"):
            raw_scores = self.model.decision_function(features)
            if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1:
                y_pred_prob = pd.Series(raw_scores[:, 1])
            else:
                y_pred_prob = pd.Series(raw_scores)
        else:
            logger.warning("Model does not expose predict_proba or decision_function.")
            return

        positive_label = self.class_labels[-1]
        precision, recall, _ = precision_recall_curve(
            target, y_pred_prob, pos_label=positive_label
        )
        prc_auc = auc(recall, precision)

        prevalence = float((target == positive_label).mean())

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

        plots_dir = resolve_project_path(os.getenv("LOCAL_PLOTS_PATH", "outputs/plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)

        pr_curve_path = plots_dir / "pr_curve.png"
        plt.savefig(pr_curve_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(pr_curve_path), artifact_path=pr_curve_path.parent.name)
        logger.info(
            "Precision Recall curve saved locally to %s and logged to MLflow.",
            pr_curve_path,
        )

    def _positive_class_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return probability scores for the positive class label."""
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        proba = self.model.predict_proba(features)
        model_classes = cast(list[object], list(getattr(self.model, "classes_", [])))
        positive_label = self.class_labels[-1]
        if positive_label not in model_classes:
            raise ValueError(
                f"Positive class {positive_label!r} was not found in model classes "
                f"{model_classes}."
            )
        positive_index = model_classes.index(positive_label)
        return pd.Series(proba[:, positive_index])
