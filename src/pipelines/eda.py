import logging
import os
from contextlib import nullcontext
from pathlib import Path

import mlflow

from src.pipelines.pipeline import Pipeline
from src.utils.utils import (
    _detect_feature_types,
    _plot_bar_chart,
    _plot_class_distribution,
    _plot_correlation_heatmap,
    _plot_datetime_distribution,
    _plot_feature_vs_target,
    _plot_histogram,
    _plot_missing_values,
    read_config,
    read_data,
    setup_mlflow,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EDAPipeline(Pipeline):
    """Pipeline that loads prepared data and plots visualizations to log locally and into MLFlow."""

    def __init__(self) -> None:
        """Initialize configuration."""
        logger.info(msg="Initializing the EDAPipeline.")
        self.config = read_config()
        if not self.config:
            raise ValueError("Configuration file is empty or not found.")

    def run(self) -> None:
        """Execute the EDA workflow end to end."""
        logger.info(msg="Starting the EDA pipeline.")

        setup_mlflow()

        input_file = self.config["data"]["input_file"]
        data_dir = Path(self.config["data"]["dir"])
        stem = input_file.removesuffix(".csv")
        prepared_path = data_dir / f"{stem}_prepared.csv"

        if not prepared_path.exists():
            raise FileNotFoundError(
                f"Prepared dataset not found at '{prepared_path}'. "
                "Run PrepareDataPipeline before EDAPipeline."
            )

        df = read_data(
            file_name=input_file, suffix="prepared", schema_obj="prepared_data"
        )
        logger.info("Prepared dataset loaded. Shape: %s", df.shape)

        target_col = str(self.config.get("target_column", "TARGET"))
        feature_cols = [c for c in df.columns if c != target_col]

        eda_dir = Path(os.getenv("LOCAL_EDA_PATH", "outputs/eda"))
        eda_dir.mkdir(parents=True, exist_ok=True)

        col_info = _detect_feature_types(df[feature_cols])
        artifacts: list[Path] = []

        # Per-column distribution plots
        for col, info in col_info.items():
            if info["type"] == "numeric":
                path = _plot_histogram(df, col, info["null_pct"], eda_dir)
            elif info["type"] == "datetime":
                path = _plot_datetime_distribution(df, col, info["null_pct"], eda_dir)
            else:  # "categorical" or "boolean"
                path = _plot_bar_chart(
                    df, col, info["cardinality"], info["null_pct"], eda_dir
                )
            artifacts.append(path)
            logger.info("Saved distribution plot: %s", path.name)

        # Feature vs target plots
        for col, info in col_info.items():
            path = _plot_feature_vs_target(df, col, target_col, info["type"], eda_dir)
            artifacts.append(path)
            logger.info("Saved feature-vs-target plot: %s", path.name)

        # Dataset-level summary plots (target included for correlation insight)
        for path in [
            _plot_correlation_heatmap(df, eda_dir),
            _plot_class_distribution(df, target_col, eda_dir),
            _plot_missing_values(df, eda_dir),
        ]:
            artifacts.append(path)
            logger.info("Saved summary plot: %s", path.name)

        # Log all artifacts to MLflow; fall back to a standalone run if needed
        ctx = nullcontext() if mlflow.active_run() else mlflow.start_run(run_name="EDA")
        with ctx:
            for path in artifacts:
                mlflow.log_artifact(str(path), artifact_path=eda_dir.name)
            logger.info("Logged %d EDA artifacts to MLflow.", len(artifacts))
