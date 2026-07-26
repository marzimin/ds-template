import argparse
import logging

import mlflow
from mlflow.exceptions import MlflowException

from src.ml.eda import EDAPipeline
from src.ml.prepare_data import PrepareDataPipeline
from src.ml.tracking import setup_mlflow
from src.ml.train_model import TrainModelPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running pipelines."""
    parser = argparse.ArgumentParser(description="Repo Template")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run the PrepareDataPipeline",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Run the EDAPipeline",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Run the TrainModelPipeline",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run name for MLflow",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for running one or both pipelines."""
    args = parse_args()

    run_name = args.run_name or "Default_Run_Name"
    selected_steps = [args.prepare_data, args.eda, args.train_model]
    if sum(selected_steps) > 1:
        raise ValueError(
            "Choose only one pipeline flag at a time. Use no flags to run "
            "prepare-data, EDA, and train-model sequentially."
        )

    try:
        tracking_uri = setup_mlflow()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    # Scope the handler to run creation only. Wrapping the pipeline bodies too
    # would relabel any MLflow error raised during a step — a model that fails
    # to serialise, an artifact that fails to upload — as a connection problem,
    # hiding the real message.
    try:
        active_run = mlflow.start_run(run_name=run_name)
    except MlflowException as exc:
        raise SystemExit(
            "Could not create an MLflow run at "
            f"{tracking_uri!r}. Start a compatible MLflow server with "
            "`mlflow server --host 127.0.0.1 --port 5000`, or set "
            "MLFLOW_TRACKING_URI to a reachable server."
        ) from exc

    with active_run:
        if args.prepare_data:
            PrepareDataPipeline().run()

        elif args.eda:
            EDAPipeline().run()

        elif args.train_model:
            TrainModelPipeline(run_name=run_name).run()

        else:
            PrepareDataPipeline().run()
            EDAPipeline().run()
            TrainModelPipeline(run_name=run_name).run()


if __name__ == "__main__":
    main()
