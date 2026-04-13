import argparse
import logging

import mlflow

from src.pipelines.prepare_data import PrepareDataPipeline
from src.pipelines.train_model import TrainModelPipeline
from src.utils.utils import setup_mlflow

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

    setup_mlflow()
    with mlflow.start_run(run_name=run_name):
        if args.prepare_data:
            PrepareDataPipeline().run()

        elif args.train_model:
            TrainModelPipeline(run_name=run_name).run()

        else:
            PrepareDataPipeline().run()
            TrainModelPipeline(run_name=run_name).run()


if __name__ == "__main__":
    main()
