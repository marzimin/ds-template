import logging

import pandas as pd

from src.config import read_config
from src.ml.io import read_data, write_data
from src.ml.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PrepareDataPipeline(Pipeline):
    """Pipeline that loads raw data, runs placeholder transforms, and persists results."""

    def __init__(self) -> None:
        """Initialize configuration."""
        logger.info("Initializing the PrepareDataPipeline.")
        self.config = read_config()
        if not self.config:
            raise ValueError("Configuration file is empty or not found.")

    def run(self) -> None:
        """Execute the data preparation workflow end to end."""
        logger.info("Starting the data preparation pipeline.")

        input_file = self.config["data"]["input_file"]

        df = read_data(file_name=input_file, raw=True, schema_obj="input_data")
        logger.info("Input data read. Shape: %s", df.shape)

        df = df.pipe(self._func1).pipe(self._func2).pipe(self._func3).pipe(self._func4)

        write_data(
            df, file_name=input_file, suffix="prepared", schema_obj="prepared_data"
        )
        logger.info("Processed data saved.")

    # data cleaning and feature engineering functions
    # all functions prior to train-test split can be written below
    @staticmethod
    def _func1(df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder transform step 1 — replace with your logic."""
        return df

    @staticmethod
    def _func2(df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder transform step 2 — replace with your logic."""
        return df

    @staticmethod
    def _func3(df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder transform step 3 — replace with your logic."""
        return df

    @staticmethod
    def _func4(df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder transform step 4 — replace with your logic."""
        return df


if __name__ == "__main__":
    PrepareDataPipeline().run()
