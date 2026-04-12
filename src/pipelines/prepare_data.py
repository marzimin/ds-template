import logging

import pandas as pd

from src.pipelines.pipeline import Pipeline
from src.utils.utils import read_data, write_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PrepareDataPipeline(Pipeline):
    """Pipeline that loads raw data, runs placeholder transforms, and persists results."""

    def __init__(self) -> None:
        """Initialise configuration."""
        logger.info("Initializing the PrepareDataPipeline.")
        # self.config = read_config() # config no longer used for data preparation stage

    def run(self) -> None:
        """Execute the data preparation workflow end to end."""
        logger.info("Starting the data preparation pipeline.")

        df = read_data(table_name="TEST_DS_TABLE_IRIS", schema_obj="input_data")
        logger.info("Input data read. Shape: %s", df.shape)

        df = df.pipe(self._func1).pipe(self._func2).pipe(self._func3).pipe(self._func4)

        write_data(
            df, table_name="TEST_DS_TABLE_IRIS_PREPARED", schema_obj="prepared_data"
        )
        logger.info("Processed data saved.")

    @staticmethod
    def _func1(df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder transform step 1.

        Multiply all numeric columns excluding target by 10 as an example transformation.

        Assumes final column in the DataFrame is the target column.

        """
        target_col = df.columns[-1]
        # exclude target column if exists from transformation
        numeric_cols = (
            df.drop(columns=[target_col]).select_dtypes(include="number").columns
        )
        df[numeric_cols] = df[numeric_cols] * 10
        return df

    @staticmethod
    def _func2(df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder transform step 2."""
        return df

    @staticmethod
    def _func3(df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder transform step 3."""
        return df

    @staticmethod
    def _func4(df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder transform step 4."""
        return df


if __name__ == "__main__":
    PrepareDataPipeline().run()
