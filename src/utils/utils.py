import logging
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pandera.pandas as pa
import yaml
from dotenv import load_dotenv

from src.utils.schemas import schemas

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


def read_config() -> dict[str, Any]:
    """Read the YAML configuration file and return its contents.

    Returns:
        dict[str, Any]: Configuration settings loaded from ``cfg/config.yaml``.
    """
    config_file_path = Path("cfg") / "config.yaml"
    with open(config_file_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return cast(dict[str, Any], config_data)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names: uppercase, replace spaces/special chars with underscores.

    Examples:
        ``sepal length (cm)`` → ``SEPAL_LENGTH_CM``
        ``target``            → ``TARGET``
    """
    df.columns = pd.Index(
        [re.sub(r"[^A-Z0-9]+", "_", col.upper()).strip("_") for col in df.columns]
    )
    return df


def read_data(
    file_name: str | None = None,
    schema_obj: str | None = None,
) -> pd.DataFrame:
    """Read data from a CSV file in the data directory.

    Args:
        file_name: CSV file name (with or without .csv extension).
        schema_obj: Optional Pandera schema key for validation.

    Returns:
        pandas.DataFrame: Loaded data with normalised column names.

    Raises:
        ValueError: If file_name is not provided.
        FileNotFoundError: If the CSV file does not exist.
    """
    if file_name is None:
        raise ValueError("file_name must be provided.")

    if not file_name.endswith(".csv"):
        file_name = f"{file_name}.csv"

    file_path = DATA_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    df = _normalise_columns(df)

    logger.info("Data read from %s. Shape: %s", file_path, df.shape)

    if schema_obj:
        try:
            schemas[schema_obj].validate(df)
            logger.info("Data schema validation passed.")
        except pa.errors.SchemaError as exc:
            logger.error("Data schema validation failed: %s", exc)
            raise

    return df


def write_data(
    df: pd.DataFrame,
    file_name: str,
    schema_obj: str | None = None,
) -> None:
    """Write data to a CSV file in the data directory.

    Args:
        df: DataFrame to write.
        file_name: Output CSV file name (with or without .csv extension).
        schema_obj: Optional Pandera schema key for validation before writing.

    Raises:
        ValueError: If DataFrame is empty.
    """
    if df.empty:
        logger.info("DataFrame is empty; skipping write.")
        return

    if schema_obj:
        try:
            schemas[schema_obj].validate(df)
            logger.info("Data schema validation passed.")
        except pa.errors.SchemaError as exc:
            logger.error("Data schema validation failed: %s", exc)
            raise

    if not file_name.endswith(".csv"):
        file_name = f"{file_name}.csv"

    file_path = DATA_DIR / file_name
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(file_path, index=False)

    logger.info(
        "Data written to %s. Rows: %s",
        file_path,
        len(df),
    )
