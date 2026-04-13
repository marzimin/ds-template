import logging
import os
import re
from pathlib import Path
from typing import Any, cast

import mlflow
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


def setup_mlflow() -> str:
    """Configure the MLflow tracking URI from environment variables.

    Idempotent: if the tracking URI is already set to the configured value,
    the URI is not reset.

    Returns:
        The active MLflow tracking URI.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    if mlflow.get_tracking_uri() != tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI set to: %s", tracking_uri)
    return tracking_uri


def read_config() -> dict[str, Any]:
    """Read the YAML configuration file and return its contents.

    Returns:
        dict[str, Any]: Configuration settings loaded from ``cfg/config.yaml``.
    """
    config_file_path = Path("cfg") / "config.yaml"
    with open(config_file_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return cast(dict[str, Any], config_data)


def _derive_filename(file_name: str, suffix: str) -> str:
    """Append a suffix to a file stem, preserving the .csv extension.

    Examples:
        ``_derive_filename("input_data.csv", "prepared")`` → ``"input_data_prepared.csv"``
        ``_derive_filename("input_data.csv", "trained")``    → ``"input_data_trained.csv"``
    """
    stem = file_name.removesuffix(".csv")
    return f"{stem}_{suffix}.csv"


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
    suffix: str | None = None,
    raw: bool = False,
    schema_obj: str | None = None,
) -> pd.DataFrame:
    """Read data from a CSV file in the configured data directory.

    Args:
        file_name: Base CSV file name (with or without .csv extension).
        suffix: Optional suffix to append to the file stem before reading
            (e.g. ``"prepared"`` resolves ``input_data.csv`` →
            ``input_data_prepared.csv``).
        raw: When ``True``, read from ``data.raw_dir``; otherwise from
            ``data.dir``.
        schema_obj: Optional Pandera schema key for validation.

    Returns:
        pandas.DataFrame: Loaded data with normalised column names.

    Raises:
        ValueError: If file_name is not provided.
        FileNotFoundError: If the CSV file does not exist.
    """
    if file_name is None:
        raise ValueError("file_name must be provided.")

    config = read_config()
    data_dir = Path(config["data"]["raw_dir"] if raw else config["data"]["dir"])

    if suffix:
        file_name = _derive_filename(file_name, suffix)
    elif not file_name.endswith(".csv"):
        file_name = f"{file_name}.csv"

    file_path = data_dir / file_name
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
    suffix: str | None = None,
    schema_obj: str | None = None,
) -> None:
    """Write data to a CSV file in the configured data directory.

    Args:
        df: DataFrame to write.
        file_name: Base CSV file name (with or without .csv extension).
        suffix: Optional suffix to append to the file stem before writing
            (e.g. ``"prepared"`` resolves ``input_data.csv`` →
            ``input_data_prepared.csv``).
        schema_obj: Optional Pandera schema key for validation before writing.
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

    if suffix:
        file_name = _derive_filename(file_name, suffix)
    elif not file_name.endswith(".csv"):
        file_name = f"{file_name}.csv"

    config = read_config()
    data_dir = Path(config["data"]["dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / file_name
    df.to_csv(file_path, index=False)

    logger.info(
        "Data written to %s. Rows: %s",
        file_path,
        len(df),
    )
