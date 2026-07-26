"""CSV reading and writing with Pandera schema validation.

Path resolution and YAML configuration live in :mod:`src.config`; plotting
helpers live in :mod:`src.ml.plots`. This module sits between them and stays
free of matplotlib and seaborn.
"""

import logging
from typing import Any, TypedDict, cast

import pandas as pd
from pandera.errors import SchemaError

from src.config import read_config, resolve_project_path, target_column
from src.schemas import build_schemas, normalise_column_name

logger = logging.getLogger(__name__)


class ColumnInfo(TypedDict):
    """Per-column metadata used to drive EDA plot selection."""

    type: str  # "numeric", "categorical", or "boolean"
    cardinality: int
    null_pct: float


def get_schema(schema_name: str, config: dict[str, Any]) -> Any:
    """Return a configured Pandera schema by name."""
    return build_schemas(
        target_column=target_column(config),
        target_values=cast(list[object] | None, config.get("target_values")),
    )[schema_name]


def _derive_filename(file_name: str, suffix: str) -> str:
    """Append a suffix to a file stem, preserving the .csv extension.

    Examples:
        ``_derive_filename("breast_cancer.csv", "prepared")`` → ``"breast_cancer_prepared.csv"``
        ``_derive_filename("breast_cancer.csv", "trained")``  → ``"breast_cancer_trained.csv"``
    """
    stem = file_name.removesuffix(".csv")
    return f"{stem}_{suffix}.csv"


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
            (e.g. ``"prepared"`` resolves ``breast_cancer.csv`` →
            ``breast_cancer_prepared.csv``).
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
    data_dir = resolve_project_path(
        config["data"]["raw_dir"] if raw else config["data"]["dir"]
    )

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
            get_schema(schema_obj, config).validate(df)
            logger.info("Data schema validation passed.")
        except SchemaError as exc:
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
            (e.g. ``"prepared"`` resolves ``breast_cancer.csv`` →
            ``breast_cancer_prepared.csv``).
        schema_obj: Optional Pandera schema key for validation before writing.
    """
    if df.empty:
        logger.info("DataFrame is empty; skipping write.")
        return

    config = read_config()

    if schema_obj:
        try:
            get_schema(schema_obj, config).validate(df)
            logger.info("Data schema validation passed.")
        except SchemaError as exc:
            logger.error("Data schema validation failed: %s", exc)
            raise

    if suffix:
        file_name = _derive_filename(file_name, suffix)
    elif not file_name.endswith(".csv"):
        file_name = f"{file_name}.csv"

    data_dir = resolve_project_path(config["data"]["dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / file_name
    df.to_csv(file_path, index=False)

    logger.info(
        "Data written to %s. Rows: %s",
        file_path,
        len(df),
    )


# Data prep utility functions


def _detect_feature_types(df: pd.DataFrame) -> dict[str, ColumnInfo]:
    """Infer the type, cardinality, and null rate for every column.

    Args:
        df: Feature-only DataFrame (target column excluded before calling).

    Returns:
        Mapping of column name to :class:`ColumnInfo`.
    """
    info: dict[str, ColumnInfo] = {}
    for col in df.columns:
        null_pct = float(df[col].isna().mean())
        cardinality = int(df[col].nunique(dropna=True))
        if pd.api.types.is_bool_dtype(df[col]):
            col_type = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        else:
            col_type = "categorical"
        info[col] = ColumnInfo(
            type=col_type, cardinality=cardinality, null_pct=null_pct
        )
    return info


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names: uppercase, replace spaces/special chars with underscores.

    Examples:
        ``sepal length (cm)`` → ``SEPAL_LENGTH_CM``
        ``target``            → ``TARGET``
    """
    df.columns = pd.Index([normalise_column_name(col) for col in df.columns])
    return df
