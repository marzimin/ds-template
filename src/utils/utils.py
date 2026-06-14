import logging
import os
from pathlib import Path
from typing import Any, TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException
from pandera.errors import SchemaError

from src.utils.schemas import build_schemas
from src.utils.schemas import normalise_column_name as _normalise_schema_column_name

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


class ColumnInfo(TypedDict):
    """Per-column metadata used to drive EDA plot selection."""

    type: str  # "numeric", "categorical", or "boolean"
    cardinality: int
    null_pct: float


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative project paths from the repository root."""
    project_path = Path(path)
    if project_path.is_absolute():
        return project_path
    return PROJECT_ROOT / project_path


def normalise_column_name(column_name: str) -> str:
    """Normalise a single column name using the project schema convention."""
    return _normalise_schema_column_name(column_name)


def setup_mlflow() -> str:
    """Configure and verify the MLflow tracking server.

    The template intentionally defaults to a local MLflow server. Start it with
    ``mlflow server`` before running the CLI, or set ``MLFLOW_TRACKING_URI`` to
    a reachable tracking server.

    Returns:
        The active MLflow tracking URI.

    Raises:
        RuntimeError: If the configured MLflow server cannot be reached.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    if mlflow.get_tracking_uri() != tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI set to: %s", tracking_uri)
    _verify_mlflow_tracking_uri(tracking_uri)
    return tracking_uri


def _verify_mlflow_tracking_uri(tracking_uri: str) -> None:
    """Check that an HTTP(S) MLflow tracking server responds before running."""
    if not tracking_uri.startswith(("http://", "https://")):
        return

    health_url = tracking_uri.rstrip("/") + "/health"
    request = Request(health_url, method="GET")
    try:
        with urlopen(request, timeout=3):
            return
    except (HTTPError, URLError, TimeoutError, OSError, MlflowException) as exc:
        raise RuntimeError(
            "MLflow tracking server is required but is not reachable at "
            f"{tracking_uri!r}. Start it with `mlflow server --host 127.0.0.1 "
            "--port 5000`, or set MLFLOW_TRACKING_URI to a reachable server. "
            "The template defaults to a server-backed tracking URI so runs, "
            "metrics, models, and artifacts are captured consistently."
        ) from exc


def read_config() -> dict[str, Any]:
    """Read the YAML configuration file and return its contents.

    Returns:
        dict[str, Any]: Configuration settings loaded from ``cfg/config.yaml``.
    """
    config_file_path = resolve_project_path(Path("cfg") / "config.yaml")
    with open(config_file_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return cast(dict[str, Any], config_data)


def get_schema(schema_name: str, config: dict[str, Any]) -> Any:
    """Return a configured Pandera schema by name."""
    return build_schemas(
        target_column=str(config.get("target_column", "TARGET")),
        target_values=cast(list[object] | None, config.get("target_values")),
    )[schema_name]


def _derive_filename(file_name: str, suffix: str) -> str:
    """Append a suffix to a file stem, preserving the .csv extension.

    Examples:
        ``_derive_filename("input_data.csv", "prepared")`` → ``"input_data_prepared.csv"``
        ``_derive_filename("input_data.csv", "trained")``    → ``"input_data_trained.csv"``
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
            (e.g. ``"prepared"`` resolves ``input_data.csv`` →
            ``input_data_prepared.csv``).
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


# EDA plotting functions


def _datetime_period(series: pd.Series) -> tuple[pd.Categorical, str]:
    """Convert a datetime Series to an ordered Categorical of period labels.

    Returns a ``pd.Categorical`` (not plain strings) so that matplotlib treats
    the values as categorical units and does not attempt to parse them as floats
    or dates — which would trigger a ``UserWarning``.

    Granularity rules:
    - More than 2 years of range → yearly (``"Y"``)
    - More than 60 days → monthly (``"M"``)
    - 60 days or fewer → daily (``"D"``)

    Args:
        series: A non-empty ``datetime64`` Series with nulls already removed.

    Returns:
        Tuple of ``(ordered_categorical, period_label)`` where *period_label*
        is one of ``"Year"``, ``"Month"``, or ``"Day"``.
    """
    date_range = (series.max() - series.min()).days
    if date_range > 365 * 2:
        period_str = series.dt.to_period("Y").astype(str)
        label = "Year"
    elif date_range > 60:
        period_str = series.dt.to_period("M").astype(str)
        label = "Month"
    else:
        period_str = series.dt.to_period("D").astype(str)
        label = "Day"
    categories = sorted(period_str.dropna().unique())
    return pd.Categorical(period_str, categories=categories, ordered=True), label


def _plot_histogram(
    df: pd.DataFrame, col: str, null_pct: float, output_dir: Path
) -> Path:
    """Plot a histogram with KDE for a numeric column.

    Null values are dropped before plotting; the null rate is annotated on the
    chart when non-zero.

    Args:
        df: DataFrame containing the column.
        col: Column name to plot.
        null_pct: Fraction of null values in the column (0–1).
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[col].dropna(), ax=ax, kde=True, bins=30)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    if null_pct > 0:
        ax.annotate(
            f"Null: {null_pct:.1%}",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            color="crimson",
        )
    path = output_dir / f"hist_{col.lower()}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_bar_chart(
    df: pd.DataFrame, col: str, cardinality: int, null_pct: float, output_dir: Path
) -> Path:
    """Plot a bar chart of value counts for a categorical or boolean column.

    When cardinality exceeds 10 only the top 10 categories by frequency are
    shown. When null values are present they are added as a separate bar so
    they don't silently disappear from the chart.

    Args:
        df: DataFrame containing the column.
        col: Column name to plot.
        cardinality: Number of distinct non-null values in the column.
        null_pct: Fraction of null values in the column (0–1).
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    _TOP_N = 10
    counts = df[col].value_counts(dropna=True)
    truncated = cardinality > _TOP_N
    if truncated:
        counts = counts.head(_TOP_N)
    if null_pct > 0:
        counts["NULL"] = int(df[col].isna().sum())

    fig, ax = plt.subplots(figsize=(max(6, len(counts) + 1), 4))
    counts.plot(kind="bar", ax=ax)
    title = col + (f" — top {_TOP_N} of {cardinality}" if truncated else "")
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    path = output_dir / f"bar_{col.lower()}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_datetime_distribution(
    df: pd.DataFrame, col: str, null_pct: float, output_dir: Path
) -> Path:
    """Plot the temporal distribution of a datetime column.

    Granularity is chosen automatically based on the column's date range:
    year for > 2 years, month for > 60 days, day otherwise. Null values are
    dropped before plotting; the null rate is annotated when non-zero.

    Args:
        df: DataFrame containing the column.
        col: Datetime column name.
        null_pct: Fraction of null values in the column (0–1).
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    data = df[col].dropna()
    period_data, period_label = _datetime_period(data)
    counts = period_data.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(max(8, len(counts) // 2 + 2), 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"{col} — temporal distribution (by {period_label.lower()})")
    ax.set_xlabel(period_label)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    if null_pct > 0:
        ax.annotate(
            f"Null: {null_pct:.1%}",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            color="crimson",
        )
    path = output_dir / f"datetime_{col.lower()}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_feature_vs_target(
    df: pd.DataFrame, col: str, target_col: str, col_type: str, output_dir: Path
) -> Path:
    """Plot the relationship between a feature column and the target.

    - **Numeric**: two side-by-side panels — overlayed histograms per class
      and a boxplot grouped by class.
    - **Datetime**: countplot by auto-detected period (year/month/day) with
      target as hue, sorted chronologically.
    - **Categorical / boolean**: grouped countplot; capped at top 10 categories
      by frequency when cardinality is high.

    Rows where the feature is null are dropped before plotting.

    Args:
        df: DataFrame containing both the feature and target columns.
        col: Feature column name.
        target_col: Target column name.
        col_type: One of ``"numeric"``, ``"datetime"``, ``"categorical"``, or
            ``"boolean"``.
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    data = df[[col, target_col]].dropna(subset=[col])

    if col_type == "numeric":
        fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(13, 5))
        sns.histplot(
            data=data,
            x=col,
            hue=target_col,
            ax=ax_hist,
            kde=True,
            alpha=0.45,
            bins=25,
        )
        ax_hist.set_title(f"{col} — distribution by {target_col}")
        sns.boxplot(data=data, x=target_col, y=col, ax=ax_box)
        ax_box.set_title(f"{col} — boxplot by {target_col}")
    elif col_type == "datetime":
        period_col, period_label = _datetime_period(data[col])
        plot_data = data.assign(**{col: period_col})
        order = list(
            plot_data[col].cat.categories
        )  # already sorted by _datetime_period
        fig, ax = plt.subplots(figsize=(max(8, len(order) // 2 + 2), 5))
        sns.countplot(data=plot_data, x=col, hue=target_col, ax=ax, order=order)
        ax.set_title(f"{col} vs {target_col} — by {period_label.lower()}")
        ax.set_xlabel(period_label)
        ax.tick_params(axis="x", rotation=45)
    else:  # "categorical" or "boolean"
        _TOP_N = 10
        plot_data = data.copy()
        if data[col].nunique() > _TOP_N:
            top_vals = data[col].value_counts().head(_TOP_N).index
            plot_data = plot_data[plot_data[col].isin(top_vals)]
        fig, ax = plt.subplots(figsize=(max(6, plot_data[col].nunique() + 1), 5))
        sns.countplot(data=plot_data, x=col, hue=target_col, ax=ax)
        ax.set_title(f"{col} vs {target_col}")
        ax.tick_params(axis="x", rotation=45)

    path = output_dir / f"vs_target_{col.lower()}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot a correlation heatmap for all numeric columns.

    The target column may be included to surface feature-target correlations.

    Args:
        df: DataFrame (numeric columns are selected automatically).
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    corr = df.select_dtypes(include="number").corr()
    size = max(8, len(corr))
    fig, ax = plt.subplots(figsize=(size, size - 1))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_class_distribution(
    df: pd.DataFrame, target_col: str, output_dir: Path
) -> Path:
    """Plot the class distribution of the target column with percentage labels.

    Args:
        df: DataFrame containing the target column.
        target_col: Target column name.
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    counts = df[target_col].value_counts()
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Class Distribution — {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + total * 0.01, f"{v / total:.1%}", ha="center", fontsize=9)
    path = output_dir / "class_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_missing_values(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the null rate for every column as a bar chart.

    Always produced regardless of whether any nulls exist. Columns are sorted
    from highest to lowest null rate.

    Args:
        df: DataFrame to inspect.
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    null_pcts = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(null_pcts) // 2 + 2), 5))
    null_pcts.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Null rate")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="x", rotation=45)
    path = output_dir / "missing_values.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
