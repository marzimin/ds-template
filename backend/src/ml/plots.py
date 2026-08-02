"""Matplotlib/seaborn plotting helpers used by the EDA and training pipelines.

This is the only module that imports matplotlib and seaborn, so that anything
needing data access without the plotting stack — the API layer in particular —
stays cheap to import.

Every function here follows the same shape: draw, save to ``output_dir``, and
return the resulting :class:`~pathlib.Path`. Deciding what to do with that file
is the caller's job, which is what lets both pipelines collect their artifacts
and log them to MLflow in one place.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


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


# Training plots


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: Any,
    class_labels: Sequence[Any],
    output_dir: Path,
) -> Path:
    """Plot a confusion matrix for a classifier's predictions.

    Args:
        y_true: True labels.
        y_pred: Predicted labels, aligned with ``y_true``.
        class_labels: Label order for the axes.
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(class_labels))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Predicted {label}" for label in class_labels],
        yticklabels=[f"Actual {label}" for label in class_labels],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_roc_curve(
    y_true: pd.Series,
    scores: pd.Series,
    positive_label: Any,
    output_dir: Path,
) -> Path:
    """Plot a ROC curve for a binary classifier.

    Args:
        y_true: True labels.
        scores: Positive-class scores, aligned with ``y_true``.
        positive_label: Label treated as the positive class.
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=positive_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid()

    path = output_dir / "roc_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pr_curve(
    y_true: pd.Series,
    scores: pd.Series,
    positive_label: Any,
    output_dir: Path,
) -> Path:
    """Plot a precision-recall curve for a binary classifier.

    A no-skill baseline is drawn at the positive class prevalence, which is the
    precision a random classifier would achieve.

    Args:
        y_true: True labels.
        scores: Positive-class scores, aligned with ``y_true``.
        positive_label: Label treated as the positive class.
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    precision, recall, _ = precision_recall_curve(
        y_true, scores, pos_label=positive_label
    )
    prc_auc = auc(recall, precision)
    prevalence = float((y_true == positive_label).mean())

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"Precision Recall Curve (area = {prc_auc:.2f})")
    ax.axhline(
        y=prevalence,
        linestyle="--",
        color="grey",
        label=f"No skill (prevalence = {prevalence:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Recall Curve")
    ax.legend()
    ax.grid()

    path = output_dir / "pr_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# Regression plots


def _plot_predicted_vs_actual(y_true: pd.Series, y_pred: Any, output_dir: Path) -> Path:
    """Scatter predictions against truth, with a 45-degree reference line.

    The regression counterpart of a confusion matrix: points on the line are
    exact, and systematic departure from it reveals bias that a single summary
    metric hides.

    Args:
        y_true: True targets.
        y_pred: Predicted targets, aligned with ``y_true``.
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    actual = pd.Series(y_true).to_numpy()
    predicted = pd.Series(y_pred).to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.6, edgecolor="none")

    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="grey", label="Perfect")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    ax.grid(alpha=0.3)

    path = output_dir / "predicted_vs_actual.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_residuals(y_true: pd.Series, y_pred: Any, output_dir: Path) -> Path:
    """Plot residuals against predictions, alongside their distribution.

    Residuals should scatter randomly around zero. Curvature suggests a missing
    non-linear term; a widening funnel suggests non-constant variance. Neither
    is visible in RMSE alone.

    Args:
        y_true: True targets.
        y_pred: Predicted targets, aligned with ``y_true``.
        output_dir: Directory where the plot file is saved.

    Returns:
        Path to the saved plot file.
    """
    predicted = pd.Series(y_pred).to_numpy()
    residuals = pd.Series(y_true).to_numpy() - predicted

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_scatter.scatter(predicted, residuals, alpha=0.6, edgecolor="none")
    ax_scatter.axhline(y=0, linestyle="--", color="grey")
    ax_scatter.set_xlabel("Predicted")
    ax_scatter.set_ylabel("Residual (actual - predicted)")
    ax_scatter.set_title("Residuals vs Predicted")
    ax_scatter.grid(alpha=0.3)

    sns.histplot(residuals, ax=ax_hist, kde=True, bins=30)
    ax_hist.axvline(x=0, linestyle="--", color="grey")
    ax_hist.set_xlabel("Residual")
    ax_hist.set_title("Residual distribution")

    path = output_dir / "residuals.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
