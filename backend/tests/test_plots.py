"""Tests for the training plot functions extracted into src.ml.plots.

The pipeline tests mock artifact logging, so these exercise the drawing code
itself: that each function writes a real file and returns its path.
"""

import pandas as pd
import pytest

from src.ml.plots import _plot_confusion_matrix, _plot_pr_curve, _plot_roc_curve


@pytest.fixture(name="binary_data")
def binary_data_fixture() -> tuple[pd.Series, pd.Series, pd.Series]:
    """True labels, predicted labels, and positive-class scores."""
    y_true = pd.Series([0, 1] * 10)
    # Same length as y_true, with the final pair flipped so the confusion
    # matrix has entries off the diagonal.
    y_pred = pd.Series([0, 1] * 9 + [1, 0])
    scores = pd.Series([0.1, 0.9] * 10)
    return y_true, y_pred, scores


def _is_png(path) -> bool:
    """Check the file really is a PNG rather than an empty placeholder."""
    return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_confusion_matrix_writes_a_png(binary_data, tmp_path):
    """The confusion matrix is drawn and its path returned."""
    y_true, y_pred, _ = binary_data
    path = _plot_confusion_matrix(y_true, y_pred, [0, 1], tmp_path)

    assert path == tmp_path / "confusion_matrix.png"
    assert _is_png(path)


def test_roc_curve_writes_a_png(binary_data, tmp_path):
    """The ROC curve is drawn and its path returned."""
    y_true, _, scores = binary_data
    path = _plot_roc_curve(y_true, scores, positive_label=1, output_dir=tmp_path)

    assert path == tmp_path / "roc_curve.png"
    assert _is_png(path)


def test_pr_curve_writes_a_png(binary_data, tmp_path):
    """The precision-recall curve is drawn and its path returned."""
    y_true, _, scores = binary_data
    path = _plot_pr_curve(y_true, scores, positive_label=1, output_dir=tmp_path)

    assert path == tmp_path / "pr_curve.png"
    assert _is_png(path)


def test_plots_do_not_leak_figures(binary_data, tmp_path):
    """Every function closes its figure.

    A pipeline draws dozens of plots per run; leaking figures would grow memory
    and eventually trip matplotlib's open-figure warning.
    """
    import matplotlib.pyplot as plt

    y_true, y_pred, scores = binary_data
    before = len(plt.get_fignums())

    _plot_confusion_matrix(y_true, y_pred, [0, 1], tmp_path)
    _plot_roc_curve(y_true, scores, 1, tmp_path)
    _plot_pr_curve(y_true, scores, 1, tmp_path)

    assert len(plt.get_fignums()) == before
