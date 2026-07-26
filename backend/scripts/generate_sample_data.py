"""Generate the bundled demo datasets into ``data/raw/``.

Three datasets ship with the template, one per supported task type, so you can
try each without finding your own data first:

    breast_cancer.csv  binary classification   (the default)
    iris.csv           multiclass classification
    diabetes.csv       regression

All three are bundled with scikit-learn, so generating them needs no network
access — which matters because this script runs during ``setup.sh`` and the
Docker build.

Only the file named by ``data.input_file`` in ``cfg/config.yaml`` is read by the
pipeline; the others sit in ``data/raw/`` until you point the config at one. See
the "dataset presets" comments in that file for the settings each needs.

Every dataset writes its target as a column literally named ``target`` so the
shipped ``target_column: "TARGET"`` works for all three — column names are
normalised to uppercase on read.

Run directly with ``uv run python scripts/generate_sample_data.py``; it is also
wired into ``setup.sh`` and the ``Dockerfile``.
"""

import logging
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from src.config import read_config, resolve_project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _breast_cancer() -> pd.DataFrame:
    """Binary classification: 569 rows, 30 numeric features, target 0/1."""
    return load_breast_cancer(as_frame=True).frame


def _iris() -> pd.DataFrame:
    """Multiclass classification: 150 rows, 4 numeric features, target 0/1/2."""
    return load_iris(as_frame=True).frame


def _diabetes() -> pd.DataFrame:
    """Regression: 442 rows, 10 numeric features, continuous disease score.

    Features arrive mean-centred and scaled, which is realistic for a
    regression demo and needs no preparation to train on.
    """
    return load_diabetes(as_frame=True).frame


#: Filename in data/raw → builder. Add an entry to ship another dataset.
#: Every builder must return a frame whose target column is named ``target``,
#: so one ``target_column`` setting in cfg/config.yaml covers them all.
DATASETS: dict[str, Callable[[], pd.DataFrame]] = {
    "breast_cancer.csv": _breast_cancer,
    "iris.csv": _iris,
    "diabetes.csv": _diabetes,
}


def main() -> None:
    """Write every demo dataset to the configured raw data directory.

    Failures are not caught: every dataset here is bundled with scikit-learn, so
    one failing means something is genuinely wrong rather than merely offline,
    and setup should stop rather than continue with a half-built data directory.
    If you add a dataset that downloads, guard that builder specifically.
    """
    config = read_config()
    raw_dir = resolve_project_path(Path(config["data"]["raw_dir"]))
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, build in DATASETS.items():
        frame = build()
        path = raw_dir / name
        frame.to_csv(path, index=False)
        logger.info("Wrote %s. Shape: %s", path, frame.shape)

    logger.info(
        "%d datasets written. The pipeline reads %r; the others are ignored "
        "until you point data.input_file at one.",
        len(DATASETS),
        config["data"]["input_file"],
    )


if __name__ == "__main__":
    main()
