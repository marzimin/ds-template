"""Generate the bundled demo datasets into ``data/raw/``.

Three datasets ship with the template, one per supported task type, so you can
try each without finding your own data first:

    breast_cancer.csv      binary classification   (the default)
    iris.csv               multiclass classification
    california_housing.csv regression

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
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris

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


def _california_housing() -> pd.DataFrame:
    """Regression: 20,640 rows, 8 numeric features, continuous house value.

    Unlike the other two this is downloaded rather than bundled with
    scikit-learn, and cached under ``~/scikit_learn_data`` afterwards.
    """
    frame = fetch_california_housing(as_frame=True).frame
    # Name the target consistently with the other datasets so one
    # `target_column` setting covers all three.
    return frame.rename(columns={"MedHouseVal": "target"})


#: Filename in data/raw → builder. Add an entry to ship another dataset.
DATASETS: dict[str, Callable[[], pd.DataFrame]] = {
    "breast_cancer.csv": _breast_cancer,
    "iris.csv": _iris,
    "california_housing.csv": _california_housing,
}


def write_dataset(name: str, build: Callable[[], pd.DataFrame], raw_dir: Path) -> bool:
    """Build one dataset and write it to ``raw_dir``.

    Args:
        name: Output file name.
        build: Callable returning the frame to write.
        raw_dir: Destination directory.

    Returns:
        True when written, False when it could not be produced.
    """
    try:
        frame = build()
    except Exception as exc:  # noqa: BLE001 - one dataset must not block the rest
        # California housing is fetched over the network. Skipping it leaves a
        # usable template rather than failing setup on an offline machine.
        logger.warning(
            "Could not generate %s (%s). The other datasets are unaffected; "
            "rerun this script when you have network access.",
            name,
            exc,
        )
        return False

    path = raw_dir / name
    frame.to_csv(path, index=False)
    logger.info("Wrote %s. Shape: %s", path, frame.shape)
    return True


def main() -> None:
    """Write every demo dataset to the configured raw data directory."""
    config = read_config()
    raw_dir = resolve_project_path(Path(config["data"]["raw_dir"]))
    raw_dir.mkdir(parents=True, exist_ok=True)

    written = [
        name for name, build in DATASETS.items() if write_dataset(name, build, raw_dir)
    ]

    selected = config["data"]["input_file"]
    logger.info(
        "%d of %d datasets written. The pipeline reads %r; the others are "
        "ignored until you point data.input_file at one.",
        len(written),
        len(DATASETS),
        selected,
    )


if __name__ == "__main__":
    main()
