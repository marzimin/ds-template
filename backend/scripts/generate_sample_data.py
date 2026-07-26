"""Generate the bundled demo dataset (Breast Cancer Wisconsin).

This writes the raw CSV that the pipeline consumes so the template runs out of
the box. It is wired into ``setup.sh`` and the ``Dockerfile``; you can also run
it manually with ``uv run python scripts/generate_sample_data.py``.

To bring your own data instead, drop a CSV into ``data/raw/`` named to match
``cfg/config.yaml:data.input_file`` and update ``src/utils/schemas.py``. The
column names below are left in their original (spaced, lower-case) form on
purpose to demonstrate the column normalisation applied on read.
"""

import logging
from pathlib import Path

from sklearn.datasets import load_breast_cancer

from src.utils.utils import read_config, resolve_project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Write the demo dataset to the configured raw data directory."""
    config = read_config()
    raw_dir = resolve_project_path(Path(config["data"]["raw_dir"]))
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / config["data"]["input_file"]

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame  # 30 feature columns + a "target" column

    df.to_csv(output_path, index=False)
    logger.info("Demo dataset written to %s. Shape: %s", output_path, df.shape)


if __name__ == "__main__":
    main()
