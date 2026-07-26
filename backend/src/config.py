"""Project paths, environment, and YAML configuration.

This module is deliberately free of heavy dependencies (no pandas, matplotlib,
seaborn, or mlflow) so that the API layer can resolve paths and read
configuration without pulling the plotting and modelling stack into its import
graph.
"""

import logging
import os
import tomllib
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_NAME = "ds-template"


def _resolve_project_root() -> Path:
    """Locate the repository root that holds ``cfg/``, ``data/`` and ``outputs/``.

    These directories live at the repository root — one level above
    ``backend/`` — so that the user-configurable surface of the template stays
    separate from the backend and frontend code that consumes it.

    Set ``DS_PROJECT_ROOT`` to override, which is how the container image and
    any non-standard checkout layout point at the right directory instead of
    relying on this file's depth on disk.

    Returns:
        Absolute path to the project root.
    """
    env_root = os.getenv("DS_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # .../<root>/backend/src/config.py -> parents[2] is <root>
    return Path(__file__).resolve().parents[2]


#: Repository root: holds cfg/, data/, outputs/ and .env.
PROJECT_ROOT = _resolve_project_root()

#: The backend project directory: holds pyproject.toml and the src package.
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# The project root holds the single .env shared by the backend, the container
# build, and docker compose. Load it explicitly rather than relying on the
# current working directory.
load_dotenv(PROJECT_ROOT / ".env")


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative project paths from the repository root."""
    project_path = Path(path)
    if project_path.is_absolute():
        return project_path
    return PROJECT_ROOT / project_path


def read_config() -> dict[str, Any]:
    """Read the YAML configuration file and return its contents.

    Returns:
        dict[str, Any]: Configuration settings loaded from ``cfg/config.yaml``.
    """
    config_file_path = resolve_project_path(Path("cfg") / "config.yaml")
    with open(config_file_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return cast(dict[str, Any], config_data)


def project_name() -> str:
    """Return the project name declared in ``backend/pyproject.toml``.

    Used to derive defaults — notably the MLflow registered model name — so that
    renaming the project in one place renames everything downstream, rather than
    leaving template placeholders scattered through the configuration.

    Returns:
        The ``[project].name`` value, or :data:`DEFAULT_PROJECT_NAME` if the
        manifest is missing or does not declare one.
    """
    pyproject_path = BACKEND_ROOT / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as file:
            manifest = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning(
            "Could not read %s; falling back to project name %r.",
            pyproject_path,
            DEFAULT_PROJECT_NAME,
        )
        return DEFAULT_PROJECT_NAME

    name = manifest.get("project", {}).get("name")
    return str(name) if name else DEFAULT_PROJECT_NAME
