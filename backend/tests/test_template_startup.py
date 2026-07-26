from unittest.mock import patch

import pytest

from src.cli import main
from src.config import read_config


def test_read_config_resolves_from_project_root(monkeypatch, tmp_path):
    """Config can be loaded when callers run from outside the repo root."""
    monkeypatch.chdir(tmp_path)

    config = read_config()

    assert config["data"]["input_file"] == "breast_cancer.csv"


def test_main_rejects_multiple_step_flags():
    """Combined step flags should fail loudly instead of silently skipping steps."""
    with (
        patch("sys.argv", ["pipeline", "--prepare-data", "--train-model"]),
        pytest.raises(ValueError, match="Choose only one pipeline flag"),
    ):
        main()
