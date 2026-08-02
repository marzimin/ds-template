"""Tests for the cached model picking up newly registered versions.

Training while the API runs used to leave the old model in memory until someone
restarted the process. These cover the staleness check that replaced that: it
must reload when the registry moves on, stay off the network otherwise, and
never let a registry outage take a working model out of service.
"""

from unittest.mock import patch

import pytest
from mlflow.exceptions import MlflowException

from src.ml import inference
from src.ml.inference import (
    LoadedModel,
    ModelNotAvailableError,
    clear_model_cache,
    get_cached_model,
)

CONFIG: dict = {}


@pytest.fixture(autouse=True)
def _clean_cache():
    """Never let one test's cached model leak into the next."""
    clear_model_cache()
    yield
    clear_model_cache()


def _model(version: str) -> LoadedModel:
    return LoadedModel(
        name="test-model",
        version=version,
        run_id=None,
        features=(),
        pyfunc_model=object(),
    )


def test_loads_once_and_reuses_within_the_interval():
    """A warm cache serves without touching the registry at all."""
    with (
        patch.object(inference, "load_model", return_value=_model("1")) as load,
        patch.object(inference, "_newest_registered_version") as newest,
    ):
        assert get_cached_model(CONFIG).version == "1"
        assert get_cached_model(CONFIG).version == "1"

    assert load.call_count == 1
    newest.assert_not_called()


def test_reloads_when_the_registry_has_a_newer_version(monkeypatch):
    """The point of the whole thing: a new version is picked up in place."""
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", "0.01")
    with (
        patch.object(
            inference, "load_model", side_effect=[_model("1"), _model("2")]
        ) as load,
        patch.object(inference, "_newest_registered_version", return_value="2"),
        # Load stamps _checked_at, the next request compares against it, then
        # the attempt is re-stamped. Advancing past the interval on the compare
        # is what puts the check on the second request.
        patch.object(inference.time, "monotonic", side_effect=[0.0, 5.0, 5.0]),
    ):
        assert get_cached_model(CONFIG).version == "1"
        assert get_cached_model(CONFIG).version == "2"

    assert load.call_count == 2


def test_does_not_reload_when_the_version_is_unchanged(monkeypatch):
    """The check is cheap; the reload is not. Only the check should repeat."""
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", "0.01")
    with (
        patch.object(inference, "load_model", return_value=_model("1")) as load,
        patch.object(
            inference, "_newest_registered_version", return_value="1"
        ) as newest,
        patch.object(inference.time, "monotonic", side_effect=[0.0, 5.0, 5.0]),
    ):
        assert get_cached_model(CONFIG).version == "1"
        assert get_cached_model(CONFIG).version == "1"

    assert load.call_count == 1
    assert newest.call_count == 1


def test_registry_outage_keeps_serving_the_loaded_model(monkeypatch):
    """A tracking server that is down must not take a working model with it."""
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", "0.01")
    with (
        patch.object(inference, "load_model", return_value=_model("1")) as load,
        patch.object(
            inference, "_latest_version", side_effect=MlflowException("down")
        ) as latest,
        patch.object(inference.time, "monotonic", side_effect=[0.0, 5.0, 5.0]),
        patch.object(inference, "registered_model_name", return_value="test-model"),
    ):
        assert get_cached_model(CONFIG).version == "1"
        # The check runs, fails, and is swallowed — no exception, no reload.
        assert get_cached_model(CONFIG).version == "1"

    assert load.call_count == 1
    # Guards against passing for the wrong reason: the outage path is only
    # covered if the check was actually attempted.
    assert latest.call_count == 1


def test_zero_disables_the_check(monkeypatch):
    """Opting out leaves POST /api/predict/reload as the only trigger."""
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", "0")
    with (
        patch.object(inference, "load_model", return_value=_model("1")),
        patch.object(inference, "_newest_registered_version") as newest,
    ):
        get_cached_model(CONFIG)
        get_cached_model(CONFIG)

    newest.assert_not_called()


def test_a_failed_check_is_not_retried_until_the_next_interval(monkeypatch):
    """A down registry costs one query per interval, not one per request."""
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", "10")
    with (
        patch.object(inference, "load_model", return_value=_model("1")),
        patch.object(
            inference, "_newest_registered_version", return_value=None
        ) as newest,
        # load, then t=20 (checks), then two requests still inside the window.
        patch.object(
            inference.time, "monotonic", side_effect=[0.0, 20.0, 20.0, 21.0, 22.0]
        ),
    ):
        for _ in range(4):
            assert get_cached_model(CONFIG).version == "1"

    assert newest.call_count == 1


def test_first_load_failure_propagates():
    """With nothing cached there is no fallback, so the error must surface."""
    with patch.object(
        inference, "load_model", side_effect=ModelNotAvailableError("nothing yet")
    ):
        with pytest.raises(ModelNotAvailableError):
            get_cached_model(CONFIG)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("15", 15.0), ("0", 0.0), ("-5", 0.0), ("nonsense", 30.0), ("", 30.0)],
)
def test_refresh_interval_parsing(monkeypatch, value, expected):
    """A malformed or negative interval falls back rather than crashing."""
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", value)
    assert inference._refresh_interval() == expected


def test_refresh_interval_defaults_when_unset(monkeypatch):
    """The knob is optional."""
    monkeypatch.delenv("MODEL_REFRESH_SECONDS", raising=False)
    assert inference._refresh_interval() == 30.0
