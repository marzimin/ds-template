"""Pandera schemas for validating data as it flows through the pipeline.

Pandera validates in **non-strict** mode: only the columns declared here must be
present and pass their checks, and any other column passes through untouched. So
you declare only the columns you actually want to guard.

Out of the box, only the target column is checked, which means **any dataset
runs without editing this file**. Naming specific feature columns here would
make the template fail on the first run with anyone else's data, so
:data:`FEATURE_COLUMNS` ships empty with a worked example below.

Add checks for the columns whose shape you care about — the ones whose silent
corruption would ruin a model. Column names are matched **after** normalisation
(uppercase, underscores), e.g. ``mean radius`` → ``MEAN_RADIUS``::

    FEATURE_COLUMNS = {
        "MEAN_RADIUS": Column(float, checks=Check.ge(0)),
        "AGE": Column(int, checks=Check.in_range(0, 120)),
    }
"""

import re
import warnings
from collections.abc import Sequence

from pandera.pandas import Check, Column, DataFrameSchema

# Ignore some Pydantic user warnings surfaced via MLflow/Pandera.
warnings.filterwarnings(
    "ignore",
    message='.*Field "model_server_url" has conflict with protected namespace.*',
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message=".*Valid config keys have changed in V2*", category=UserWarning
)

#: Feature columns to validate. Empty by default so any dataset runs unedited;
#: see the module docstring for how to add your own.
FEATURE_COLUMNS: dict[str, Column] = {}


def normalise_column_name(column_name: str) -> str:
    """Normalise a single column name the same way CSV reads do.

    This rule is a cross-repository contract, not an internal detail.
    de-template reproduces it in ``scripts/demo_handoff.py`` to predict what
    this project will call each exported column, and cannot import it — the two
    are separate repositories. Changing the rule here silently invalidates the
    names asserted there. See ``docs/handoff.md``.
    """
    return re.sub(r"[^A-Z0-9]+", "_", column_name.upper()).strip("_")


def build_schemas(
    target_column: str = "TARGET",
    target_values: Sequence[object] | None = None,
) -> dict[str, DataFrameSchema]:
    """Build the pipeline schemas for the configured target.

    Args:
        target_column: Name of the target column, before normalisation.
        target_values: Permitted target values. Supply these for classification
            to catch stray labels. Leave unset for regression — a continuous
            target has no fixed set of values, and checking one would reject
            every valid row.

    Returns:
        Schemas keyed ``input_data``, ``prepared_data``, and ``output_data``.
    """
    name = normalise_column_name(target_column)

    # Only constrain the target's values when a set was configured. This is what
    # lets the same schema serve classification and regression.
    checks = Check.isin(list(target_values)) if target_values else None
    target_schema = {name: Column(checks=checks, nullable=False)}
    prediction_schema = {"PREDICTION": Column(checks=checks, nullable=False)}

    return {
        "input_data": DataFrameSchema({**FEATURE_COLUMNS, **target_schema}),
        "prepared_data": DataFrameSchema({**FEATURE_COLUMNS, **target_schema}),
        "output_data": DataFrameSchema(
            {**FEATURE_COLUMNS, **target_schema, **prediction_schema}
        ),
    }


# Dictionary of schemas for lookup by name using demo defaults.
schemas = build_schemas()
