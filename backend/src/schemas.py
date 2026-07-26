"""Pandera schemas for validating data as it flows through the pipeline.

These schemas are intentionally minimal. Pandera validates in **non-strict**
mode by default: only the columns declared below must be present and pass their
checks — any additional columns are allowed through untouched. So you only need
to declare the few columns you actually want to guard.

The demo dataset is the Breast Cancer Wisconsin set; ``MEAN_RADIUS`` and
``MEAN_TEXTURE`` are declared as representative numeric features. When you bring
your own data, swap these column names (and the target) for your own — usually a
one- or two-line change per schema. Column names are matched **after**
normalisation (uppercase, underscores), e.g. ``mean radius`` → ``MEAN_RADIUS``.
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

# A couple of representative feature columns. Other columns pass through
# untouched (non-strict validation) — add more here to tighten the contract.
_FEATURE_COLUMNS = {
    "MEAN_RADIUS": Column(float, checks=Check.ge(0)),
    "MEAN_TEXTURE": Column(float, checks=Check.ge(0)),
}

_DEFAULT_TARGET_VALUES = [0, 1]


def normalise_column_name(column_name: str) -> str:
    """Normalise a single column name the same way CSV reads do."""
    return re.sub(r"[^A-Z0-9]+", "_", column_name.upper()).strip("_")


def build_schemas(
    target_column: str = "TARGET",
    target_values: Sequence[object] | None = None,
) -> dict[str, DataFrameSchema]:
    """Build schemas using the configured target column and binary values."""
    valid_target_values = list(target_values or _DEFAULT_TARGET_VALUES)
    target_schema = {
        normalise_column_name(target_column): Column(
            checks=Check.isin(valid_target_values)
        )
    }

    input_data_schema = DataFrameSchema({**_FEATURE_COLUMNS, **target_schema})
    prepared_data_schema = DataFrameSchema({**_FEATURE_COLUMNS, **target_schema})
    output_data_schema = DataFrameSchema(
        {
            **_FEATURE_COLUMNS,
            **target_schema,
            "PREDICTION": Column(checks=Check.isin(valid_target_values)),
        }
    )

    return {
        "input_data": input_data_schema,
        "prepared_data": prepared_data_schema,
        "output_data": output_data_schema,
    }


# Dictionary of schemas for lookup by name using demo defaults.
schemas = build_schemas()
