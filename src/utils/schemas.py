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

import warnings

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

_TARGET_COLUMN = {"TARGET": Column(int, checks=Check.isin([0, 1]))}

# input_data and prepared_data share the same contract; output_data additionally
# carries the model's PREDICTION column.
input_data_schema = DataFrameSchema({**_FEATURE_COLUMNS, **_TARGET_COLUMN})

prepared_data_schema = DataFrameSchema({**_FEATURE_COLUMNS, **_TARGET_COLUMN})

output_data_schema = DataFrameSchema(
    {
        **_FEATURE_COLUMNS,
        **_TARGET_COLUMN,
        "PREDICTION": Column(int, checks=Check.isin([0, 1])),
    }
)

# Dictionary of schemas for lookup by name
schemas = {
    "input_data": input_data_schema,
    "prepared_data": prepared_data_schema,
    "output_data": output_data_schema,
}
