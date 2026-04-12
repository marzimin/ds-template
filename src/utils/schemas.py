import warnings

from pandera.pandas import Check, Column, DataFrameSchema

# Ignore some Pydantic user warnings
warnings.filterwarnings(
    "ignore",
    message='.*Field "model_server_url" has conflict with protected namespace.*',
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message=".*Valid config keys have changed in V2*", category=UserWarning
)

# ---------------------------------------------------------------------------
# Replace the column definitions below with your own dataset's columns.
# All 30 features of the Breast Cancer Wisconsin dataset are listed as an
# example.  input_data_schema and prepared_data_schema share the same
# structure; output_data_schema adds a PREDICTION column.
# ---------------------------------------------------------------------------

_FEATURE_COLUMNS = {
    "MEAN_RADIUS": Column(float, checks=Check.ge(0)),
    "MEAN_TEXTURE": Column(float, checks=Check.ge(0)),
    "MEAN_PERIMETER": Column(float, checks=Check.ge(0)),
    "MEAN_AREA": Column(float, checks=Check.ge(0)),
    "MEAN_SMOOTHNESS": Column(float, checks=Check.ge(0)),
    "MEAN_COMPACTNESS": Column(float, checks=Check.ge(0)),
    "MEAN_CONCAVITY": Column(float, checks=Check.ge(0)),
    "MEAN_CONCAVE_POINTS": Column(float, checks=Check.ge(0)),
    "MEAN_SYMMETRY": Column(float, checks=Check.ge(0)),
    "MEAN_FRACTAL_DIMENSION": Column(float, checks=Check.ge(0)),
    "RADIUS_ERROR": Column(float, checks=Check.ge(0)),
    "TEXTURE_ERROR": Column(float, checks=Check.ge(0)),
    "PERIMETER_ERROR": Column(float, checks=Check.ge(0)),
    "AREA_ERROR": Column(float, checks=Check.ge(0)),
    "SMOOTHNESS_ERROR": Column(float, checks=Check.ge(0)),
    "COMPACTNESS_ERROR": Column(float, checks=Check.ge(0)),
    "CONCAVITY_ERROR": Column(float, checks=Check.ge(0)),
    "CONCAVE_POINTS_ERROR": Column(float, checks=Check.ge(0)),
    "SYMMETRY_ERROR": Column(float, checks=Check.ge(0)),
    "FRACTAL_DIMENSION_ERROR": Column(float, checks=Check.ge(0)),
    "WORST_RADIUS": Column(float, checks=Check.ge(0)),
    "WORST_TEXTURE": Column(float, checks=Check.ge(0)),
    "WORST_PERIMETER": Column(float, checks=Check.ge(0)),
    "WORST_AREA": Column(float, checks=Check.ge(0)),
    "WORST_SMOOTHNESS": Column(float, checks=Check.ge(0)),
    "WORST_COMPACTNESS": Column(float, checks=Check.ge(0)),
    "WORST_CONCAVITY": Column(float, checks=Check.ge(0)),
    "WORST_CONCAVE_POINTS": Column(float, checks=Check.ge(0)),
    "WORST_SYMMETRY": Column(float, checks=Check.ge(0)),
    "WORST_FRACTAL_DIMENSION": Column(float, checks=Check.ge(0)),
}

_TARGET_COLUMN = {"TARGET": Column(int, checks=Check.isin([0, 1]))}

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
