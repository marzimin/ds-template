import warnings

from pandera import Check, Column, DataFrameSchema

# Ignore some Pydantic user warnings
warnings.filterwarnings(
    "ignore",
    message='.*Field "model_server_url" has conflict with protected namespace.*',
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message=".*Valid config keys have changed in V2*", category=UserWarning
)


# Define individual schemas
input_data_schema = DataFrameSchema(
    {
        "SEPAL_LENGTH_CM": Column(
            float, checks=Check.ge(0)
        ),  # Greater than or equal to 0
        "SEPAL_WIDTH_CM": Column(float, checks=Check.ge(0)),
        "PETAL_LENGTH_CM": Column(float, checks=Check.ge(0)),
        "PETAL_WIDTH_CM": Column(float, checks=Check.ge(0)),
        "TARGET": Column(
            int, checks=Check.isin([0, 1, 2])
        ),  # Assuming target is categorical with values 0, 1, 2
    }
)

prepared_data_schema = DataFrameSchema(
    {
        "SEPAL_LENGTH_CM": Column(
            float, checks=Check.ge(0)
        ),  # Greater than or equal to 0
        "SEPAL_WIDTH_CM": Column(float, checks=Check.ge(0)),
        "PETAL_LENGTH_CM": Column(float, checks=Check.ge(0)),
        "PETAL_WIDTH_CM": Column(float, checks=Check.ge(0)),
        "TARGET": Column(
            int, checks=Check.isin([0, 1, 2])
        ),  # Assuming target is categorical with values 0, 1, 2
    }
)

output_data_schema = DataFrameSchema(
    {
        "SEPAL_LENGTH_CM": Column(
            float, checks=Check.ge(0)
        ),  # Greater than or equal to 0
        "SEPAL_WIDTH_CM": Column(float, checks=Check.ge(0)),
        "PETAL_LENGTH_CM": Column(float, checks=Check.ge(0)),
        "PETAL_WIDTH_CM": Column(float, checks=Check.ge(0)),
        "TARGET": Column(
            int, checks=Check.isin([0, 1, 2])
        ),  # Assuming target is categorical with values 0, 1, 2
        "PREDICTION": Column(
            int, checks=Check.isin([0, 1, 2])
        ),  # Predictions should also be in the same set
    }
)

# Dictionary of schemas
schemas = {
    "input_data": input_data_schema,
    "prepared_data": prepared_data_schema,
    "output_data": output_data_schema,
}
