from unittest.mock import patch

import pandas as pd
import pytest

from src.pipelines.prepare_data import PrepareDataPipeline


@pytest.fixture(name="data")
def sample_df_fixture():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})


def test_run_pipeline(data):
    expected_df = data.copy()
    expected_df["col1"] = expected_df["col1"] * 10  # as per transform() logic

    with (
        patch("src.pipelines.prepare_data.read_data", return_value=data) as mock_read,
        patch("src.pipelines.prepare_data.write_data") as mock_write,
    ):
        pipeline = PrepareDataPipeline()
        pipeline.run()

        mock_read.assert_called_once_with(
            table_name="TEST_DS_TABLE_IRIS", schema_obj="input_data"
        )

        mock_write.assert_called_once()

        written_df = mock_write.call_args[0][0]
        kwargs = mock_write.call_args.kwargs

        pd.testing.assert_frame_equal(written_df, expected_df)
        assert kwargs["table_name"] == "TEST_DS_TABLE_IRIS_PREPARED"
        assert kwargs["schema_obj"] == "prepared_data"
