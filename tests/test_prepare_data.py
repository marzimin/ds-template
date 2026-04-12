from unittest.mock import patch

import pandas as pd

from src.pipelines.prepare_data import PrepareDataPipeline

_MOCK_CONFIG = {
    "data": {
        "input_file": "input_data.csv",
        "prepared_file": "prepared_data.csv",
        "output_file": "output_data.csv",
    }
}


def test_run_pipeline():
    """Test that the pipeline reads input, applies transforms, and writes output."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})

    with (
        patch("src.pipelines.prepare_data.read_config", return_value=_MOCK_CONFIG),
        patch("src.pipelines.prepare_data.read_data", return_value=data) as mock_read,
        patch("src.pipelines.prepare_data.write_data") as mock_write,
    ):
        pipeline = PrepareDataPipeline()
        pipeline.run()

        mock_read.assert_called_once_with(
            file_name="input_data.csv", schema_obj="input_data"
        )

        mock_write.assert_called_once()
        written_df = mock_write.call_args[0][0]
        kwargs = mock_write.call_args.kwargs

        # _func1–_func4 are all no-ops: output should equal input unchanged
        pd.testing.assert_frame_equal(written_df, data)
        assert kwargs["file_name"] == "prepared_data.csv"
        assert kwargs["schema_obj"] == "prepared_data"
