from unittest.mock import patch

import pandas as pd

from src.ml.prepare_data import PrepareDataPipeline

_MOCK_CONFIG = {
    "data": {
        "dir": "data/processed",
        "raw_dir": "data/raw",
        "input_file": "breast_cancer.csv",
    }
}


def test_run_pipeline():
    """Test that the pipeline reads input, applies transforms, and writes output."""
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})

    with (
        patch("src.ml.prepare_data.read_config", return_value=_MOCK_CONFIG),
        patch("src.ml.prepare_data.read_data", return_value=data) as mock_read,
        patch("src.ml.prepare_data.write_data") as mock_write,
    ):
        pipeline = PrepareDataPipeline()
        pipeline.run()

        mock_read.assert_called_once_with(
            file_name="breast_cancer.csv", raw=True, schema_obj="input_data"
        )

        mock_write.assert_called_once()
        written_df = mock_write.call_args[0][0]
        kwargs = mock_write.call_args.kwargs

        # _func1–_func4 are all no-ops: output should equal input unchanged
        pd.testing.assert_frame_equal(written_df, data)
        assert kwargs["file_name"] == "breast_cancer.csv"
        assert kwargs["suffix"] == "prepared"
        assert kwargs["schema_obj"] == "prepared_data"
