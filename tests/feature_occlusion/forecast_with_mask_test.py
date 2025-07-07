import xarray as xr
import numpy as np
from unittest import mock
from pkg.feature_occlusion import forecast_with_mask
import math


def dummy_ds():
    data = np.ones((1, 2, 2, 4, 4), dtype=np.float32)
    return xr.Dataset(
        {
            "temp": (("batch", "time", "level", "lat", "lon"), data),
        },
        coords={
            "batch": [0],
            "time": [0, 1],
            "level": [500, 850],
            "lat": [45, 50, 55, 60],
            "lon": [340, 345, 350, 355],
        }
    )


@mock.patch("pkg.feature_occlusion.forecast_with_mask.run_forecast")
@mock.patch("pkg.feature_occlusion.forecast_with_mask.apply_mask_to_input")
@mock.patch("pkg.feature_occlusion.forecast_with_mask.create_feature_mask")
def test_run_forecast_with_mask_calls_all_steps(
    mock_create_mask, mock_apply_mask, mock_run_forecast
):
    input_ds = dummy_ds()
    climatology = dummy_ds()

    mock_create_mask.return_value = "FAKE_MASK"
    mock_apply_mask.return_value = "FAKE_MASKED_INPUT"

    forecast_with_mask.run_forecast_with_mask(
        input_data=input_ds,
        climatology=climatology,
        output_name="occluded_test",
        predictions_folder="/tmp/test_preds",
        variables=["temp"],
        pressure_levels=[500],
        regions=["north_germany"],
        time_steps=[0],
        model="graphcast_test",
        upload_to_gcs=True
    )

    mock_create_mask.assert_called_once_with(input_ds, ["temp"], [500], ["north_germany"], [0])
    mock_apply_mask.assert_called_once_with(
        input_data=input_ds,
        climatology_data=climatology,
        mask="FAKE_MASK"
    )
    mock_run_forecast.assert_called_once_with(
        model_name="graphcast_test",
        input_data_or_path="FAKE_MASKED_INPUT",
        output_name="occluded_test",
        upload_to_gcs=True,
        predictions_folder="/tmp/test_preds",
    )


@mock.patch("pkg.feature_occlusion.forecast_with_mask.run_forecast_with_mask")
def test_run_multiple_forecasts_with_mask_runs_all_combinations(mock_run):
    input_ds = dummy_ds()
    climatology = dummy_ds()

    vars_ = [["temp"]]
    levels = [[500], [850]]
    regions = [["north_germany"], ["storm_region"]]
    time_steps = [[0], [1]]

    forecast_with_mask.run_multiple_forecasts_with_mask(
        input_data=input_ds,
        climatology=climatology,
        predictions_folder="/tmp/multi",
        variable_groups=vars_,
        pressure_levels=levels,
        regions=regions,
        time_steps=time_steps,
        model="graphcast_mock",
        upload_to_gcs=False
    )

    expected_count = math.prod(map(len, [vars_, levels, regions, time_steps]))
    assert mock_run.call_count == expected_count
    for call in mock_run.call_args_list:
        args, kwargs = call
        assert kwargs["model"] == "graphcast_mock"
        assert kwargs["upload_to_gcs"] is False
        assert kwargs["input_data"] is input_ds
        assert kwargs["climatology"] is climatology
        assert "/tmp/multi" in kwargs["predictions_folder"]
