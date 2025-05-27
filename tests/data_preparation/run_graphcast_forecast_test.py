import os
import io
import pytest
import dataclasses
import xarray as xr
import numpy as np
import functools
from unittest import mock

from pkg.data_preparation.run_graphcast_forecast import (
    _get_model_checkpoint,
    _load_normalization_data,
    _build_model,
    run_forecast,
)


# --- Helpers & Fixtures ----------------------------------------------------

class DummyBlob:
    def __init__(self, name, content=b"dummy"):
        self.name = name
        self._buf = io.BytesIO(content)

    def open(self, mode):
        # Return a fresh file-like Object each time
        return io.BytesIO(self._buf.getvalue())


@dataclasses.dataclass
class DummyTaskConfig:
    bar: int


class DummyCKPT:
    def __init__(self):
        self.model_config = {"foo": 1}
        self.task_config  = DummyTaskConfig(bar=2)
        self.params       = {"baz": 3}


@pytest.fixture(autouse=True)
def noop_makedirs(monkeypatch):
    # Avoid creating real directories
    monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)


# --- Tests -------------------------------------------------------

def test_get_model_checkpoint_success(monkeypatch):
    # Fake GCS bucket listing a matching blob
    fake_blobs = [DummyBlob("graphcast/params/GRAPHCAST_test.chkpt")]
    fake_bucket = mock.MagicMock()
    fake_bucket.list_blobs.return_value = fake_blobs
    fake_bucket.blob.return_value    = fake_blobs[0]
    fake_client = mock.MagicMock(bucket=lambda name: fake_bucket)

    # Patch the anonymous client creation
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.storage.Client",
        mock.Mock(create_anonymous_client=lambda: fake_client),
    )
    # Patch checkpoint.load to return our DummyCKPT
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.checkpoint.load",
        lambda f, cls: DummyCKPT(),
    )

    ckpt, bucket, prefix = _get_model_checkpoint("GRAPHCAST")
    assert isinstance(ckpt, DummyCKPT)
    assert bucket is fake_bucket
    assert prefix == "graphcast/"


def test_get_model_checkpoint_notfound(monkeypatch):
    fake_bucket = mock.MagicMock()
    fake_bucket.list_blobs.return_value = []
    fake_client = mock.MagicMock(bucket=lambda name: fake_bucket)

    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.storage.Client",
        mock.Mock(create_anonymous_client=lambda: fake_client),
    )

    with pytest.raises(FileNotFoundError):
        _get_model_checkpoint("NON_EXISTENT_MODEL")


def test_load_normalization_data(monkeypatch):
    # Prepare a dummy Dataset for xr.load_dataset
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.xr.load_dataset",
        lambda f: ds,
    )

    fake_bucket = mock.MagicMock()
    # Each open() must return a fresh BytesIO
    fake_bucket.blob.return_value.open.side_effect = lambda mode: io.BytesIO(b"")

    diffs, mean, std = _load_normalization_data(fake_bucket, "myprefix/")
    assert isinstance(diffs, xr.Dataset)
    assert isinstance(mean,  xr.Dataset)
    assert isinstance(std,   xr.Dataset)


def test_build_model(monkeypatch):
    # Stub out the GraphCast wrappers
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.graphcast.GraphCast",
        lambda mc, tc: "GRAPHCAST_INSTANCE",
    )
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.casting.Bfloat16Cast",
        lambda pred: f"CAST({pred})",
    )
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.normalization.InputsAndResiduals",
        lambda pred, **kw: f"NORM({pred})",
    )
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.autoregressive.Predictor",
        lambda pred, gradient_checkpointing: f"PRED({pred})",
    )

    dummy_ckpt = DummyCKPT()
    diffs = mean = std = xr.Dataset()
    constructor = _build_model(dummy_ckpt, diffs, mean, std)
    result = constructor({"a": 1}, {"b": 2})
    assert result == "PRED(NORM(CAST(GRAPHCAST_INSTANCE)))"


@mock.patch("pkg.data_preparation.run_graphcast_forecast.xr.Dataset.to_netcdf", lambda self, path: None)
@mock.patch("pkg.data_preparation.run_graphcast_forecast.gcs.upload_file")
@mock.patch("pkg.data_preparation.run_graphcast_forecast.xr.open_dataset")
@mock.patch("pkg.data_preparation.run_graphcast_forecast.rollout.chunked_prediction")
@mock.patch("pkg.data_preparation.run_graphcast_forecast.data_utils.extract_inputs_targets_forcings")
@mock.patch("pkg.data_preparation.run_graphcast_forecast._load_normalization_data")
@mock.patch("pkg.data_preparation.run_graphcast_forecast._get_model_checkpoint")
def test_run_forecast_happy_path(
    mock_getckpt,
    mock_loadnorm,
    mock_extract,
    mock_rollout,
    mock_open_ds,
    mock_upload,
    tmp_path,
):
    # 1) Fake checkpoint & normalization
    ckpt = DummyCKPT()
    bucket, prefix = object(), "graphcast/"
    mock_getckpt.return_value = (ckpt, bucket, prefix)

    diffs = mean = std = xr.Dataset()
    mock_loadnorm.return_value = (diffs, mean, std)

    # 2) Fake input dataset
    fake_input = xr.Dataset({"X": ("time", [0.0])})
    mock_open_ds.return_value.load.return_value = fake_input

    # 3) Fake extract_inputs_targets_forcings
    fake_inp = np.zeros((1,))
    fake_targ = np.ones((1,))
    fake_forc = np.full((1,), 2.0)
    mock_extract.return_value = (fake_inp, fake_targ, fake_forc)

    # 4) Fake rollout
    out_ds = xr.Dataset({"Y": ("time", [42.0])})
    mock_rollout.return_value = out_ds

    # 5) Run, with upload=True
    out_folder = tmp_path / "predictions"
    result = run_forecast(
        model_name="graphcast",
        input_path="irrelevant.nc",
        output_name="myout",
        upload_to_gcs=True,
        predictions_folder=str(out_folder),
        forecast_hours=12,
    )

    expected = os.path.join(str(out_folder), "myout.nc")
    assert result == expected
    mock_upload.assert_called_once_with(expected, expected)


def test_run_forecast_no_upload(tmp_path, monkeypatch):
    # When upload_to_gcs=False, upload_file should never be invoked
    ckpt = DummyCKPT()
    bucket, prefix = object(), "graphcast/"
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast._get_model_checkpoint",
        lambda name: (ckpt, bucket, prefix),
    )
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast._load_normalization_data",
        lambda b, p: (xr.Dataset(), xr.Dataset(), xr.Dataset()),
    )
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.xr.open_dataset",
        lambda path: mock.MagicMock(load=lambda: xr.Dataset({"X": ("time",[0.])})),
    )
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.data_utils.extract_inputs_targets_forcings",
        lambda *a, **kw: (np.zeros((1,)), np.zeros((1,)), np.zeros((1,))),
    )
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.rollout.chunked_prediction",
        lambda *a, **kw: xr.Dataset({"Y": ("time",[0.])}),
    )
    # Stub out to_netcdf & upload_file
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.xr.Dataset.to_netcdf",
        lambda self, path: None,
    )
    called = []
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast.gcs.upload_file",
        lambda src, dst: called.append((src, dst))
    )

    result = run_forecast(
        model_name="graphcast",
        input_path="irrelevant.nc",
        output_name="test",
        upload_to_gcs=False,
        predictions_folder=str(tmp_path/"preds"),
        forecast_hours=6,
    )
    assert os.path.basename(result) == "test.nc"
    assert called == []


def test_run_forecast_missing_stats_bucket(monkeypatch):
    # If bucket or prefix is falsy, should raise RuntimeError
    monkeypatch.setattr(
        "pkg.data_preparation.run_graphcast_forecast._get_model_checkpoint",
        lambda name: (DummyCKPT(), None, None),
    )

    with pytest.raises(RuntimeError) as exc:
        run_forecast(
            model_name="graphcast",
            input_path="irrelevant.nc",
            output_name="fail",
            upload_to_gcs=False,
            predictions_folder="unused",
            forecast_hours=6,
        )
    assert "lacks normalization stats" in str(exc.value)
