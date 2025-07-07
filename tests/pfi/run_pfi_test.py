import pytest
import xarray as xr
import numpy as np
import pandas as pd
from unittest import mock
from pkg.pfi import run_pfi


@pytest.fixture
def dummy_dataset():
    data = np.ones((1, 2, 2, 4, 4))  # shape: (batch, time, level, lat, lon)
    coords = {
        "batch": [0],
        "time": [0, 1],
        "level": [500, 850],
        "lat": [45, 50, 55, 60],
        "lon": [340, 345, 350, 355],
    }
    ds = xr.Dataset({
        "temp": ("batch time level lat lon".split(), data),
        "v": ("batch time level lat lon".split(), data * 2),
    }, coords=coords)
    return ds


def test_permute_variable_with_mask_permuted_values(dummy_dataset):
    ds = dummy_dataset
    mask = xr.ones_like(ds, dtype=int)
    mask["temp"][:] = 1  # mask everything

    shuffled = run_pfi.permute_variable_with_mask(ds, "temp", mask)

    assert "temp" in shuffled
    assert shuffled["temp"].shape == ds["temp"].shape


def test_compute_rmse_produces_float():
    lat = [54, 55]
    lon = [9, 10]
    shape = (2, 2)

    u10 = xr.DataArray(np.ones(shape), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    v10 = xr.DataArray(np.ones(shape), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

    ds = xr.Dataset({
        "10m_u_component_of_wind": u10,
        "10m_v_component_of_wind": v10
    })

    rmse = run_pfi.compute_rmse(ds, ds)
    assert isinstance(rmse, float)
    assert rmse == 0.0


@mock.patch("pkg.pfi.run_pfi.run_forecast")
def test_run_single_permutation_calls_forecast(mock_forecast, dummy_dataset):
    ds = dummy_dataset
    mask = xr.ones_like(ds, dtype=int)
    run_pfi.run_single_permutation(ds, ["temp"], mask, run_index=0, folder="test_folder")

    mock_forecast.assert_called_once()


@mock.patch("pkg.pfi.run_pfi.run_single_permutation")
@mock.patch("pkg.pfi.run_pfi.create_feature_mask")
def test_run_pfi_main_loop(mock_create_mask, mock_run_single, dummy_dataset):
    mock_create_mask.return_value = xr.ones_like(dummy_dataset, dtype=int)

    run_pfi.run_pfi(
        ds_input=dummy_dataset,
        features=["temp"],
        pressure_levels=[500],
        regions=["north_germany"],
        time_steps=[0],
        repetitions=2,
        folder="test_output"
    )

    assert mock_run_single.call_count == 2


def test_evaluate_pfi_folder(tmp_path):
    lat, lon = [54, 55], [9, 10]
    shape = (2, 2)
    u10 = xr.DataArray(np.ones(shape), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    v10 = xr.DataArray(np.ones(shape), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    base = xr.Dataset({"10m_u_component_of_wind": u10, "10m_v_component_of_wind": v10})

    path = tmp_path / "pfi"
    path.mkdir()
    base.to_netcdf(path / "temp_run0.nc")
    base.to_netcdf(path / "temp_run1.nc")

    result = run_pfi.evaluate_pfi_folder(reference=base, target=base, folder=str(path))
    assert isinstance(result, pd.DataFrame)
    assert "mean_rmse" in result.columns