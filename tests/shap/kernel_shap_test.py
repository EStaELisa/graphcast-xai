import pytest
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import os
from unittest import mock

from pkg.shap.kernel_shap import run_kernel_shap


@pytest.fixture
def dummy_data():
    lat = [53.0, 54.0, 55.0]
    lon = [8.0, 9.0, 10.0]
    time = pd.date_range("2023-01-01", periods=1)
    dims = ("time", "lat", "lon")
    shape = (1, 3, 3)
    coords = {"time": time, "lat": lat, "lon": lon}

    def make_var(val):
        return xr.DataArray(np.full(shape, val, dtype=float), dims=dims, coords=coords)

    input_ds = xr.Dataset({"temp": make_var(280), "v": make_var(10)})
    clim_ds = xr.Dataset({"temp": make_var(275), "v": make_var(0)})

    return input_ds, clim_ds


def test_run_kernel_shap_executes(dummy_data):
    input_ds, clim_ds = dummy_data
    features = ["temp", "v"]

    with tempfile.TemporaryDirectory() as shap_dir:
        with mock.patch("pkg.shap.kernel_shap.run_forecast") as mock_forecast, \
             mock.patch("pkg.shap.kernel_shap.xr.open_dataset") as mock_open, \
             mock.patch("pkg.shap.kernel_shap.compute_wind_speed") as mock_ws:

            mock_forecast.return_value = None

            mock_ds = xr.Dataset({
                "10m_u_component_of_wind": input_ds["v"],
                "10m_v_component_of_wind": input_ds["v"]
            })
            mock_open.return_value = mock_ds

            ws_val = np.sqrt(200)
            wind_array = xr.DataArray(
                np.full((1, 3, 3), ws_val, dtype=float),
                dims=("time", "lat", "lon"),
                coords={"time": input_ds.time, "lat": input_ds.lat, "lon": input_ds.lon}
            )

            call_counter = {"count": 0}

            # Return different SHAP outcomes based on order of call
            def fake_compute_wind_speed(*args, **kwargs):
                if call_counter["count"] == 0:
                    result = xr.Dataset({"10m_wind_speed": wind_array * 0})
                else:
                    result = xr.Dataset({"10m_wind_speed": wind_array})
                call_counter["count"] += 1
                return result

            mock_ws.side_effect = fake_compute_wind_speed

            df = run_kernel_shap(features, input_ds, clim_ds, shap_folder=shap_dir, nsamples=10)

            assert isinstance(df, pd.DataFrame)
            assert set(df.columns) == {"feature", "shap_value"}
            assert all(f in df["feature"].values for f in features)
            assert np.isclose(df["shap_value"].sum(), ws_val, atol=1e-3)
            assert os.path.exists(os.path.join(shap_dir, "shap_values.csv"))
