import os
import numpy as np
import xarray as xr
import pytest
from unittest import mock
from pkg.mask.climatology import compute_era5_climatology

@pytest.fixture
def synthetic_era5_dataset(tmp_path):
    """
    Creates matching pressure and surface files with 2 time steps, one per year,
    to simulate climatology input data.
    """
    lat = [50.0]
    lon = [8.0]
    times = np.array(["1979-02-18T10:00", "1980-02-18T10:00"], dtype="datetime64")

    for i, t in enumerate(times):
        year = str(t)[:4]
        pressure = xr.Dataset({
            "temperature": (("time", "level", "lat", "lon"), [[[[280 + i]]]]),
        }, coords={
            "time": [t],
            "level": [850],
            "lat": lat,
            "lon": lon
        })
        pressure.to_netcdf(tmp_path / f"era5_{year}_pressure.nc")

        surface = xr.Dataset({
            "t2m": (("time", "lat", "lon"), [[[270 + i]]]),
        }, coords={
            "valid_time": [t],
            "latitude": lat,
            "longitude": lon
        })
        surface.to_netcdf(tmp_path / f"era5_{year}_surface.nc")

    return tmp_path

def test_compute_era5_climatology_output(synthetic_era5_dataset):
    output_path = synthetic_era5_dataset / "climatology.nc"

    result = compute_era5_climatology(
        input_folder=str(synthetic_era5_dataset),
        output_path=str(output_path)
    )

    assert output_path.exists()
    assert isinstance(result, xr.Dataset)
    assert "2m_temperature" in result.data_vars
    assert "temperature" in result.data_vars
    assert result.sizes["time"] == 1  # Only one hour (10:00)
    assert np.allclose(result["2m_temperature"].values, 270.5)
    assert np.allclose(result["temperature"].values, 280.5)

def test_missing_surface_file_raises(synthetic_era5_dataset):
    # Delete one surface file to simulate error
    for f in os.listdir(synthetic_era5_dataset):
        if f.endswith("_surface.nc"):
            os.remove(synthetic_era5_dataset / f)
            break

    with pytest.raises(FileNotFoundError, match="Missing surface file"):
        compute_era5_climatology(
            input_folder=str(synthetic_era5_dataset),
            output_path=str(synthetic_era5_dataset / "out.nc")
        )

@mock.patch("pkg.mask.climatology.gcs.upload_file")
def test_gcs_upload_triggered(mock_upload, synthetic_era5_dataset):
    output_path = synthetic_era5_dataset / "climatology.nc"

    compute_era5_climatology(
        input_folder=str(synthetic_era5_dataset),
        output_path=str(output_path),
        upload_to_gcs=True,
        gcs_folder="test_folder"
    )

    mock_upload.assert_called_once()
    call_args = mock_upload.call_args[0]
    assert "climatology.nc" in call_args[0]  # local path
    assert "test_folder/climatology.nc" in call_args[1]  # GCS path
