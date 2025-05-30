import zipfile
import pytest
import xarray as xr
from unittest import mock
from tempfile import TemporaryDirectory
from pkg.mask.download_era5_mask_data import (
    download_era5_for_climatology,
    unzip_surface_archives,
    download_era5_for_climatology_with_unzipping
)

@pytest.fixture
def fake_zip_dataset(tmp_path):
    """
    Creates a dummy ZIP with NetCDFs mimicking a surface ZIP file.
    """
    ds = xr.Dataset(
        {"t2m": (("time", "lat", "lon"), [[[280.0]]])},
        coords={
            "latitude": [50.0],
            "longitude": [8.0],
            "valid_time": ["1979-02-18T10:00"]
        }
    )
    nc_path = tmp_path / "dummy.nc"
    ds.to_netcdf(nc_path)

    zip_path = tmp_path / "era5_1979_0218_1000_13lev_025deg_surface.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(nc_path, arcname="member.nc")

    return zip_path, tmp_path

def test_unzip_surface_archives(fake_zip_dataset):
    zip_path, folder = fake_zip_dataset
    unzip_surface_archives(str(folder))

    expected_nc = folder / "era5_1979_0218_1000_13lev_025deg_surface.nc"
    assert expected_nc.exists()

    ds = xr.open_dataset(expected_nc)
    assert "2m_temperature" in ds or "t2m" in ds

@mock.patch("pkg.mask.download_era5_mask_data.cdsapi.Client")
def test_download_era5_for_climatology_calls_api(mock_client, tmp_path):
    client_instance = mock_client.return_value
    client_instance.retrieve.return_value = None

    download_era5_for_climatology(
        years=["1979"],
        month=2,
        day=18,
        times=["10:00"],
        out_dir=str(tmp_path),
        upload_to_gcs=False
    )

    # Check that surface and pressure files were requested
    assert client_instance.retrieve.call_count == 2
    calls = [call[0][0] for call in client_instance.retrieve.call_args_list]
    assert "reanalysis-era5-single-levels" in calls[0]
    assert "reanalysis-era5-pressure-levels" in calls[1]

@mock.patch("pkg.mask.download_era5_mask_data.download_era5_for_climatology")
@mock.patch("pkg.mask.download_era5_mask_data.unzip_surface_archives")
def test_download_with_unzipping_wrapper(mock_unzip, mock_download):
    download_era5_for_climatology_with_unzipping(
        years=[1979],
        month=2,
        day=18,
        times=["10:00"]
    )
    mock_download.assert_called_once()
    mock_unzip.assert_called_once()
