import numpy as np
import xarray as xr
from unittest import mock
from pkg.data_preparation import download_input


def test_make_folder_creates_directory():
    with mock.patch("os.makedirs") as makedirs_mock:
        download_input._make_folder("some/path")
        makedirs_mock.assert_called_once_with("some/path", exist_ok=True)


@mock.patch("pkg.data_preparation.download_input._make_folder")
@mock.patch("zipfile.ZipFile")
def test_extract_surface_zip(mock_zipfile_cls, mock_make_folder):
    mock_zip = mock.MagicMock()
    mock_zipfile_cls.return_value.__enter__.return_value = mock_zip

    download_input._extract_surface_zip("file.zip", "output/")
    mock_make_folder.assert_called_once_with("output/")
    mock_zip.extractall.assert_called_once_with("output/")


def test_download_pressure_data_calls_retrieve():
    client = mock.MagicMock()
    download_input._download_pressure_data(
        client, "output.nc", "2024", "05", "10", ["00:00", "06:00"], ["1000", "850"], 0.25
    )
    client.retrieve.assert_called_once()
    args, _ = client.retrieve.call_args
    request_dict = args[1]
    assert request_dict["format"] == "netcdf"
    assert "temperature" in request_dict["variable"]


def test_download_surface_data_calls_retrieve():
    client = mock.MagicMock()
    download_input._download_surface_data(
        client, "output.zip", "2024", "05", "10", ["00:00", "06:00"], 0.25
    )
    client.retrieve.assert_called_once()
    args, _ = client.retrieve.call_args
    request_dict = args[1]
    assert "2m_temperature" in request_dict["variable"]


def test_combine_and_finalize_merges_and_transforms():
    pressure = xr.Dataset({
        "t": (("batch", "time", "level", "lat", "lon"), np.random.rand(1, 2, 2, 2, 2)),
        "u": (("batch", "time", "level", "lat", "lon"), np.random.rand(1, 2, 2, 2, 2)),
        "v": (("batch", "time", "level", "lat", "lon"), np.random.rand(1, 2, 2, 2, 2)),
        "z": (("batch", "time", "level", "lat", "lon"), np.random.rand(1, 2, 2, 2, 2)),
        "w": (("batch", "time", "level", "lat", "lon"), np.random.rand(1, 2, 2, 2, 2)),
        "q": (("batch", "time", "level", "lat", "lon"), np.random.rand(1, 2, 2, 2, 2)),
        "time": (("time",), [0, 1]),
        "level": [1000, 850],
    })

    surface = xr.Dataset({
        "t2m": (("batch", "time", "lat", "lon"), np.random.rand(1, 2, 2, 2)),
        "u10": (("batch", "time", "lat", "lon"), np.random.rand(1, 2, 2, 2)),
        "v10": (("batch", "time", "lat", "lon"), np.random.rand(1, 2, 2, 2)),
        "msl": (("batch", "time", "lat", "lon"), np.random.rand(1, 2, 2, 2)),
        "tp": (("batch", "time", "lat", "lon"), np.random.rand(1, 2, 2, 2)),
        "time": (("time",), [0, 1])
    })

    ds = download_input._combine_and_finalize(pressure, surface)
    assert "temperature" in ds
    assert "2m_temperature" in ds
    assert "geopotential_at_surface" in ds
    assert "datetime" in ds.coords


@mock.patch("pkg.data_preparation.download_input._make_folder")
@mock.patch("pkg.data_preparation.download_input._download_pressure_data")
@mock.patch("pkg.data_preparation.download_input._download_surface_data")
@mock.patch("pkg.data_preparation.download_input._extract_surface_zip")
@mock.patch("pkg.data_preparation.download_input.xr.open_dataset")
@mock.patch("pkg.data_preparation.download_input.xr.merge")
@mock.patch("pkg.data_preparation.download_input.xr.concat")
@mock.patch("pkg.data_preparation.download_input._combine_and_finalize")
@mock.patch("pkg.data_preparation.download_input._add_land_sea_mask")
@mock.patch("pkg.data_preparation.download_input.gcs.upload_file")
@mock.patch("pkg.data_preparation.download_input.cdsapi.Client")
def test_prepare_graphcast_input_minimal(
    mock_cds_client, mock_upload, mock_add_mask, mock_combine,
    mock_concat, mock_merge, mock_open, mock_extract,
    mock_download_surface, mock_download_pressure, mock_mkdir
):
    # Minimal fake datasets with valid_time so renames succeed
    fake_pressure = xr.Dataset({
        "valid_time":     (("time",),  [0]),
        "latitude":       (("lat",),   [0]),
        "longitude":      (("lon",),   [0]),
        "pressure_level": (("level",), [1000]),
    })
    fake_surface = xr.Dataset({
        "valid_time": (("time",), [0]),
        "latitude":   (("lat",),  [0]),
        "longitude":  (("lon",),  [0]),
    })

    # open_dataset → pressure, then inst, then acc
    mock_open.side_effect = [fake_pressure, fake_surface, fake_surface]

    # Short-circuit merge & concat
    mock_merge.side_effect  = lambda datasets, compat="override": datasets[0]
    mock_concat.side_effect = lambda datasets, dim="time": datasets[0]

    # combine and mask are no-ops
    mock_combine.return_value = fake_pressure

    # Patch __setitem__ on Dataset to bypass xarray’s merge in surface_ds["time"] = ...
    with mock.patch.object(xr.Dataset, "__setitem__", lambda self, key, value: None), \
        mock.patch.object(xr.Dataset, "to_netcdf", lambda self, path: None):
        download_input.prepare_graphcast_input(
            date="2024-05-10",
            start_time="00:00",
            n_steps=1,
            upload_to_gcs=True
        )

    # Verify that each step in the pipeline was invoked
    mock_download_pressure.assert_called_once()
    mock_download_surface.assert_called_once()
    mock_extract.assert_called_once()
    mock_open.assert_called()
    mock_merge.assert_called()     
    mock_concat.assert_called()    
    mock_combine.assert_called_once()
    mock_add_mask.assert_called_once()
    mock_upload.assert_called_once()




