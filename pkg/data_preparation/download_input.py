import os
import cdsapi
import xarray as xr
import numpy as np
import zipfile
from pkg.gcs_utils import client as gcs


def _make_folder(name):
    """
    Create a directory if it does not exist.

    Args:
        name (str): Path to the folder.
    """
    os.makedirs(name, exist_ok=True)


def _download_pressure_data(client, output_path, year, month, day, times, levels, resolution):
    """
    Download ERA5 pressure-level data from CDS.

    Args:
        client (cdsapi.Client): CDS API client.
        output_path (str): Path to save NetCDF file.
        year (str): Year as "YYYY".
        month (str): Month as "MM".
        day (str): Day as "DD".
        times (List[str]): List of times (e.g., ["00:00", "12:00"]).
        levels (List[str]): Pressure levels to retrieve.
        resolution (float): Grid resolution in degrees.
    """
    print("📥 Downloading pressure-level data...")
    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'temperature', 'u_component_of_wind', 'v_component_of_wind',
                'geopotential', 'vertical_velocity', 'specific_humidity'
            ],
            'pressure_level': levels,
            'year': year, 'month': month, 'day': day,
            'time': times,
            'grid': [resolution, resolution],
            'format': 'netcdf',
        },
        output_path
    )


def _download_surface_data(client, output_path, year, month, day, times, resolution):
    """
    Download ERA5 single-level surface data from CDS.

    Args:
        client (cdsapi.Client): CDS API client.
        output_path (str): File path for the downloaded ZIP.
        year (str): Year as "YYYY".
        month (str): Month as "MM".
        day (str): Day as "DD".
        times (List[str]): List of times.
        resolution (float): Grid resolution in degrees.
    """
    print("📥 Downloading surface-level data...")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature', '10m_u_component_of_wind',
                '10m_v_component_of_wind', 'mean_sea_level_pressure',
                'total_precipitation'
            ],
            'year': year, 'month': month, 'day': day,
            'time': times,
            'grid': [resolution, resolution],
            'format': 'netcdf',
        },
        output_path
    )


def _extract_surface_zip(zip_path, extract_to):
    """
    Extract a ZIP archive containing surface data.

    Args:
        zip_path (str): Path to the ZIP file.
        extract_to (str): Directory to extract the contents to.
    """
    print("Extracting surface ZIP...")
    _make_folder(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def _process_datasets(pressure_path, extract_dir):
    """
    Load and preprocess pressure-level and surface-level datasets.

    Args:
        pressure_path (str): Path to pressure-level NetCDF file.
        extract_dir (str): Directory with extracted surface-level files.

    Returns:
        Tuple[xr.Dataset, xr.Dataset]: Pressure and surface datasets with aligned dimensions.
    """
    pressure_ds = xr.open_dataset(pressure_path, engine="netcdf4")
    pressure_ds = pressure_ds.rename({
        "valid_time": "time", "latitude": "lat", "longitude": "lon", "pressure_level": "level"
    }).expand_dims("batch")

    instant_ds = xr.open_dataset(f"{extract_dir}/data_stream-oper_stepType-instant.nc", engine="netcdf4")
    accum_ds = xr.open_dataset(f"{extract_dir}/data_stream-oper_stepType-accum.nc", engine="netcdf4")
    surface_ds = xr.merge([instant_ds, accum_ds]).rename({
        "valid_time": "time", "latitude": "lat", "longitude": "lon"
    }).expand_dims("batch")
    surface_ds["time"] = pressure_ds["time"]

    return pressure_ds, surface_ds


def _add_land_sea_mask(client, ds, year, month, day, resolution, output_path):
    """
    Add land-sea mask to the dataset.

    Args:
        client (cdsapi.Client): CDS API client.
        ds (xr.Dataset): Dataset to augment with land-sea mask.
        year (str): Year of data.
        month (str): Month of data.
        day (str): Day of data.
        resolution (float): Spatial resolution.
        output_path (str): File path to save the downloaded mask.
    """
    print("📥 Downloading land-sea mask...")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['land_sea_mask'],
            'year': year, 'month': month, 'day': day,
            'time': ['00:00'],
            'grid': [resolution, resolution],
            'format': 'netcdf',
        },
        output_path
    )
    lsm_ds = xr.open_dataset(output_path)
    lsm_ds = lsm_ds.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
    lsm = lsm_ds["lsm"].isel(time=0).squeeze()
    ds["land_sea_mask"] = lsm


def _combine_and_finalize(pressure_ds, surface_ds):
    """
    Merge pressure and surface datasets and apply final transformations.

    Args:
        pressure_ds (xr.Dataset): Pressure-level dataset.
        surface_ds (xr.Dataset): Surface-level dataset.

    Returns:
        xr.Dataset: Combined and cleaned dataset ready for GraphCast.
    """
    ds = xr.merge([pressure_ds, surface_ds]).rename({
        "t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
        "z": "geopotential", "w": "vertical_velocity", "q": "specific_humidity",
        "t2m": "2m_temperature", "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind", "msl": "mean_sea_level_pressure",
        "tp": "total_precipitation_6hr"
    }).drop_vars(["number", "expver"], errors="ignore")

    time_coord = ds["time"]
    ds = ds.assign_coords(datetime=xr.DataArray(
        np.broadcast_to(time_coord.values, (ds.sizes["batch"], len(time_coord))),
        dims=("batch", "time")
    ))

    geop_surface = ds["geopotential"].sel(level=1000).isel(time=0)
    if "time" in geop_surface.dims:
        geop_surface = geop_surface.isel(time=0)
    ds["geopotential_at_surface"] = geop_surface

    return ds


def prepare_graphcast_input(
    date: str,
    times: list[str],
    levels: int = 13,
    resolution: float = 0.25,
    name: str = None,
    upload_to_gcs: bool = False
):
    """
    Download and prepare ERA5 data as GraphCast-ready input.

    Args:
        date (str): Date in "YYYY-MM-DD" format.
        times (List[str]): List of forecast times (e.g., ["00:00", "12:00"]).
        levels (int): Number of pressure levels (13 or 37).
        resolution (float): Grid resolution (0.25 or 1.0).
        name (Optional[str]): Optional custom name for dataset (used in file naming).
        upload_to_gcs (bool): Whether to upload the resulting NetCDF file to GCS.
    """
    assert levels in [13, 37], "Only 13 or 37 pressure levels supported."
    assert resolution in [0.25, 1.0], "Only 0.25 or 1.0 degree resolution supported."

    pressure_levels = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000'] if levels == 13 else [
        str(l) for l in [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500,
                         550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]]

    year, month, day = date.split("-")
    time_tag = "-".join(t.replace(":", "") for t in times)
    res_tag = f"res{int(resolution * 100)}"
    tag = f"{date}-{levels}lev-{time_tag}-{res_tag}"
    dataset_name = name if name else tag
    folder = f"data/input_data/{dataset_name}"
    _make_folder(folder)

    c = cdsapi.Client()

    _download_pressure_data(c, f"{folder}/era5_pressure.nc", year, month, day, times, pressure_levels, resolution)
    _download_surface_data(c, f"{folder}/era5_surface.zip", year, month, day, times, resolution)
    _extract_surface_zip(f"{folder}/era5_surface.zip", f"{folder}/surface_extracted")

    pressure_ds, surface_ds = _process_datasets(f"{folder}/era5_pressure.nc", f"{folder}/surface_extracted")
    combined_ds = _combine_and_finalize(pressure_ds, surface_ds)
    _add_land_sea_mask(c, combined_ds, year, month, day, resolution, f"{folder}/land_sea_mask.nc")

    filename = f"graphcast_ready_input_{dataset_name}.nc"
    local_file = f"{folder}/{filename}"
    combined_ds.to_netcdf(local_file)
    print(f"✅ Saved locally: {local_file}")

    if upload_to_gcs:
        gcs_path = f"input_data/{dataset_name}/{filename}"
        gcs.upload_file(local_file, gcs_path)
        print(f"☁️  Uploaded to GCS: gs://{gcs.BUCKET_NAME}/{gcs_path}")

