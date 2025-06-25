import os
import cdsapi
import xarray as xr
import numpy as np
import zipfile
from datetime import datetime, timedelta
from pkg.gcs_utils import client as gcs


def _make_folder(name):
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    name (str): 
        Path to the folder to create.
    """
    os.makedirs(name, exist_ok=True)


def _download_pressure_data(client, output_path, year, month, day, times, levels, resolution):
    """
    Download ERA5 pressure-level data from CDS.

    Parameters
    ----------
    client: cdsapi.Client
        The Climate Data Store (CDS) API client.
    output_path: str
        Path to save NetCDF file.
    year: str
        Year as "YYYY".
    month: str
        Month as "MM".
    day: str
        Day as "DD".
    times: List[str]
        List of times (e.g., ["00:00", "12:00"]).
    levels: List[str]
        Pressure levels to retrieve.
    resolution: float
        Grid resolution in degrees.
    """
    print("⬇️ Downloading pressure-level data...")
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

    Parameters
    ----------
    client: cdsapi.Client
        The Climate Data Store (CDS) API client.
    output_path: str
        File path for the downloaded ZIP.
    year: str
        Year as "YYYY".
    month: str
        Month as "MM".
    day: str
        Day as "DD".
    times: List[str]:
        List of times.
    resolution: float
        Grid resolution in degrees.
    """
    print("⬇️ Downloading surface-level data...")
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

    Parameters
    ----------
    zip_path: str
        The path to the ZIP file containing surface data.
    extract_to: str
        Directory to extract the contents to.   
    """
    print("Extracting surface ZIP...")
    _make_folder(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def _add_land_sea_mask(client, ds, year, month, day, resolution, output_path):
    """
    Add land-sea mask to the dataset.

    Parameters
    ----------
    client: cdsapi.Client
        The Climate Data Store (CDS) API client. 
    ds: xr.Dataset
        Dataset to augment with land-sea mask.
    year: str
        Year of data.
    month: str
        Month of data.
    day: str
        Day of data.
    resolution: float
        Spatial resolution.
    output_path: str
        File path to save the downloaded mask.
    """
    print("⬇️ Downloading land-sea mask...")
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

    Parameters
    ----------  
    pressure_ds: xr.Dataset
        Dataset containing pressure-level data.
    surface_ds: xr.Dataset
        Dataset containing surface-level data.  
    
    Returns
    -------
    xr.Dataset
        Combined dataset with renamed variables and additional coordinates.
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
    start_time: str,
    n_steps: int = 3,
    step_hours: int = 6,
    levels: int = 13,
    resolution: float = 0.25,
    output_folder: str = "data/input_data",
    name: str = None,
    upload_to_gcs: bool = False
):
    """
    Download and prepare ERA5 data as GraphCast-ready input,
    allowing forecasts that cross midnight, with customizable output folder.

    Parameters
    ----------
    date: str
        Date in "YYYY-MM-DD" format.
    start_time: str
        Start time in "HH:MM" format.       
    n_steps: int
        Number of forecast steps (default is 3).
    step_hours: int
        Hours between each step (default is 6).
    levels: int
        Number of pressure levels (13 or 37, default is 13).    
    resolution: float
        Spatial resolution in degrees (0.25 or 1.0, default is 0.25).
    output_folder: str  
        Folder to save the prepared input data (default is "data/input_data").
    name: str, optional
        Custom name for the dataset. If not provided, a default name is generated.  
    upload_to_gcs: bool
        Whether to upload the prepared input data to Google Cloud Storage (default is False).
    """
    assert levels in [13, 37], "Only 13 or 37 pressure levels supported."
    assert resolution in [0.25, 1.0], "Only 0.25 or 1.0 degree resolution supported."

    # build list of datetimes
    dt0 = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
    datetimes = [dt0 + i * timedelta(hours=step_hours) for i in range(n_steps)]

    # group times by calendar date
    times_by_date: dict[str, list[str]] = {}
    for dt in datetimes:
        key = dt.strftime("%Y-%m-%d")
        times_by_date.setdefault(key, []).append(dt.strftime("%H:%M"))

    # naming/tagging
    times_tag = "-".join(dt.strftime("%H%M") for dt in datetimes)
    res_tag = f"res{int(resolution * 100)}"
    tag = f"{date}-{levels}lev-{times_tag}-{res_tag}"
    dataset_name = name or tag

    # make base folder
    folder = os.path.join(output_folder, dataset_name)
    _make_folder(folder)

    # download per-day
    c = cdsapi.Client()
    pressure_levels = (
        ['50','100','150','200','250','300','400','500','600','700','850','925','1000']
        if levels == 13 else
        [str(l) for l in [1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,975,1000]]
    )

    for day, times in times_by_date.items():
        year, month, daynum = day.split("-")
        # pressure
        p_out = os.path.join(folder, f"era5_pressure_{day}.nc")
        _download_pressure_data(c, p_out, year, month, daynum, times, pressure_levels, resolution)

        # surface
        s_zip = os.path.join(folder, f"era5_surface_{day}.zip")
        _download_surface_data(c, s_zip, year, month, daynum, times, resolution)
        _extract_surface_zip(s_zip, os.path.join(folder, f"surface_extracted_{day}"))

    # load, rename, expand, and concat
    p_datasets = []
    s_datasets = []
    for day in times_by_date:
        # pressure
        p = xr.open_dataset(os.path.join(folder, f"era5_pressure_{day}.nc"), engine="netcdf4")
        p = p.rename({
            "valid_time": "time", "latitude": "lat", "longitude": "lon", "pressure_level": "level"
        }).expand_dims("batch")
        p_datasets.append(p)

        # surface
        inst = xr.open_dataset(os.path.join(folder, f"surface_extracted_{day}",
                                              "data_stream-oper_stepType-instant.nc"), engine="netcdf4")
        acc = xr.open_dataset(os.path.join(folder, f"surface_extracted_{day}",
                                             "data_stream-oper_stepType-accum.nc"), engine="netcdf4")
        s = xr.merge([inst, acc], compat="override").rename({
            "valid_time": "time", "latitude": "lat", "longitude": "lon"
        }).expand_dims("batch")
        s_datasets.append(s)

    # concat along time
    pressure_ds = xr.concat(p_datasets, dim="time")
    surface_ds = xr.concat(s_datasets, dim="time")
    surface_ds["time"] = pressure_ds["time"]

    # merge, mask, save
    combined = _combine_and_finalize(pressure_ds, surface_ds)
    # use last day for mask metadata
    last = list(times_by_date.keys())[-1].split("-")
    _add_land_sea_mask(c, combined, last[0], last[1], last[2], resolution,
                       os.path.join(folder, "land_sea_mask.nc"))

    filename = f"graphcast_ready_input_{dataset_name}.nc"
    local_file = os.path.join(folder, filename)
    combined.to_netcdf(local_file)
    print(f"✅ Saved locally: {local_file}")

    if upload_to_gcs:
        gcs_path = f"input_data/{dataset_name}/{filename}"
        gcs.upload_file(local_file, gcs_path)

