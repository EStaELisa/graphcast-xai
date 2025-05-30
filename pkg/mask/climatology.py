import os
import glob
import numpy as np
import xarray as xr
from pkg.gcs_utils import client as gcs


def compute_era5_climatology(
    input_folder: str,
    output_path: str,
    upload_to_gcs: bool = False,
    gcs_folder: str = "mask_climatology"
) -> xr.Dataset:
    """
    Compute ERA5 daily climatology by averaging fixed UTC hours across multiple years.

    This function merges surface and pressure-level ERA5 datasets for the same dates and times
    across years, and computes the mean climatology for each hour (e.g., 10:00 and 16:00 UTC).

    Parameters
    ----------
    input_folder : str
        Directory containing yearly ERA5 NetCDF files (surface and pressure) for a specific day
        and time range (e.g., 0218_1000_1600).
    output_path : str
        Output path for the resulting climatology NetCDF file.
    upload_to_gcs : bool, optional
        If True, uploads the output file to Google Cloud Storage (default is False).
    gcs_folder : str, optional
        GCS folder name to upload the file into. Only used if `upload_to_gcs=True`.

    Returns
    -------
    xr.Dataset
        The computed climatology dataset containing averaged variables per time slot.
        Dimensions are typically: time (e.g., 2 values), pressure_level, lat, lon.
    """

    # Find pressure-level ERA5 files
    pres_pattern = os.path.join(input_folder, "era5_*_pressure.nc")
    pres_files = sorted(glob.glob(pres_pattern))
    if not pres_files:
        raise FileNotFoundError(f"No pressure files found with pattern {pres_pattern}")

    all_years_data = []

    for pres in pres_files:
        # Corresponding surface-level file
        surf = pres.replace("_pressure.nc", "_surface.nc")
        if not os.path.isfile(surf):
            raise FileNotFoundError(f"Missing surface file: {surf}")

        # Load datasets
        p = xr.open_dataset(pres, engine="netcdf4")
        s = xr.open_dataset(surf, engine="netcdf4")

        # Rename coordinates to standard names
        rename_map = {"latitude": "lat", "longitude": "lon", "valid_time": "time"}
        p_rename = {k: v for k, v in rename_map.items() if k in p.coords}
        s_rename = {k: v for k, v in rename_map.items() if k in s.coords}
        if p_rename:
            p = p.rename(p_rename)
        if s_rename:
            s = s.rename(s_rename)

        # Rename surface variable names to CF-compliant
        surface_rename = {
            "t2m": "2m_temperature",
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "msl": "mean_sea_level_pressure",
            "tp": "total_precipitation"
        }
        s = s.rename({k: v for k, v in surface_rename.items() if k in s})

        # Merge datasets and standardize time
        ds = xr.merge([p, s], compat="override").astype("float32")
        try:
            ds = xr.decode_cf(ds)
        except Exception:
            if np.issubdtype(ds.time.dtype, np.integer):
                base_date = np.datetime64("1979-02-18T00:00:00")
                ds["time"] = base_date + ds["time"].astype("timedelta64[h]")

        all_years_data.append(ds)

    # Concatenate along time
    combined = xr.concat(all_years_data, dim="time")

    # Average over the same hour-of-day across years (e.g., 10, 16 UTC)
    climatology = (
        combined.groupby(combined.time.dt.hour)
        .mean("time", keep_attrs=True)
        .rename({"hour": "time"})
    )

    # Replace `time` coordinate with proper datetimes
    base_date = np.datetime64("1979-02-18T00:00")
    climatology = climatology.assign_coords({
        "time": [base_date + np.timedelta64(int(h), "h") for h in climatology.time.values]
    })

    # Add metadata
    climatology.attrs["description"] = (
        "ERA5 climatology computed as mean per hour (e.g., 10:00, 16:00 UTC) across years"
    )

    # Clean up encodings for better portability
    for var in climatology.variables:
        climatology[var].encoding = {}

    # Save to NetCDF
    climatology.to_netcdf(output_path)

    # Optional GCS upload
    if upload_to_gcs:
        fname = os.path.basename(output_path)
        gcs_path = f"{gcs_folder}/{fname}"
        gcs.upload_file(output_path, gcs_path)

    return climatology

