import os
import glob
import numpy as np
import xarray as xr
from pkg.gcs_utils import client as gcs


def _postprocess_climatology(climatology: xr.Dataset) -> xr.Dataset:
    """Postprocesses the raw climatology for use in masking workflows."""

    # Rename variable names to match GraphCast naming
    rename_map = {
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "z": "geopotential",
        "q": "specific_humidity",
        "w": "vertical_velocity",
        "total_precipitation": "total_precipitation_6hr"
    }
    climatology = climatology.rename({k: v for k, v in rename_map.items() if k in climatology})

    # Rename dimension to match GraphCast convention
    if "pressure_level" in climatology.dims:
        climatology = climatology.rename({"pressure_level": "level"})

    # Wrap longitudes to 0–360 degrees and sort
    if "lon" in climatology.coords:
        climatology = climatology.assign_coords(lon=(climatology["lon"] % 360))
        climatology = climatology.sortby("lon")

    return climatology


def compute_era5_climatology(
    input_folder: str,
    output_path: str,
    upload_to_gcs: bool = False,
    gcs_folder: str = "mask_climatology"
) -> xr.Dataset:
    """
    Compute ERA5 daily climatology by averaging fixed UTC hours across multiple years.
    """

    pres_pattern = os.path.join(input_folder, "era5_*_pressure.nc")
    pres_files = sorted(glob.glob(pres_pattern))
    if not pres_files:
        raise FileNotFoundError(f"No pressure files found with pattern {pres_pattern}")

    all_years_data = []

    for pres in pres_files:
        surf = pres.replace("_pressure.nc", "_surface.nc")
        if not os.path.isfile(surf):
            raise FileNotFoundError(f"Missing surface file: {surf}")

        p = xr.open_dataset(pres, engine="netcdf4")
        s = xr.open_dataset(surf, engine="netcdf4")

        rename_map = {"latitude": "lat", "longitude": "lon", "valid_time": "time"}
        p = p.rename({k: v for k, v in rename_map.items() if k in p.coords})
        s = s.rename({k: v for k, v in rename_map.items() if k in s.coords})

        surface_rename = {
            "t2m": "2m_temperature",
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "msl": "mean_sea_level_pressure",
            "tp": "total_precipitation"
        }
        s = s.rename({k: v for k, v in surface_rename.items() if k in s})

        ds = xr.merge([p, s], compat="override").astype("float32")
        try:
            ds = xr.decode_cf(ds)
        except Exception:
            if np.issubdtype(ds.time.dtype, np.integer):
                base_date = np.datetime64("1979-02-18T00:00:00")
                ds["time"] = base_date + ds["time"].astype("timedelta64[h]")

        all_years_data.append(ds)

    combined = xr.concat(all_years_data, dim="time")

    # Average over hour of day (e.g., 10, 16, etc.)
    climatology = (
        combined.groupby(combined.time.dt.hour)
        .mean("time", keep_attrs=True)
        .rename({"hour": "time"})
    )

    base_date = np.datetime64("1979-02-18T00:00")
    climatology = climatology.assign_coords({
        "time": [base_date + np.timedelta64(int(h), "h") for h in climatology.time.values]
    })

    # Add metadata
    climatology.attrs["description"] = (
        "ERA5 climatology computed as mean per hour (e.g., 10:00, 16:00 UTC) across years"
    )

    # Clean up encodings
    for var in climatology.variables:
        climatology[var].encoding = {}

    # Apply climatology postprocessing (rename variables, wrap lon, etc.)
    climatology = _postprocess_climatology(climatology)

    climatology.to_netcdf(output_path)

    if upload_to_gcs:
        fname = os.path.basename(output_path)
        gcs_path = f"{gcs_folder}/{fname}"
        gcs.upload_file(output_path, gcs_path)

    return climatology
