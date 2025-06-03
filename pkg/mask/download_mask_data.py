import os
import cdsapi
from typing import Sequence, Union
from pkg.gcs_utils import client as gcs
import zipfile
import xarray as xr

def download_era5_for_climatology(
    *,
    years: Sequence[Union[int,str]],
    month: int,
    day: int,
    times: Sequence[str],
    levels: int = 13,
    grid: float = 1.0,
    area: Sequence[float] = (90, -180, -90, 180),
    out_dir: str = "data/mask_era5",
    upload_to_gcs: bool = False,
) -> str:
    """
    Download ERA5 single-level and pressure-level data for a specific date across multiple years.

    This function fetches ERA5 reanalysis data (surface and pressure-level variables) for a fixed
    calendar day across a list of years, which can be used to compute daily climatologies.

    Parameters
    ----------
    years : Sequence[int or str]
        List of years to download (e.g. range(1979, 2016) or ['1979', '1980', ...]).
    month : int
        Month of the target date (1–12).
    day : int
        Day of the target date (1–31).
    times : Sequence[str]
        Times of day (UTC) to download (e.g., ['10:00', '16:00']).
    levels : int, optional
        Number of vertical pressure levels to download. Must be 13 or 37. Default is 13.
    grid : float, optional
        Horizontal grid spacing in degrees. Must be 0.25 or 1.0. Default is 0.25.
    area : Sequence[float], optional
        Bounding box as [North, West, South, East] in degrees. Default is global.
    out_dir : str, optional
        Directory to save the output NetCDF and ZIP files. Default is 'data/mask_era5'.
    upload_to_gcs : bool, optional
        If True, upload the output files to GCS under `mask_era5/`.

    Returns
    -------
    str
        Path to the output directory containing the downloaded files.
    """
    # validate
    assert levels in (13, 37), "levels must be 13 or 37"
    assert grid in (0.25, 1.0), "grid must be 0.25 or 1.0"

    # pressure‐level lists
    plevels_13 = ["50","100","150","200","250","300","400","500","600","700","850","925","1000"]
    plevels_37 = [str(l) for l in (
        1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,350,
        400,450,500,550,600,650,700,750,775,800,825,850,875,900,925,950,
        975,1000
    )]
    pressure_levels = plevels_13 if levels == 13 else plevels_37

    os.makedirs(out_dir, exist_ok=True)
    client = cdsapi.Client()

    for yr in years:
        ys = str(yr)
        tag_times = "_".join(t.replace(":", "") for t in times)
        base_fn = f"era5_{ys}_{month:02d}{day:02d}_{tag_times}_{levels}lev_{str(grid).replace('.','p')}deg"

        # 1) surface file
        surf_fn = f"{base_fn}_surface.zip"
        surf_path = os.path.join(out_dir, surf_fn)
        if not os.path.exists(surf_path):
            print(f"⬇️  [{ys}] surface → {surf_fn}")
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format":       "netcdf",
                    "variable": [
                        "2m_temperature",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "mean_sea_level_pressure",
                        "total_precipitation"
                    ],
                    "year":   ys,
                    "month":  f"{month:02d}",
                    "day":    f"{day:02d}",
                    "time":   list(times),
                    "area":   list(area),
                    "grid":   [grid, grid],
                },
                surf_path
            )
            if upload_to_gcs:
                gcs.upload_file(surf_path, f"mask_era5/{surf_fn}")

        # 2) pressure-level file
        pres_fn = f"{base_fn}_pressure.nc"
        pres_path = os.path.join(out_dir, pres_fn)
        if not os.path.exists(pres_path):
            print(f"⬇️  [{ys}] pressure → {pres_fn}")
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "format":       "netcdf",
                    "variable": [
                        "temperature",
                        "u_component_of_wind",
                        "v_component_of_wind",
                        "geopotential",
                        "specific_humidity",
                        "vertical_velocity"
                    ],
                    "pressure_level": pressure_levels,
                    "year":           ys,
                    "month":          f"{month:02d}",
                    "day":            f"{day:02d}",
                    "time":           list(times),
                    "area":           list(area),
                    "grid":           [grid, grid],
                },
                pres_path
            )
            if upload_to_gcs:
                gcs.upload_file(pres_path, f"mask_era5/{pres_fn}")

    print("✅  All years downloaded to", out_dir)


def unzip_surface_archives(folder: str, suffix: str = "_surface.zip") -> None:
    """
    Unzip and merge ERA5 surface-level NetCDF files from ZIP archives.

    For each ZIP file in the given folder matching the suffix (e.g. *_surface.zip), this function
    extracts all `.nc` members, merges them into a single dataset, and writes the result to a 
    single NetCDF file.

    Parameters
    ----------
    folder : str
        Directory containing ERA5 ZIP files (e.g., output of `download_era5_for_climatology`).
    suffix : str, optional
        Suffix to identify surface ZIP files. Default is "_surface.zip".

    Returns
    -------
    None
        Writes `.nc` files into the same folder. Does not return a value.
    """
    for fname in os.listdir(folder):
        if fname.endswith(suffix):
            zip_path = os.path.join(folder, fname)
            tag = fname.replace(".zip", "")
            out_path = os.path.join(folder, f"{tag}.nc")

            print(f"🔓 Unzipping and merging {fname} → {tag}.nc")

            with zipfile.ZipFile(zip_path, 'r') as zf:
                nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
                datasets = []
                for member in nc_members:
                    with zf.open(member) as src:
                        tmp_path = os.path.join(folder, f"__tmp_{os.path.basename(member)}")
                        with open(tmp_path, "wb") as tmp:
                            tmp.write(src.read())
                        ds = xr.open_dataset(tmp_path).rename({
                            "latitude": "lat", "longitude": "lon", "valid_time": "time"
                        })
                        datasets.append(ds)

                merged = xr.merge(datasets, compat="override")
                merged.to_netcdf(out_path)
                print(f"✅ Merged and saved to {out_path}")

                # Clean up temp files
                for member in nc_members:
                    tmp_path = os.path.join(folder, f"__tmp_{os.path.basename(member)}")
                    os.remove(tmp_path)

def download_era5_for_climatology_with_unzipping(
    years,
    month,
    day,
    times,
    levels=13, 
    grid=1.0, 
    out_dir="data/mask_era5",
    upload_to_gcs=False
):
    """
    Download and unzip ERA5 reanalysis data (surface and pressure-level) for a specific day across years.

    This is a convenience wrapper that combines:
      1. Downloading ERA5 data across years.
      2. Unzipping and merging the surface ZIP files into NetCDF format.

    Parameters
    ----------
    years : Sequence[int or str]
        List of years to download.
    month : int
        Month of the target day.
    day : int
        Day of the month.
    times : Sequence[str]
        Times of day to download (e.g. ['10:00', '16:00']).
    out_dir : str, optional
        Directory to store downloaded and extracted files. Default is 'data/mask_era5'.
    upload_to_gcs : bool, optional
        Whether to upload the downloaded files to GCS. Default is False.

    Returns
    -------
    None
        Data is downloaded and unzipped in-place. Does not return a value.
    """
    download_era5_for_climatology(
        years=years,
        month=month,
        day=day,
        times=times,
        levels=levels, 
        grid=grid, 
        out_dir=out_dir,
        upload_to_gcs=upload_to_gcs
    )
    unzip_surface_archives(folder=out_dir)