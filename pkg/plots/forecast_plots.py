import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Union


def plot_wind_speed_comparison(
    prediction: str | xr.Dataset,
    input_data: str | xr.Dataset,
    lon_min: float = 7.625,
    lon_max: float = 10.625,
    lat_min: float = 53.125,
    lat_max: float = 55.125,
    time_index: int = -1,
    time_label: str = None,
    v_min: float = None,
    v_max: float = None
):
    """
    Plot observed vs. predicted wind speed over a specified region.

    Parameters
    ----------
    prediction: str | xr.Dataset
        Path to the prediction NetCDF file or the dataset itself.
    input_data: str | xr.Dataset
        Path to the ERA5 ground truth NetCDF file or the dataset itself.
    lon_min, lon_max, lat_min, lat_max: float
        Bounding box coordinates.
    time_index: int
        Index for the time step to compare (default: -1 for last).
    time_label: str, optional
        Label for the time in plot titles.
    v_min, v_max: float, optional
        Minimum and maximum values for the color scale.
        If None, they will be determined from the data.
    """

    # Load datasets
    if isinstance(prediction, str):
        with xr.open_dataset(prediction) as ds:
            predictions = ds.load()
    else:
        predictions = prediction

    if isinstance(input_data, str):
        with xr.open_dataset(input_data) as ds:
            example_batch = ds.load()
    else:
        example_batch = input_data

    # Select wind components at desired time index
    u_true = example_batch["10m_u_component_of_wind"].isel(time=time_index).squeeze("batch")
    v_true = example_batch["10m_v_component_of_wind"].isel(time=time_index).squeeze("batch")
    wind_true = np.sqrt(u_true**2 + v_true**2)

    u_pred = predictions["10m_u_component_of_wind"].isel(time=time_index).squeeze("batch")
    v_pred = predictions["10m_v_component_of_wind"].isel(time=time_index).squeeze("batch")
    wind_pred = np.sqrt(u_pred**2 + v_pred**2)

    # Region slice
    region = dict(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    true_region = wind_true.sel(**region)
    pred_region = wind_pred.sel(**region)

    # Color scale (Auto-scale if not provided)
    if v_min is None or v_max is None:
        combined = xr.concat([true_region, pred_region], dim="concat")
        if v_min is None:
            v_min = float(combined.min())
        if v_max is None:
            v_max = float(combined.max())

    vmin, vmax = v_min, v_max

    time_label = time_label or "Unknown UTC"
    titles = [f"Observed Wind Speed at {time_label}", f"Predicted Wind Speed at {time_label}"]

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={'projection': ccrs.Mercator()})


    for ax, data, title in zip(axs, [true_region, pred_region], titles):
        im = data.plot.pcolormesh(
            ax=ax, cmap="viridis", add_colorbar=False, transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax
        )
        ax.set_title(title)

        # Round to avoid floating-point artifacts
        extent = [round(lon_min, 3), round(lon_max, 3), round(lat_min, 3), round(lat_max, 3)]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}

        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3)

        # Add city markers
        city_coords = {
            "Hamburg": (10.0, 53.55),
            "Kiel": (10.1228, 54.3233),
            "Flensburg": (9.4469, 54.7939),
            "Cuxhaven": (8.6942, 53.8647),
            "Büsum": (8.8581, 54.1291),
            "Bremen": (8.8012, 53.0793),
            "Bremerhaven": (8.5809, 53.5396),
        }

        for city, (lon, lat) in city_coords.items():
            ax.plot(lon, lat, 'wo', markersize=5, transform=ccrs.PlateCarree())
            ax.text(lon + 0.05, lat, city, color='white', fontsize=9, transform=ccrs.PlateCarree())

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.025, pad=0.02)
    cbar.set_label("10m Wind Speed (m/s)")

    plt.subplots_adjust(wspace=0.15, right=0.87)
    plt.show()



def evaluate_wind_prediction(
    era5_data: Union[str, xr.Dataset],
    prediction_data: Union[str, xr.Dataset],
    region_lat: tuple[float, float] = (55.0, 53.0),
    region_lon: tuple[float, float] = (8.0, 10.0),
    era5_hour: int = 22
) -> dict:
    """
    Compare 10m wind speed prediction to ERA5 over a specified region and time.

    Parameters
    ----------
    era5_data : str or xr.Dataset
        ERA5 dataset or path to NetCDF file.
    prediction_data : str or xr.Dataset
        Forecast dataset or path to NetCDF file.
    region_lat : tuple
        Latitude bounds (north, south), e.g., (55.0, 53.0).
    region_lon : tuple
        Longitude bounds (west, east), e.g., (8.0, 10.0).
    era5_hour : int
        UTC hour from ERA5 to compare against (default: 22 for 22:00 UTC).

    Returns
    -------
    dict
        Dictionary with MAE, RMSE, and grid metadata.
    """

    # Load data if paths
    if isinstance(era5_data, str):
        era5 = xr.open_dataset(era5_data).load()
    else:
        era5 = era5_data.load()

    if isinstance(prediction_data, str):
        pred = xr.open_dataset(prediction_data).load()
    else:
        pred = prediction_data.load()

    # Find ERA5 time index for the specified hour
    time_sel = int(np.where(era5.time.dt.hour == era5_hour)[0][0])

    # Compute ERA5 wind speed at selected hour
    u_obs = era5["10m_u_component_of_wind"].isel(time=time_sel)
    v_obs = era5["10m_v_component_of_wind"].isel(time=time_sel)
    wind_obs = np.sqrt(u_obs**2 + v_obs**2)

    # Predicted wind speed (assumed single timestep at time=0)
    u_pred = pred["10m_u_component_of_wind"].isel(time=0)
    v_pred = pred["10m_v_component_of_wind"].isel(time=0)
    wind_pred = np.sqrt(u_pred**2 + v_pred**2)

    # Select region
    lat_slice = slice(*region_lat)
    lon_slice = slice(*region_lon)
    obs_region = wind_obs.sel(lat=lat_slice, lon=lon_slice)
    pred_region = wind_pred.sel(lat=lat_slice, lon=lon_slice)

    # Error computation
    error = pred_region - obs_region
    mae = np.abs(error).mean(skipna=True).item()
    rmse = np.sqrt((error**2).mean(skipna=True)).item()

    # Diagnostics
    grid_info = {
        "mae": mae,
        "rmse": rmse,
        "lat_points": obs_region.sizes["lat"],
        "lon_points": obs_region.sizes["lon"],
        "total_points": obs_region.size
    }

    print(f"Mean Absolute Error (MAE):  {mae:.2f} m/s")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} m/s")
    print(f"Grid: {grid_info['lat_points']} lat × {grid_info['lon_points']} lon "
          f"= {grid_info['total_points']} points")

    return grid_info
