import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pkg.mask.generate_mask import compute_wind_speed

def plot_shap_feature_importance(shap_df_path: str, title: str = "KernelSHAP Feature Attribution"):
    """
    Plot SHAP feature importance from a CSV file with color-coded bars.

    Parameters
    ----------
    shap_df_path : str
        Path to the SHAP values CSV file. Must contain 'feature' and 'shap_value' columns.
    title : str, optional
        Title for the plot.
    """
    shap_df = pd.read_csv(shap_df_path)

    # Pretty names for features
    pretty_names = {
        "2m_temperature": "2m Temperature",
        "10m_u_component_of_wind+10m_v_component_of_wind": "10m Wind Speed",
        "total_precipitation_6hr": "Total Precipitation (6h)",
        "mean_sea_level_pressure": "Mean Sea Level Pressure",
        "temperature": "Temperature",
        "u_component_of_wind+v_component_of_wind": "Wind Speed",
        "geopotential": "Geopotential",
        "vertical_velocity": "Vertical Velocity",
        "specific_humidity": "Specific Humidity",
    }

    # Sort by SHAP value
    plot_df = shap_df.sort_values("shap_value", ascending=False).copy()
    plot_df["label"] = plot_df["feature"].map(pretty_names).fillna(plot_df["feature"])

    # Color: blue (positive), red (negative)
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in plot_df["shap_value"]]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        plot_df["label"],
        plot_df["shap_value"],
        color=colors
    )

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("SHAP Contribution to 10 m Wind Speed (m/s)")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()



def compare_mean_wind_speed_input_climatology(
    input_ds: xr.Dataset,
    climatology_ds: xr.Dataset,
    lat_bounds: tuple = (53.0, 55.0),
    lon_bounds: tuple = (8.0, 10.0),
    u_name: str = "10m_u_component_of_wind",
    v_name: str = "10m_v_component_of_wind",
    verbose: bool = True
) -> dict:
    """
    Compare regional mean 10 m wind speed between input and climatology datasets over North Germany (default).

    Parameters
    ----------
    input_ds : xr.Dataset
        Input dataset with 10m wind components.
    climatology_ds : xr.Dataset
        Climatology dataset with 10m wind components.
    lat_bounds : tuple, optional
        Latitude bounds as (min_lat, max_lat).
    lon_bounds : tuple, optional
        Longitude bounds as (min_lon, max_lon).
    u_name : str, optional
        Variable name for u-component of wind.
    v_name : str, optional
        Variable name for v-component of wind.
    verbose : bool, optional
        If True, print the results.

    Returns
    -------
    dict
        Dictionary with mean wind speeds for input and climatology.
    """
    # Compute wind speed
    input_ws = compute_wind_speed(input_ds, u_name=u_name, v_name=v_name, new_name="10m_wind_speed")["10m_wind_speed"]
    clim_ws  = compute_wind_speed(climatology_ds, u_name=u_name, v_name=v_name, new_name="10m_wind_speed")["10m_wind_speed"]

    # Select last timestep
    input_last = input_ws.isel(time=-1)
    clim_last  = clim_ws.isel(time=-1)

    # Drop batch dimension if present
    if "batch" in input_last.dims:
        input_last = input_last.isel(batch=0)
    if "batch" in clim_last.dims:
        clim_last = clim_last.isel(batch=0)

    # Subset region
    lat_cond = (input_last.lat >= lat_bounds[0]) & (input_last.lat <= lat_bounds[1])
    lon_cond = (input_last.lon >= lon_bounds[0]) & (input_last.lon <= lon_bounds[1])

    input_reg = input_last.where(lat_cond & lon_cond, drop=True)
    clim_reg  = clim_last.where(lat_cond & lon_cond, drop=True)

    # Compute mean
    mean_input_ws = input_reg.mean(dim=["lat", "lon"]).item()
    mean_clim_ws  = clim_reg.mean(dim=["lat", "lon"]).item()

    if verbose:
        print(f"North Germany mean 10 m wind at last time (input)      : {mean_input_ws:.3f} m/s")
        print(f"North Germany mean 10 m wind at last time (climatology): {mean_clim_ws:.3f} m/s")

    return {
        "input_mean_wind_speed": mean_input_ws,
        "climatology_mean_wind_speed": mean_clim_ws
    }
