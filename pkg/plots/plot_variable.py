import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_variable(
    ds,
    time,
    var_name,
    level=None,
    extent=[-25, 45, 30, 75],
    cmap=None,
    title=None
):
    """
    Plot a field from an xarray Dataset (surface or pressure level) on a PlateCarree map.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing surface or pressure-level atmospheric variables.
    time : int or str
        Time index (integer) or a string matching the time coordinate values.
    var_name : str
        Variable name to plot, e.g. 'temperature', '10m_wind_speed', or 'wind_speed'.
    level : float, optional
        Pressure level in hPa (e.g. 850, 500). Ignored for surface variables.
    extent : list
        Map extent as [lon_min, lon_max, lat_min, lat_max].
    cmap : str, optional
        Matplotlib colormap name to use for the plot.
    title : str, optional
        Custom title to override default.
    """

    # Select time
    try:
        da_time = ds.sel(time=time)
        time_str = str(da_time.time.values)[:16]
    except Exception:
        da_time = ds.isel(time=time)
        time_str = str(ds.time.values[time])[:16]

    # Surface variable if no "level" is passed
    if level is not None and "level" in ds.dims:
        da = da_time.sel(level=level)
        level_label = f" at {level} hPa"
    else:
        da = da_time
        level_label = ""

    # Compute wind speed (10m or pressure level) if requested
    if var_name in ["wind_speed", "10m_wind_speed"]:
        u_var = "10m_u_component_of_wind" if "10m" in var_name else "u_component_of_wind"
        v_var = "10m_v_component_of_wind" if "10m" in var_name else "v_component_of_wind"
        u = da[u_var]
        v = da[v_var]
        data = np.sqrt(u**2 + v**2)
        label = "10m Wind Speed (m/s)" if "10m" in var_name else f"Wind Speed{level_label} (m/s)"
    else:
        data = da[var_name]
        label = f"{var_name.replace('_', ' ').capitalize()}{level_label}"

    # Colormap defaults
    cmap_map = {
        'geopotential': 'coolwarm',
        'temperature': 'RdYlBu_r',
        'specific_humidity': 'PuBu',
        'vertical_velocity': 'seismic',
        'u_component_of_wind': 'viridis',
        'v_component_of_wind': 'viridis',
        'wind_speed': 'viridis',
        '10m_wind_speed': 'viridis',
        '2m_temperature': 'RdYlBu_r',
        '10m_u_component_of_wind': 'viridis',
        '10m_v_component_of_wind': 'viridis',
        'total_precipitation_6hr': 'Blues',
        'mean_sea_level_pressure': 'coolwarm'
    }
    default_cmap = cmap_map.get(var_name, 'viridis')
    use_cmap = cmap if cmap is not None else default_cmap

    # Plotting
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    data.plot(
        ax=ax,
        cmap=use_cmap,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={"label": label},
        add_labels=False
    )

    ax.set_title(title or f"{label} — {time_str}")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    plt.tight_layout()
    return fig, ax
