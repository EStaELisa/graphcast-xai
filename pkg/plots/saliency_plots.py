import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_atmospheric_saliency(
    sal_ds: xr.Dataset,
    input_ds: xr.Dataset,
    varname: str,
    target_level: float,
    time_index: int = -1,
    batch_index: int = 0,
    extent: list = [-25, 45, 30, 75],
    cmap: str = "RdBu_r",
    title: str | None = None
):
    """
    Plot a single‐level atmospheric saliency map over Europe.

    Parameters
    ----------
    sal_ds : xr.Dataset
        Saliency dataset containing '<varname>_saliency' with dims
        (batch, time, level, lat, lon).
    input_ds : xr.Dataset
        Original input dataset, used for real 'level', 'lat', 'lon' coords.
    varname : str
        Base variable name, e.g. "geopotential", "temperature", etc.
    target_level : float
        The pressure level (hPa) to slice, e.g. 500.
    time_index : int
        Which time index to plot (default: -1, the last one).
    batch_index : int
        Which batch index to plot (default: 0).
    extent : list
        [lon_min, lon_max, lat_min, lat_max] for the map extent.
    cmap : str
        Matplotlib colormap.
    title : str or None
        If given, use this as the plot title; otherwise a default is built.
    """
    da = sal_ds[f"{varname}_saliency"]
    # find integer index of the nearest level in input_ds
    levels = input_ds["level"].values
    lvl_idx = int(np.argmin(np.abs(levels - target_level)))

    # slice out the 2D field
    da2d = da.isel(batch=batch_index, time=time_index, level=lvl_idx)

    # re‐attach real lat/lon coords
    da2d = da2d.assign_coords(
        lat=input_ds["lat"],
        lon=input_ds["lon"]
    )

    # build the figure with Cartopy
    fig, ax = plt.subplots(
        figsize=(10,4),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    pcm = da2d.plot.pcolormesh(
        ax=ax,
        x="lon", y="lat",
        cmap=cmap,
        vmin=-np.abs(da2d).max(),
        vmax= np.abs(da2d).max(),
        add_colorbar=False,
        transform=ccrs.PlateCarree()
    )

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":", linewidth=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label(f"∂Wind/∂{varname.replace('_',' ').title()}")

    if title is None:
        title = (
            f"{varname.replace('_',' ').title()} Saliency @ "
            f"{levels[lvl_idx]:.0f} hPa → 10 m Wind"
        )
    ax.set_title(title + f" (t={time_index})")
    plt.tight_layout()
    plt.show()


def plot_aggregated_saliency(
    sal_ds: xr.Dataset,
    input_ds: xr.Dataset,
    varname: str,
    method: str = "mean",
    time_index: int = -1,
    batch_index: int = 0,
    extent: list = [-25, 45, 30, 75],
    cmap: str = "RdBu_r",
    title: str | None = None
):
    """
    Aggregate saliency over all pressure levels and plot the 2D map.

    Parameters
    ----------
    sal_ds : xr.Dataset
        Saliency dataset containing '<varname>_saliency' with dims
        (batch, time, level, lat, lon).
    input_ds : xr.Dataset
        Original input dataset, used for real 'lat', 'lon' coords.
    varname : str
        Base variable name, e.g. "geopotential", "temperature", etc.
    method : str
        Aggregation along the level axis: "mean" or "sum".
    time_index : int
        Which time index to plot (default: -1, the last one).
    batch_index : int
        Which batch index to plot (default: 0).
    extent : list
        [lon_min, lon_max, lat_min, lat_max] for the map extent.
    cmap : str
        Matplotlib colormap.
    title : str or None
        If given, use this as the plot title; otherwise a default is built.
    """
    # pull the saliency
    da = sal_ds[f"{varname}_saliency"]
    
    # aggregate over levels
    if method == "mean":
        da2d = da.mean(dim="level")
    elif method == "sum":
        da2d = da.sum(dim="level")
    else:
        raise ValueError("method must be 'mean' or 'sum'")
    
    # select the desired time & batch slice
    da2d = da2d.isel(batch=batch_index, time=time_index)
    
    # reattach real coords
    da2d = da2d.assign_coords(
        lat=input_ds["lat"],
        lon=input_ds["lon"]
    )
    
    # build the map
    fig, ax = plt.subplots(
        figsize=(10,4),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    vmax = np.abs(da2d).max()
    pcm = da2d.plot.pcolormesh(
        ax=ax, x="lon", y="lat",
        cmap=cmap, vmin=-vmax, vmax=vmax,
        add_colorbar=False,
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":", linewidth=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    agg_label = "Mean" if method=="mean" else "Sum"
    cbar.set_label(f"{agg_label} ∂Wind/∂{varname.replace('_',' ').title()}")
    
    if title is None:
        title = (
            f"{varname.replace('_',' ').title()} Saliency ({agg_label} over levels)"
        )
    ax.set_title(f"{title} – t={time_index}, batch={batch_index}")
    plt.tight_layout()
    plt.show()


def plot_surface_saliency(
    ds_sal: xr.Dataset,
    input_ds: xr.Dataset,
    varname: str,
    time_index: int = -1,
    extent: list = [-25, 45, 30, 75],
    cmap: str = "RdBu_r"
):
    """
    Plot a surface‐only saliency field (no level dim) on a European map.
    
    Parameters
    ----------
    ds_sal : xr.Dataset
      Dataset containing `<varname>_saliency` with dims (batch, time, lat, lon).
    input_ds : xr.Dataset
      Original input, used for lat/lon coords.
    varname : str
      The base variable name, e.g. "2m_temperature" or "mean_sea_level_pressure".
    extent : list
      [lon_min, lon_max, lat_min, lat_max] for map zoom.
    cmap : str
      A matplotlib colormap name.
    """
    da = ds_sal[f"{varname}_saliency"]  # dims: (batch, time, lat, lon)

    # pick batch=0, last time
    da2d = da.isel(batch=0, time=time_index)

    # re‐attach real lat/lon coords
    da2d = da2d.assign_coords(lat=input_ds["lat"], lon=input_ds["lon"])

    # plot
    fig, ax = plt.subplots(
        figsize=(8, 5),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    pcm = da2d.plot.pcolormesh(
        ax=ax,
        x="lon", y="lat",
        cmap=cmap,
        vmin=-np.abs(da2d).max(),
        vmax= np.abs(da2d).max(),
        add_colorbar=False,
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":", linewidth=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label(f"∂Wind/∂{varname.replace('_',' ').title()}")

    ax.set_title(f"{varname.replace('_',' ').title()} Saliency → 10 m Wind (22:00 UTC)")
    plt.tight_layout()
    plt.show()
