import xarray as xr
from typing import Optional, List


def create_feature_mask(
    ds: xr.Dataset,
    variables: Optional[List[str]] = None,
    pressure_levels: Optional[List[int]] = None,
    regions: Optional[List[str]] = None,
    time_steps: Optional[List[int]] = None
) -> xr.Dataset:
    """
    Generate a binary mask (1=masked, 0=keep) for specified variables, pressure levels, regions, and time steps.
    All conditions act exclusively, i.e., only data points that match all criteria are masked.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to be masked.
    variables : list of str, optional
        Names of variables to mask.
    pressure_levels : list of int, optional
        Pressure levels to mask (only for 4D variables).
    regions : list of str, optional
        Named spatial regions to mask (e.g., 'north_germany').
    time_steps : list of int, optional
        Time indices to mask (e.g., [0], [-1], or [0, -1]).

    Returns
    -------
    xr.Dataset
        Dataset with binary mask (same shape as ds), 1 where data should be masked.
    """
    mask = xr.zeros_like(ds, dtype="int")

    for var in ds.data_vars:
        if variables and var not in variables:
            continue

        var_mask = xr.ones_like(ds[var], dtype="int")

        # --- Pressure level masking ---
        if pressure_levels:
            level_dim = None
            for dim_name in ["pressure_level", "level"]:
                if dim_name in ds[var].dims:
                    level_dim = dim_name
                    break

            if level_dim:
                level_mask = ds[level_dim].isin(pressure_levels)
                level_mask_expanded = level_mask
                for dim in ds[var].dims:
                    if dim != level_dim:
                        level_mask_expanded = level_mask_expanded.broadcast_like(ds[var])
                var_mask = var_mask.where(level_mask_expanded, 0)

        # --- Region masking ---
        if regions and {"lat", "lon"}.issubset(ds[var].dims):
            spatial_combined = xr.full_like(ds[var], True, dtype=bool)
            for region in regions:
                region_mask = get_region_mask(ds, region)
                region_mask_expanded = region_mask
                for dim in ds[var].dims:
                    if dim not in {"lat", "lon"}:
                        region_mask_expanded = region_mask_expanded.broadcast_like(ds[var])
                spatial_combined &= region_mask_expanded
            var_mask = var_mask.where(spatial_combined, 0)

        # --- Time step masking ---
        if time_steps and "time" in ds[var].dims:
            time_mask = xr.zeros_like(ds[var], dtype=bool)
            for t in time_steps:
                time_mask |= ds[var].time == ds[var].time[t]
            var_mask = var_mask.where(time_mask, 0)

        mask[var] = var_mask

    return mask



def get_region_mask(ds: xr.Dataset, region: str) -> xr.DataArray:
    """
    Return a lat/lon mask for a named region.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with lat/lon coordinates.
    region : str
        Name of the predefined region.

    Returns
    -------
    xr.DataArray
        Boolean mask array for the region.
    """

    region_bounds = {
        "atlantic_jet_stream":     (60.0, 280.0, 40.0, 340.0),  
        "upper_level_jet_stream":  (60.0, 340.0, 45.0, 15.0),   
        "storm_core_10utc":        (58.0, 352.5, 54.0, 359.0),  
        "storm_core_16utc":        (58.0, 358.0, 54.0, 8.0),    
        "storm_region":            (60.0, 350.0, 45.0, 10.0),  
        "high_pressure_south":     (45.0, 340.0, 30.0, 30.0),  
        "low_pressure_north":      (70.0, 340.0, 60.0, 30.0),  
        "north_germany":           (60.0, 10.0, 52.0, 18.0),  
    }

    if region not in region_bounds:
        raise ValueError(f"Unknown region: {region}")

    N, W, S, E = region_bounds[region]

    lat_mask = (ds.lat <= N) & (ds.lat >= S)

    if W <= E:
        lon_mask = (ds.lon >= W) & (ds.lon <= E)
    else:
        # Wrap-around case
        lon_mask = (ds.lon >= W) | (ds.lon <= E)

    return lat_mask & lon_mask

