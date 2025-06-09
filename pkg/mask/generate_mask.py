import xarray as xr
import numpy as np
import os
import pickle
from typing import Optional, List


def create_feature_mask(
    ds: xr.Dataset,
    variables: Optional[List[str]] = None,
    pressure_levels: Optional[List[int]] = None,
    regions: Optional[List[str]] = None,
    time_steps: Optional[List[int]] = None
) -> xr.Dataset:
    """
        Build a binary mask (1=keep, 0=masked) over all data_vars in `ds` according to:

        1. `variables`:   which variables to include (default: all).
        2. `pressure_levels`: only mask those pressure/level indices.
        3. `regions`:     spatial regions to mask, including:
            • static regions (e.g. "north_germany", "storm_region", etc.)
            • "dynamic_uljs" (per‐timestep jet coordinates)
            • "storm_core_10utc" or "storm_core_16utc" (fixed‐time cores)
            • the meta‐region "storm_core"
        4. `time_steps`:  which time indices to keep (all others get zeroed out).

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset with dims like ('batch','time','level','lat','lon') or
            ('batch','time','lat','lon').
        variables : list of str, optional
            Only compute masks for these data_vars. Default is all lat/lon‐bearing vars.
        pressure_levels : list of int, optional
            Only apply pressure‐level masks on those levels.
        regions : list of str, optional
            See above for allowed entries and the special "storm_core" behavior.
        time_steps : list of int, optional
            After spatial masking, zero out all time indices *not* in this list—unless
            overridden by the `"storm_core" + time_steps` shortcut.

        Returns
        -------
        xr.Dataset
            A Dataset of 0/1 masks matching the shape and coords of `ds`.

        """

    # ———— (0) Smart expand for storm_core + time_steps ————
    if regions and "storm_core" in regions:
        # remove the meta-entry
        regions = [r for r in regions if r != "storm_core"]
        # if they also specified time_steps, only add the cores they asked for
        if time_steps:
            new = []
            for t in time_steps:
                if t == 0:
                    new.append("storm_core_10utc")
                elif t == 1:
                    new.append("storm_core_16utc")
            regions += new
            # disable the generic time_steps filter—our region logic already did it
            time_steps = None
        else:
            # no time_steps given → add both
            regions += ["storm_core_10utc", "storm_core_16utc"]

    # —————— (1) Precompute ULJS coords ——————
    uljs_coords_by_time = None
    if regions and "dynamic_uljs" in regions:
        cache_path = "uljs_coords_by_time.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                uljs_coords_by_time = pickle.load(f)
        else:
            uljs_coords_by_time = get_uljs_coordinates_by_time(ds)
            with open(cache_path, "wb") as f:
                pickle.dump(uljs_coords_by_time, f)

    # —————— (2) Initialize mask_ds as you had it ——————
    mask_ds = xr.zeros_like(ds, dtype="int")

    for var in ds.data_vars:
        if variables and var not in variables:
            continue

        var_mask = xr.ones_like(ds[var], dtype="int")

        # — Pressure‐level masking (unchanged) —
        if pressure_levels:
            level_dim = next((d for d in ("pressure_level","level")
                              if d in ds[var].dims), None)
            if level_dim:
                lvl_m = ds[level_dim].isin(pressure_levels)
                for d in ds[var].dims:
                    if d!=level_dim:
                        lvl_m = lvl_m.broadcast_like(ds[var])
                var_mask = var_mask.where(lvl_m, 0)

        # — Region‐based masking (unchanged) —
        if regions and {"lat","lon"}.issubset(ds[var].dims):
            spatial_combined = xr.full_like(ds[var], True, dtype=bool)
            for region in regions:
                if region == "dynamic_uljs":
                    # … your ULJS code …
                    pass
                elif region in ("storm_core_10utc","storm_core_16utc"):
                    idx = 0 if region=="storm_core_10utc" else 1
                    if "time" not in ds[var].dims:
                        continue
                    m2d = get_region_mask(ds, region)
                    broad2D = m2d
                    for d in ds[var].dims:
                        if d not in ("lat","lon"):
                            broad2D = broad2D.broadcast_like(ds[var])
                    tf = (ds.time==ds.time.values[idx]).broadcast_like(ds[var])
                    spatial_combined &= ( ~tf ) | broad2D
                else:
                    # … your other static‐region code …
                    pass
            var_mask = var_mask.where(spatial_combined, 0)

        # — Time‐step masking — only if we didn’t already bake it in —
        if time_steps and "time" in ds[var].dims:
            tmask = xr.zeros_like(ds[var], dtype=bool)
            for t in time_steps:
                tmask |= (ds.time==ds.time[t])
            var_mask = var_mask.where(tmask, 0)

        mask_ds[var] = var_mask

    return mask_ds






def get_region_mask(ds: xr.Dataset, region: str) -> xr.DataArray:
    """
    Return a 2D (lat, lon) boolean mask for a named static region.
    """
    region_bounds = {
        "atlantic_jet_stream":     (60.0, 280.0, 40.0, 340.0),
        "upper_level_jet_stream":  (60.0, 340.0, 45.0, 11.0),
        "storm_core_10utc":        (58.0, 352.5, 54.0, 359.0),
        "storm_core_16utc":        (58.0, 358.0, 55.0, 8.0),
        "storm_region":            (60.0, 350.0, 45.0, 10.0),
        "high_pressure_south":     (45.0, 340.0, 30.0, 30.0),
        "low_pressure_north":      (70.0, 340.0, 60.0, 30.0),
        "north_germany":           (60.0, 10.0, 52.0, 18.0),

        "quad_nw": (60.0, 340.0, 52.5,  0.0),
        "quad_ne": (60.0,   0.0, 52.5, 20.0),
        "quad_sw": (52.5, 340.0, 45.0,  0.0),
        "quad_se": (52.5,   0.0, 45.0, 20.0),
    }

    if region not in region_bounds:
        raise ValueError(f"Unknown region: {region}")

    N, W, S, E = region_bounds[region]
    lat_mask = (ds.lat <= N) & (ds.lat >= S)
    if W <= E:
        lon_mask = (ds.lon >= W) & (ds.lon <= E)
    else:
        lon_mask = (ds.lon >= W) | (ds.lon <= E)

    return lat_mask & lon_mask


def compute_wind_speed(
    ds: xr.Dataset,
    u_name: str = "u_component_of_wind",
    v_name: str = "v_component_of_wind",
    new_name: str = "wind_speed"
) -> xr.Dataset:
    """
    Adds wind_speed = sqrt(u^2 + v^2) into ds.
    """
    u = ds[u_name]
    v = ds[v_name]
    ds[new_name] = np.sqrt(u**2 + v**2)
    return ds


def get_uljs_coordinates_by_time(
    ds: xr.Dataset,
    wind_threshold: float = 70.0,
    target_level: float = 300.0
) -> dict:
    """
    Returns a dictionary {time_index: [(lat_idx, lon_idx), …]} for each time slice
    where wind_speed (at target_level) exceeds the threshold, restricted to a lat/lon box.
    """
    ds = compute_wind_speed(ds)
    lvl_idx = int(np.argmin(np.abs(ds.level.values - target_level)))
    wind = ds["wind_speed"].isel(level=lvl_idx, batch=0)  # dims: (time, lat, lon)

    lat_mask = (ds.lat <= 60.0) & (ds.lat >= 45.0)
    lon_mask = (ds.lon >= 340.0) | (ds.lon <= 11.0)
    geo_box = lat_mask & lon_mask
    geo_box = geo_box.broadcast_like(wind.isel(time=0))

    coords_by_time = {}
    for t in range(wind.sizes["time"]):
        wslice = wind.isel(time=t)
        m = (wslice > wind_threshold) & geo_box
        lat_idxs, lon_idxs = np.where(m.values)
        coords_by_time[t] = list(zip(lat_idxs, lon_idxs))

    return coords_by_time
