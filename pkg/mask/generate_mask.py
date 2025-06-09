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
    Generate a binary mask (1=masked, 0=keep) for specified variables, pressure levels,
    regions, and time steps. All conditions act with AND logic, except storm_core_10utc
    only applies at time index 0 and storm_core_16utc only at time index 1.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to be masked. Expected dims: ('batch','time','level','lat','lon')
        for 3D fields, or ('batch','time','lat','lon') for surface fields.
    variables : list of str, optional
        If given, only mask these data_vars; otherwise mask all lat/lon‐bearing data_vars.
    pressure_levels : list of int, optional
        If provided, mask only these levels (4D vars that have a 'level' or 'pressure_level' dim).
    regions : list of str, optional
        Possible entries:
          - "dynamic_uljs"
          - "storm_core_10utc"
          - "storm_core_16utc"
          - any static region defined in get_region_mask (e.g. "storm_region", "north_germany", etc.)
    time_steps : list of int, optional
        If provided, only these time indices are masked in addition to the region logic.

    Returns
    -------
    xr.Dataset
        A new Dataset of the same shape, with 0/1 ints: 1 means “masked,” 0 means “keep.”
    """

    # —————— (1) Precompute dynamic ULJS coords if requested ——————
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

    # —————— (2) Initialize output mask (0 everywhere = “keep”) ——————
    mask_ds = xr.zeros_like(ds, dtype="int")

    for var in ds.data_vars:
        if variables and var not in variables:
            continue

        # Start from “all kept” (1 = keep) and then set parts to 0 to mask.
        var_mask = xr.ones_like(ds[var], dtype="int")

        # —————— (3) Pressure‐level masking (if requested) ——————
        if pressure_levels:
            level_dim = None
            for dim_name in ("pressure_level", "level"):
                if dim_name in ds[var].dims:
                    level_dim = dim_name
                    break
            if level_dim:
                lvl_mask = ds[level_dim].isin(pressure_levels)
                for d in ds[var].dims:
                    if d != level_dim:
                        lvl_mask = lvl_mask.broadcast_like(ds[var])
                var_mask = var_mask.where(lvl_mask, 0)

        # —————— (4) Region‐based masking ——————
        if regions and {"lat", "lon"}.issubset(ds[var].dims):
            spatial_combined = xr.full_like(ds[var], True, dtype=bool)

            for region in regions:
                # —--- (a) Dynamic ULJS (time‐varying) --------------------------
                if region == "dynamic_uljs":
                    if "wind" not in var:
                        # only apply ULJS to wind variables (u or v)
                        continue

                    # Build a full‐shape boolean “dyn_mask,” default False everywhere:
                    dyn_mask = xr.zeros_like(ds[var], dtype=bool)

                    # For each time index, mark True exactly at the ULJS coordinates for that time
                    for t_idx in range(ds.sizes["time"]):
                        coords = uljs_coords_by_time.get(t_idx, [])
                        if not coords:
                            continue
                        lat_idxs, lon_idxs = zip(*coords)
                        tmp = np.zeros((ds.sizes["lat"], ds.sizes["lon"]), dtype=bool)
                        tmp[np.array(lat_idxs), np.array(lon_idxs)] = True
                        da2d = xr.DataArray(
                            tmp,
                            coords={"lat": ds.lat, "lon": ds.lon},
                            dims=("lat", "lon"),
                        )
                        slice_like = ds[var].isel(time=slice(t_idx, t_idx + 1))
                        broad = da2d.broadcast_like(slice_like)
                        dyn_mask.loc[dict(time=ds.time.values[t_idx])] = broad.isel(time=0)


                    spatial_combined &= dyn_mask

                # —--- (b) Storm cores at fixed times --------------------------
                # elif region in ("storm_core_10utc", "storm_core_16utc"):
                #     # storm_core_10utc → only mask at time index 0
                #     # storm_core_16utc → only mask at time index 1
                #     expected_time_index = 0 if region == "storm_core_10utc" else 1
                #     if "time" not in ds[var].dims:
                #         continue

                #     # (i) Build a 2D mask in (lat, lon) for that core
                #     twoD = get_region_mask(ds, region)  # dims: (lat, lon)

                #     # (ii) Broadcast that 2D mask to full dims of ds[var]
                #     broad2D = twoD
                #     for d in ds[var].dims:
                #         if d not in ("lat", "lon"):
                #             broad2D = broad2D.broadcast_like(ds[var])

                #     # (iii) Build a time‐flag that is True only at expected_time_index
                #     timeflag = (ds.time == ds.time.values[expected_time_index])
                #     timeflag_full = timeflag.broadcast_like(ds[var])

                #     # (iv) “Gate” the 2D mask with that time‐flag:
                #     #      At time = expected_time_index → apply broad2D.
                #     #      At all other times → keep everything True.
                #     combined_core_mask = (~timeflag_full) | broad2D

                #     spatial_combined &= combined_core_mask

                
                # —--- (b) Storm core(s) at their times -----------------------
                elif region == "storm_core":
                    # map each index → its fixed core name
                    core_map = {
                        0: "storm_core_10utc",
                        1: "storm_core_16utc",
                    }

                    # decide which time‐indices to apply:
                    # if user passed time_steps=[…], use that; else mask both cores
                    times_to_mask = time_steps if time_steps is not None else list(core_map.keys())

                    # start from “keep everything” (True everywhere)
                    core_keep = xr.full_like(ds[var], True, dtype=bool)

                    for t_idx in times_to_mask:
                        if t_idx not in core_map:
                            continue

                        # (i) 2D mask for this core
                        twoD = get_region_mask(ds, core_map[t_idx])  # dims: (lat, lon)

                        # (ii) broadcast it to full var shape
                        broad2D = twoD
                        for dim in ds[var].dims:
                            if dim not in ("lat", "lon"):
                                broad2D = broad2D.broadcast_like(ds[var])

                        # (iii) pick out the exact time slice
                        this_time = ds.time[t_idx]
                        timeflag = (ds.time == this_time).broadcast_like(ds[var])

                        # (iv) at that time → mask where broad2D == True
                        #      elsewhere → keep True
                        keep_here = ~(timeflag & broad2D)

                        # (v) accumulate: only points that survive all cores
                        core_keep &= keep_here

                    # finally carve out the storm_core mask
                    spatial_combined &= core_keep



                # —--- (c) Static regions (all other names) --------------------
                else:
                    base2D = get_region_mask(ds, region)  # dims: (lat, lon)
                    broad2D = base2D
                    for d in ds[var].dims:
                        if d not in ("lat", "lon"):
                            broad2D = broad2D.broadcast_like(ds[var])
                    spatial_combined &= broad2D

            var_mask = var_mask.where(spatial_combined, 0)

        # —————— (5) Explicit time‐step masking (if requested) ——————
        if time_steps and "time" in ds[var].dims:
            tmask = xr.zeros_like(ds[var], dtype=bool)
            for t in time_steps:
                tmask |= ds["time"] == ds["time"][t]
            var_mask = var_mask.where(tmask, 0)

        # Save this variable’s mask into the output
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
