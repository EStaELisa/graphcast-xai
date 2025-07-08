import os
import re
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Dict


def plot_mean_wind_speed_comparison_feature_occlusion(
    masked_folder: str,
    input_path: Union[str, os.PathLike],
    region_lat: slice = slice(55.0, 53.0),
    region_lon: slice = slice(7.5, 10.5),
    title: str = "Mean Wind Speed at 22:00 UTC over North Germany (10m) – Feature Occlusion",
    rename_map: Optional[dict] = None,
    order: Optional[List[str]] = None,
    order_by_size: bool = False,
):
    """
    Plots the mean wind speed at 10m height for a specified region, comparing un
    masked and masked datasets.

    Parameters
    ----------
    masked_folder: str
        Path to the folder containing masked datasets.
    input_path: Union[str, os.PathLike]
        Path to the unmasked dataset.
    region_lat: slice
        Latitude slice for the region of interest.
    region_lon: slice
        Longitude slice for the region of interest.
    title: str
        Title of the plot.
    rename_map: Optional[dict]
        Dictionary to rename labels in the plot.
    order: Optional[List[str]]
        List to specify the order of labels in the plot.
    order_by_size: bool
        If True, orders the bars by size.
    """
    def mean_wind_speed(ds, time_sel=None):
        if time_sel is not None:
            u10 = ds["10m_u_component_of_wind"].isel(time=time_sel)
            v10 = ds["10m_v_component_of_wind"].isel(time=time_sel)
        else:
            u10 = ds["10m_u_component_of_wind"].squeeze()
            v10 = ds["10m_v_component_of_wind"].squeeze()

        wind_speed = np.sqrt(u10**2 + v10**2)
        return wind_speed.sel(lat=region_lat, lon=region_lon).mean().item()

    wind_means = {}

    # Unmasked
    unmasked_ds = xr.open_dataset(input_path)
    if "time" in unmasked_ds and unmasked_ds.time.size > 1:
        try:
            time_sel = int(np.where(unmasked_ds.time.dt.hour == 22)[0][0])
        except (AttributeError, IndexError):
            time_sel = 0
    else:
        time_sel = None

    unmasked_mean = mean_wind_speed(unmasked_ds, time_sel)

    # Masked
    for fname in sorted(os.listdir(masked_folder)):
        if not fname.endswith(".nc"):
            continue
        path = os.path.join(masked_folder, fname)
        ds = xr.open_dataset(path)
        label = fname.replace("masked_", "").replace(".nc", "")
        display_label = rename_map[label] if rename_map and label in rename_map else label
        wind_means[display_label] = mean_wind_speed(ds)

    # Apply custom order if provided
    if order:
        labels = [lbl for lbl in order if lbl in wind_means]
    else:
        labels = list(wind_means.keys())

    values = [wind_means[lbl] for lbl in labels]

    # Optionally order by size
    if order_by_size:
        sorted_pairs = sorted(zip(values, labels))
        values, labels = zip(*sorted_pairs)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.axhline(unmasked_mean, color='black', linestyle='--', label='Unmasked Reference')
    plt.ylabel("10m Wind Speed at 22:00 UTC (m/s)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_region_level_heatmap(
    masked_folder: Union[str, os.PathLike],
    reference_path: Union[str, os.PathLike],
    region_lat: slice = slice(55.0, 53.0),
    region_lon: slice = slice(7.5, 10.5),
    title: str = "Δ Wind Speed – Region vs Pressure Level",
    rename_map: Optional[Dict[str, str]] = None,
    level_order: Optional[list] = None,
    region_order: Optional[list] = None
):
    """
    Plots a heatmap of the difference in mean wind speed at 10m height for different regions and pressure levels.
    
    Parameters
    ----------
    masked_folder: Union[str, os.PathLike]
        Path to the folder containing masked datasets.
    reference_path: Union[str, os.PathLike]
        Path to the reference dataset for comparison.
    region_lat: slice
        Latitude slice for the region of interest.
    region_lon: slice
        Longitude slice for the region of interest.
    title: str
        Title of the plot.
    rename_map: Optional[Dict[str, str]]
        Dictionary to rename regions and levels in the plot.
    level_order: Optional[list]
        List to specify the order of pressure levels in the plot.
    region_order: Optional[list]
        List to specify the order of regions in the plot.
    """
    def parse_filename(fname):
        name = os.path.splitext(fname)[0]

        groups = {}
        level_match = re.search(r"_lev([0-9]+|none)", name)
        region_match = re.search(r"_reg([A-Za-z0-9_]+)", name)

        if level_match:
            groups["Level"] = level_match.group(1)
        if region_match:
            groups["Region"] = region_match.group(1)

        return groups

    def mean_value(ds):
        # Compute 10m wind speed
        if "10m_u_component_of_wind" in ds and "10m_v_component_of_wind" in ds:
            u10 = ds["10m_u_component_of_wind"].squeeze()
            v10 = ds["10m_v_component_of_wind"].squeeze()
            wind_speed = np.sqrt(u10**2 + v10**2)
            return wind_speed.sel(lat=region_lat, lon=region_lon).mean().item()
        else:
            raise ValueError("Wind components not found in dataset.")

    # Reference wind speed (should always contain 10m_u and 10m_v)
    reference_ds = xr.open_dataset(reference_path)
    reference_mean = mean_value(reference_ds)

    records = []

    for fname in sorted(os.listdir(masked_folder)):
        if not fname.endswith(".nc"):
            continue

        groups = parse_filename(fname)
        if "Region" not in groups or "Level" not in groups:
            continue

        level = groups["Level"]
        region = groups["Region"]

        level = rename_map.get(level, level) if rename_map else level
        region = rename_map.get(region, region) if rename_map else region

        try:
            ds = xr.open_dataset(os.path.join(masked_folder, fname))
            value = mean_value(ds)
            delta = value - reference_mean
            records.append((region, level, delta))
        except Exception as e:
            print(f"⚠️ Skipped {fname} due to: {e}")
            continue

    if not records:
        raise ValueError("No valid data records parsed. Check filename patterns and contents.")

    df = pd.DataFrame(records, columns=["Region", "Level", "Δ Wind Speed"])

    if level_order:
        df["Level"] = pd.Categorical(df["Level"], categories=level_order, ordered=True)
    if region_order:
        df["Region"] = pd.Categorical(df["Region"], categories=region_order, ordered=True)

    pivot_df = df.pivot_table(index="Region", columns="Level", values="Δ Wind Speed", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        pivot_df.values,
        cmap="coolwarm",
        vmin=-6,
        vmax=6
    )

    ax.set_xticks(np.arange(pivot_df.shape[1]))
    ax.set_yticks(np.arange(pivot_df.shape[0]))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            val = pivot_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Wind Speed (m/s)")

    ax.set_title(title)
    ax.set_xlabel("Pressure Level")
    ax.set_ylabel("Region")
    plt.tight_layout()
    plt.show()


def plot_feature_grouped_heatmap(
    masked_folder: Union[str, os.PathLike],
    reference_path: Union[str, os.PathLike],
    group_x: str = "Pressure Levels",
    group_y: str = "Variable",
    region_lat: slice = slice(55.0, 53.0),
    region_lon: slice = slice(7.5, 10.5),
    title: str = "Δ Wind Speed – Grouped Heatmap",
    rename_map: Optional[Dict[str, str]] = None,
    order: Optional[Dict[str, List[str]]] = None
):
    """
    Plots a heatmap of the difference in mean wind speed at 10m height for different groups (e.g., pressure levels and variables).

    Parameters
    ----------
    masked_folder: Union[str, os.PathLike]
        Path to the folder containing masked datasets.
    reference_path: Union[str, os.PathLike]
        Path to the reference dataset for comparison.
    group_x: str
        Name of the group for the x-axis (e.g., "Pressure Levels").
    group_y: str
        Name of the group for the y-axis (e.g., "Variable").
    region_lat: slice
        Latitude slice for the region of interest.
    region_lon: slice
        Longitude slice for the region of interest.
    title: str
        Title of the plot.
    rename_map: Optional[Dict[str, str]]
        Dictionary to rename groups in the plot.
    order: Optional[Dict[str, List[str]]]
        Dictionary to specify the order of groups in the plot.  
    """
    def parse_filename_general(fname):
        name = fname.replace("masked_", "").replace(".nc", "")
        groups = {}

        stop_tokens = ["lev(", "reg", "t"]
        for token in stop_tokens:
            token_idx = name.find(token)
            if token_idx != -1:
                var_str = name[:token_idx].rstrip("_")
                name = name[token_idx:]
                break
        else:
            var_str = name
            name = ""

        try:
            parsed = ast.literal_eval(var_str)
            groups["Variable"] = "\n".join(parsed) if isinstance(parsed, tuple) else parsed
        except Exception:
            groups["Variable"] = var_str

        group_patterns = {
            "Pressure Levels": r"lev\((.*?)\)",
            "Region": r"reg(\(.*?\)|[A-Za-z0-9_]+)",
            "Time Step": r"t([0-9\-]+|none)"
        }

        for key, pattern in group_patterns.items():
            m = re.search(pattern, name)
            if m:
                groups[key] = m.group(1)

        return groups

    def mean_wind_speed(ds):
        u10 = ds["10m_u_component_of_wind"].squeeze()
        v10 = ds["10m_v_component_of_wind"].squeeze()
        wind_speed = np.sqrt(u10**2 + v10**2)
        return wind_speed.sel(lat=region_lat, lon=region_lon).mean().item()

    # Load reference mean wind speed
    reference_ds = xr.open_dataset(reference_path)
    reference_mean = mean_wind_speed(reference_ds)

    records = []

    # Loop through masked files
    for fname in sorted(os.listdir(masked_folder)):
        if not fname.endswith(".nc"):
            continue

        groups = parse_filename_general(fname)
        if group_x not in groups or group_y not in groups:
            continue

        x_val = rename_map.get(groups[group_x], groups[group_x]) if rename_map else groups[group_x]
        y_val = rename_map.get(groups[group_y], groups[group_y]) if rename_map else groups[group_y]

        ds = xr.open_dataset(os.path.join(masked_folder, fname))
        value = mean_wind_speed(ds)
        delta = value - reference_mean
        records.append((y_val, x_val, delta))

    if not records:
        raise ValueError("No valid data records were parsed. Check filename patterns and group names.")

    df = pd.DataFrame(records, columns=[group_y, group_x, "Δ Wind Speed"])

    if order:
        if group_y in order:
            df[group_y] = pd.Categorical(df[group_y], categories=order[group_y], ordered=True)
        if group_x in order:
            df[group_x] = pd.Categorical(df[group_x], categories=order[group_x], ordered=True)

    pivot_df = df.pivot_table(index=group_y, columns=group_x, values="Δ Wind Speed", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot_df.values, cmap="coolwarm", vmin=-np.abs(pivot_df.values).max(), vmax=np.abs(pivot_df.values).max())

    ax.set_xticks(np.arange(pivot_df.shape[1]))
    ax.set_yticks(np.arange(pivot_df.shape[0]))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            val = pivot_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Wind Speed (m/s)")

    ax.set_title(title)
    ax.set_xlabel(group_x)
    ax.set_ylabel(group_y)
    plt.tight_layout()
    plt.show()


def plot_feature_grouped_heatmap_special_for_pressure_level(
    masked_folder: Union[str, os.PathLike],
    reference_path: Union[str, os.PathLike],
    group_x: str = "Pressure Levels",
    group_y: str = "Variable",
    region_lat: slice = slice(55.0, 53.0),
    region_lon: slice = slice(7.5, 10.5),
    title: str = "Δ Wind Speed – Grouped Heatmap",
    rename_map: Optional[Dict[str, str]] = None,
    order: Optional[Dict[str, List[str]]] = None
):
    """
    Plots a heatmap of the difference in mean wind speed at 10m height for different groups (e.g., pressure levels and variables).
    This function is specialized for pressure levels, allowing for more flexible parsing of filenames.  
    
    Parameters
    ----------
    masked_folder: Union[str, os.PathLike]
        Path to the folder containing masked datasets.  
    reference_path: Union[str, os.PathLike]
        Path to the reference dataset for comparison.
    group_x: str
        Name of the group for the x-axis (e.g., "Pressure Levels").
    group_y: str
        Name of the group for the y-axis (e.g., "Variable").
    region_lat: slice
        Latitude slice for the region of interest.
    region_lon: slice
        Longitude slice for the region of interest.
    title: str
        Title of the plot.
    rename_map: Optional[Dict[str, str]]
        Dictionary to rename groups in the plot.
    order: Optional[Dict[str, List[str]]]
        Dictionary to specify the order of groups in the plot.
    """
    def parse_filename_general(fname):
        name = fname.replace("masked_", "").replace(".nc", "")
        groups = {}

        # Match variable block at the beginning (before lev/reg/t)
        var_match = re.match(r"^(.*?)(?=_lev|_reg|_t)", name)
        var_str = var_match.group(1) if var_match else name

        try:
            parsed = ast.literal_eval(var_str)
            groups["Variable"] = "\n".join(parsed) if isinstance(parsed, tuple) else parsed
        except Exception:
            groups["Variable"] = var_str

        group_patterns = {
            "Pressure Levels": r"_lev(\d+)",
            "Region": r"_reg([A-Za-z0-9_]+)",
            "Time Step": r"_t([A-Za-z0-9_\-]+)"
        }

        for key, pattern in group_patterns.items():
            m = re.search(pattern, name)
            if m:
                groups[key] = m.group(1)

        return groups


    def mean_wind_speed(ds):
        u10 = ds["10m_u_component_of_wind"].squeeze()
        v10 = ds["10m_v_component_of_wind"].squeeze()
        wind_speed = np.sqrt(u10**2 + v10**2)
        return wind_speed.sel(lat=region_lat, lon=region_lon).mean().item()

    # Load reference mean wind speed
    reference_ds = xr.open_dataset(reference_path)
    reference_mean = mean_wind_speed(reference_ds)

    records = []

    # Loop through masked files
    for fname in sorted(os.listdir(masked_folder)):
        if not fname.endswith(".nc"):
            continue

        groups = parse_filename_general(fname)
        if group_x not in groups or group_y not in groups:
            continue

        x_val = rename_map.get(groups[group_x], groups[group_x]) if rename_map else groups[group_x]
        y_val = rename_map.get(groups[group_y], groups[group_y]) if rename_map else groups[group_y]

        ds = xr.open_dataset(os.path.join(masked_folder, fname))
        value = mean_wind_speed(ds)
        delta = value - reference_mean
        records.append((y_val, x_val, delta))

    if not records:
        raise ValueError("No valid data records were parsed. Check filename patterns and group names.")

    df = pd.DataFrame(records, columns=[group_y, group_x, "Δ Wind Speed"])

    if order:
        if group_y in order:
            df[group_y] = pd.Categorical(df[group_y], categories=order[group_y], ordered=True)
        if group_x in order:
            df[group_x] = pd.Categorical(df[group_x], categories=order[group_x], ordered=True)

    pivot_df = df.pivot_table(index=group_y, columns=group_x, values="Δ Wind Speed", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(pivot_df.values, cmap="coolwarm", vmin=-np.abs(pivot_df.values).max(), vmax=np.abs(pivot_df.values).max())

    ax.set_xticks(np.arange(pivot_df.shape[1]))
    ax.set_yticks(np.arange(pivot_df.shape[0]))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(pivot_df.shape[0]):
        for j in range(pivot_df.shape[1]):
            val = pivot_df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ Wind Speed (m/s)")

    ax.set_title(title)
    ax.set_xlabel(group_x)
    ax.set_ylabel(group_y)
    plt.tight_layout()
    plt.show()