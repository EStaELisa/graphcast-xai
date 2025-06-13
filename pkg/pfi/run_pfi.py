import xarray as xr
import numpy as np
from typing import List, Optional
import pandas as pd
from pathlib import Path
from typing import Union

from pkg.mask.generate_mask import create_feature_mask, compute_wind_speed
from pkg.forecast.run_forecast import run_forecast


def permute_variable_with_mask(
    ds: xr.Dataset,
    var_name: str,
    mask: xr.Dataset,
    level_value: Optional[float] = None,
) -> xr.Dataset:
    """
    Permute values of a variable in ds, restricted to where mask == 1.
    """
    ds_shuffled = ds.copy(deep=True)
    var = ds[var_name]
    m = mask[var_name]

    if level_value is not None and "level" in var.dims:
        # Convert level value to index
        level_coord = ds.level.values
        level_index = int(np.where(level_coord == level_value)[0][0])
        var = var.isel(level=level_index)
        m = m.isel(level=level_index)
    else:
        level_index = None

    shuffled = var.copy(deep=True)
    coords = np.where(m.values == 1)
    values = var.values[coords]
    np.random.shuffle(values)
    shuffled.values[coords] = values

    if level_index is not None and "level" in ds_shuffled[var_name].dims:
        ds_shuffled[var_name].loc[dict(level=ds.level[level_index])] = shuffled
    else:
        ds_shuffled[var_name] = shuffled

    return ds_shuffled



def run_single_permutation(
    ds_input: xr.Dataset,
    var: Union[str, List[str]],
    mask: xr.Dataset,
    run_index: int,
    folder: str,
    model_name: str = "graphcast_small",
    upload_to_gcs: bool = False
):
    if isinstance(var, str):
        var = [var]

    ds_shuffled = ds_input.copy(deep=True)

    # Permute all levels for each variable in the group
    for v in var:
        if "level" in ds_input[v].dims:
            for level_value in ds_input.level.values:
                ds_shuffled = permute_variable_with_mask(ds_shuffled, v, mask, level_value=level_value)
        else:
            ds_shuffled = permute_variable_with_mask(ds_shuffled, v, mask, level_value=None)

    # Construct sensible output name
    var_name = "_".join(var)
    output_name = f"{var_name}_run{run_index}"

    # Run forecast
    run_forecast(
        model_name=model_name,
        input_data_or_path=ds_shuffled,
        output_name=output_name,
        predictions_folder=f"../data/pfi/{folder}",
        upload_to_gcs=upload_to_gcs
    )



def run_pfi(
    ds_input: xr.Dataset,
    features: List[str] = None,
    pressure_levels: List[float] = None,
    regions: Optional[List[str]] = None,
    time_steps: Optional[List[int]] = None,
    folder: str = "../data/pfi",
    repetitions: int = 1,
    model_name: str = "graphcast_small",
    upload_to_gcs: bool = False
):
    """
    Runs forecast after permuting each level of each feature, then saves one forecast per variable/repetition.
    """

    # Default to all variables
    if features is None:
        features = list(ds_input.data_vars)

    # Default to all levels in the dataset
    if pressure_levels is None:
        if "level" in ds_input.dims:
            pressure_levels = list(ds_input.level.values)
        elif "pressure_level" in ds_input.dims:
            pressure_levels = list(ds_input.pressure_level.values)
        else:
            pressure_levels = [None]

    # Inside run_pfi loop
    for var_group in features:
        if isinstance(var_group, str):
            var_group = [var_group]

        # Create joint mask for all vars
        mask = create_feature_mask(
            ds=ds_input,
            variables=var_group,
            pressure_levels=pressure_levels,  # <- apply to all levels at once
            regions=regions,
            time_steps=time_steps
        )

        for r in range(repetitions):
            print(f"Permuting all levels of {var_group} (run {r+1}/{repetitions})...")
            run_single_permutation(
                ds_input=ds_input,
                var=var_group,
                mask=mask,
                run_index=r,
                folder=folder,
                model_name=model_name,
                upload_to_gcs=upload_to_gcs
            )




def compute_rmse(pred: xr.Dataset, target: xr.Dataset) -> float:
    """
    Compute RMSE of 10m wind speed (from u10, v10) over 53–55°N, 8–10°E.
    """
    u_name = "10m_u_component_of_wind"
    v_name = "10m_v_component_of_wind"
    new_name = "10m_wind_speed"

    # Compute wind speed for both prediction and target
    pred_ws = compute_wind_speed(pred, u_name=u_name, v_name=v_name, new_name=new_name)
    target_ws = compute_wind_speed(target, u_name=u_name, v_name=v_name, new_name=new_name)

    pred_da = pred_ws[new_name]
    target_da = target_ws[new_name]

    # Subset region
    lat_cond = (pred_da.lat >= 53.0) & (pred_da.lat <= 55.0)
    lon_cond = (pred_da.lon >= 8.0) & (pred_da.lon <= 10.0)
    pred_sub = pred_da.where(lat_cond & lon_cond, drop=True)
    target_sub = target_da.where(lat_cond & lon_cond, drop=True)

    # Compute RMSE
    diff_sq = (pred_sub - target_sub) ** 2
    return np.sqrt(diff_sq.mean().item())


def evaluate_pfi_folder(
    reference: Union[str, xr.Dataset],
    target: Union[str, xr.Dataset],
    folder: str
) -> pd.DataFrame:
    """
    Evaluates permuted forecasts in a folder and summarizes feature-wise importance.

    Parameters
    ----------
    reference : str or xr.Dataset
        Baseline forecast (used to compute baseline RMSE).
    target : str or xr.Dataset
        Ground truth (e.g., input data or ERA5, used to compute RMSEs).
    folder : str
        Folder with forecast NetCDF files (named like 'feature_run#.nc').

    Returns
    -------
    pd.DataFrame
        Summary table with mean_rmse, std, SEM, delta_rmse for each feature.
    """
    if isinstance(reference, str):
        reference = xr.open_dataset(reference)
    if isinstance(target, str):
        target = xr.open_dataset(target)

    # ✅ Use target to compute the true baseline RMSE
    baseline_rmse = compute_rmse(reference, target)

    folder_path = Path(folder)
    forecast_files = list(folder_path.glob("*.nc"))

    rows = []
    for file in forecast_files:
        filename = file.stem
        if "_run" not in filename:
            continue
        try:
            feature, run = filename.rsplit("_run", 1)
            run = int(run)
        except ValueError:
            continue

        pred = xr.open_dataset(file)
        rmse = compute_rmse(pred, target)

        rows.append({"feature": feature, "run": run, "rmse": rmse})

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No valid forecast files found in folder.")

    summary = (
        df.groupby("feature")["rmse"]
        .agg([
            ("mean_rmse", "mean"),
            ("std_rmse", "std"),
            ("n", "count")
        ])
        .assign(
            sem_rmse=lambda d: d["std_rmse"] / np.sqrt(d["n"]),
            delta_rmse=lambda d: d["mean_rmse"] - baseline_rmse,
            baseline_rmse=baseline_rmse
        )
        .reset_index()
    )

    return summary
