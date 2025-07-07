
from typing import List, Union, Optional
import xarray as xr
import os
import itertools

from pkg.mask.generate_mask import create_feature_mask
from pkg.mask.apply_mask import apply_mask_to_input
from pkg.forecast.run_forecast import run_forecast


def run_forecast_with_mask(
    input_data: xr.Dataset,
    climatology: xr.Dataset,
    output_name: str,
    predictions_folder: str,
    variables: Optional[List[str]] = None,
    pressure_levels: Optional[List[int]] = None,
    regions: Optional[List[str]] = None,
    time_steps: Optional[List[int]] = None,
    model: str = "graphcast_small",
    upload_to_gcs=False
):
    """
    Runs a GraphCast forecast using masked input data.

    Creates a binary mask to replace selected features (e.g., variables, regions, levels, or time steps)
    in the input data with values from the climatology. The modified input is then used to run the forecast.

    Parameters
    ----------
    input_data : xr.Dataset
        Forecast input dataset in GraphCast format.
    climatology : xr.Dataset
        Climatology dataset to fill in masked regions.
    output_name : str
        Filename (without extension) for the forecast output.
    predictions_folder : str
        Directory to save the output NetCDF file.
    variables, pressure_levels, regions, time_steps : list, optional
        Criteria for masking the input.
    model : str
        Name of the GraphCast model to use (default: "graphcast_small").
    upload_to_gcs : bool
        If True, upload the result to Google Cloud Storage.
    """
    # Make mask
    mask = create_feature_mask(input_data, variables, pressure_levels, regions, time_steps)

    # Apply mask to input data
    masked_input = apply_mask_to_input(
        input_data=input_data,
        climatology_data=climatology,
        mask=mask,
    )

    # Run the GraphCast to get the forecast
    run_forecast(
        model_name=model,
        input_data_or_path=masked_input,
        output_name=output_name,
        upload_to_gcs=upload_to_gcs,
        predictions_folder=predictions_folder,
    )


def run_multiple_forecasts_with_mask(
    input_data: xr.Dataset,
    climatology: xr.Dataset,
    predictions_folder: str,
    variable_groups: Optional[List[Union[str, List[str]]]] = None,
    pressure_levels: Optional[List[Union[int, List[int]]]] = None,
    regions: Optional[List[Union[str, List[str]]]] = None,
    time_steps: Optional[List[Union[int, List[int]]]] = None,
    model: str = "graphcast_small",
    upload_to_gcs: bool = False,
):
    """
    Runs multiple GraphCast forecasts using combinations of masked features.

    Parameters
    ----------
    input_data : xr.Dataset
        Forecast input dataset in GraphCast format.
    climatology : xr.Dataset
        Climatology dataset to fill in masked regions.
    predictions_folder : str
        Directory to save the forecast outputs.
    variable_groups : list of str or list of list of str, optional
        Variable names or groups of variables to mask (e.g. "temperature", ["u", "v"]).
    pressure_levels : list of int or list of list of int, optional
        One or more pressure levels or level groups to mask.
    regions : list of str or list of list of str, optional
        One or more region names or region groups to mask.
    time_steps : list of int or list of list of int, optional
        One or more time step indices or index groups to mask.
    model : str
        Name of the GraphCast model to use.
    upload_to_gcs : bool
        If True, upload results to Google Cloud Storage.
    """

    os.makedirs(predictions_folder, exist_ok=True)

    variable_groups = variable_groups or [None]
    pressure_levels = pressure_levels or [None]
    regions = regions or [None]
    time_steps = time_steps or [None]

    combinations = itertools.product(variable_groups, pressure_levels, regions, time_steps)

    for variables, levels, regs, steps in combinations:
        # Normalization function to handle both single values and lists
        def _fmt(x):
            if isinstance(x, list):
                return "_".join(str(v) for v in x)
            return str(x) if x is not None else "none"

        variables_str = _fmt(variables)
        levels_str = _fmt(levels)
        regions_str = _fmt(regs)
        steps_str = _fmt(steps)

        output_name = f"masked_{variables_str}_lev{levels_str}_reg{regions_str}_t{steps_str}"

        print(f"🚀 Running forecast with mask on: vars={variables}, levels={levels}, regions={regs}, time_steps={steps}")

        # Normalize inputs to lists (so create_feature_mask gets a List[str], not a tuple)
        variables_list = None
        if variables:
            if isinstance(variables, (list, tuple)):
                variables_list = list(variables)
            else:
                variables_list = [variables]

        levels_list = None
        if levels is not None:
            if isinstance(levels, list):
                levels_list = levels
            else:
                levels_list = [levels]

        if regs:
            if isinstance(regs, (list, tuple)):
                regions_list = list(regs)
            else:
                regions_list = [regs]
        else:
            regions_list = None

        steps_list = None
        if steps is not None:
            if isinstance(steps, list):
                steps_list = steps
            else:
                steps_list = [steps]

        try:
            run_forecast_with_mask(
                input_data=input_data,
                climatology=climatology,
                output_name=output_name,
                predictions_folder=predictions_folder,
                variables=variables_list,
                pressure_levels=levels_list,
                regions=regions_list,
                time_steps=steps_list,
                model=model,
                upload_to_gcs=upload_to_gcs,
            )

        except Exception as e:
            print(f"‼ run_forecast_with_mask FAILED for regs = {regs!r}")
            print(f"   Exception message: {e}")
            raise
