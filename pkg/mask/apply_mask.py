import xarray as xr
import numpy as np

def ensure_required_climatology_vars(climatology: xr.Dataset, input_data: xr.Dataset) -> xr.Dataset:
    climatology = climatology.copy()

    for var in input_data.data_vars:
        if var not in climatology:
            shape = tuple(input_data[var].sizes[dim] for dim in input_data[var].dims)
            dummy = np.zeros(shape, dtype=np.float32)

            climatology[var] = xr.DataArray(
                dummy,
                dims=input_data[var].dims,
                coords={dim: input_data.coords[dim] for dim in input_data[var].dims}
            )

    return climatology

def apply_mask_to_input(
    input_data: xr.Dataset,
    climatology_data: xr.Dataset,
    mask: xr.Dataset
) -> xr.Dataset:
    """
    Apply a binary mask to input data, replacing masked values with climatology.

    Parameters
    ----------
    input_data : xr.Dataset
        The original dataset used as input for forecasting.
    climatology_data : xr.Dataset
        The climatology dataset providing values for masked regions.
    mask : xr.Dataset
        Binary dataset (1 = mask/replace, 0 = keep input). Must match input in shape and coordinates.

    Returns
    -------
    xr.Dataset
        The masked input dataset with selected values replaced by climatology.
    """

    # Fix time mismatch if necessary
    if "time" in climatology_data.dims and "time" in input_data.dims:
        if not climatology_data["time"].equals(input_data["time"]):
            climatology_data = climatology_data.copy()
            climatology_data["time"] = input_data["time"]

    # Check variable consistency
    if set(mask.data_vars) != set(input_data.data_vars):
        raise ValueError("Mask and input data must have the same variables.")

    # Check shape and coordinates for each variable
    for var in input_data.data_vars:
        if input_data[var].shape != mask[var].shape:
            raise ValueError(f"Shape mismatch for variable '{var}' between input and mask.")
        if input_data[var].dims != mask[var].dims:
            raise ValueError(f"Dimension mismatch for variable '{var}' between input and mask.")
        
    # Add missing variables for climatology
    climatology_data = ensure_required_climatology_vars(climatology_data, input_data)

    # Apply mask: keep input where mask == 0, use climatology where mask == 1
    masked_input = xr.Dataset()
    for var in input_data.data_vars:
        masked_input[var] = input_data[var].where(mask[var] == 0, climatology_data[var])

    # Copy coordinates and attributes
    for coord in input_data.coords:
        masked_input.coords[coord] = input_data.coords[coord]
    masked_input.attrs = input_data.attrs

    return masked_input

