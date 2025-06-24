import dataclasses
import functools
import os
from typing import Dict, Tuple, Union

import xarray as xr
import numpy as np

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp

from graphcast import autoregressive, casting, data_utils, graphcast, normalization
from pkg.forecast.run_forecast import get_model_checkpoint, load_normalization_data
from pkg.gcs_utils import client as gcs

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def save_saliency_to_netcdf(saliency_pytree, template_inputs, path, description="Gradient saliency"):
    """Save a pytree of saliency gradients to a NetCDF file.

    Parameters
    ----------
    saliency_pytree : pytree
        A pytree of gradients, e.g. the output of jax.grad.
    template_inputs : dict or xr.Dataset
        A dict or Dataset mapping variable names to DataArrays, used to determine dimensions.
    path : str
        The path where the NetCDF file will be saved.
    description : str
        A description to be saved in the file's attributes.
    """
    data_vars = {}
    for name, grad in saliency_pytree.items():
        arr = np.asarray(getattr(grad, "data", grad)).astype("float32")
        dims = template_inputs[name].dims if isinstance(template_inputs, dict) else template_inputs[name].dims
        data_vars[f"{name}_saliency"] = (dims, arr)
    xr.Dataset(data_vars, attrs={"description": description}).to_netcdf(path)

# -----------------------------------------------------------------------------
# Generic saliency primitives
# -----------------------------------------------------------------------------

def compute_grad_times_input(grad_fn, inputs):
    """
    Compute gradient × input saliency for each field in `inputs`.

    Parameters
    ----------
    grad_fn : Callable
        A function that takes `inputs` (your model inputs pytree) and
        returns a pytree of gradients with the same structure.
    inputs : pytree
        The same inputs you pass to `grad_fn`, e.g. the dict of arrays
        returned by your data_utils.extract_* call.

    Returns
    -------
    grad_x_input : pytree
        A pytree matching `inputs` where each leaf is grad*input.
    """
    grads = grad_fn(inputs)
    # elementwise multiply each gradient array by its corresponding input array
    grad_x_input = jax.tree_map(lambda g, x: g * x, grads, inputs)
    return grad_x_input

# --------------------------------------------------------------------------

def compute_integrated_gradients(grad_fn, inputs, baseline=None, m_steps=50):
    """
    Compute integrated gradients using a fori_loop to accumulate gradients.
    This function computes the integrated gradients by accumulating gradients
    at multiple steps between the baseline and the inputs, then averaging them.
    
    Parameters
    ----------
    grad_fn : Callable
        A function that takes `inputs` (your model inputs pytree) and
        returns a pytree of gradients with the same structure.

    inputs : pytree
        The same inputs you pass to `grad_fn`, e.g. the dict of arrays
        returned by your data_utils.extract_* call.
    baseline : dict or xr.Dataset, optional
        A baseline pytree or xarray.Dataset to use for integrated gradients.
        If None, defaults to a pytree of zeros with the same structure as `inputs`.
    m_steps : int, optional
        The number of steps to use for the integration. More steps will
        result in a more accurate estimate, but will also be more
        computationally expensive. Defaults to 50.
    
    Returns
    -------
    ig : pytree
        A pytree of integrated gradients with the same structure as `inputs`.
        Each leaf is the integrated gradient for the corresponding input.
    """

    if baseline is not None and isinstance(baseline, xr.Dataset):
        baseline = convert_xr_to_pytree(baseline, inputs)
    if baseline is None:
        baseline = jax.tree_map(lambda x: jnp.zeros_like(x), inputs)
    if not isinstance(baseline, dict):
        raise ValueError(f"Baseline must be a dict of arrays (or xr.Dataset), got {type(baseline)}")

    # accumulator for sum of grads
    init_sum = jax.tree_map(lambda x: jnp.zeros_like(x), inputs)

    def body_fun(i, sum_grads):
        alpha = (i + 1) / m_steps
        inp_k = jax.tree_map(lambda b, x: b + alpha * (x - b), baseline, inputs)
        gk = grad_fn(inp_k)
        sum_grads = jax.tree_map(lambda s, g: s + g, sum_grads, gk)
        return sum_grads

    # run the accumulation loop
    sum_grads = lax.fori_loop(0, m_steps, body_fun, init_sum)

    # average and scale by (inputs - baseline)
    avg_grads = jax.tree_map(lambda s: s / m_steps, sum_grads)
    ig = jax.tree_map(lambda ag, x, b: (x - b) * ag, avg_grads, inputs, baseline)
    return ig

# -----------------------------------------------------------------------------
# Utility converters
# -----------------------------------------------------------------------------

def convert_xr_to_pytree(xr_ds: xr.Dataset, inputs_template: dict) -> dict:
    """
    Convert an xarray.Dataset to the same dict‐pytree structure as `inputs_template`.
    Any variable missing in `xr_ds` is filled with zeros of the right shape.
    If a variable has a time dimension longer than needed, it is sliced or averaged.
    
    Parameters
    ----------
    xr_ds : xr.Dataset
        The xarray dataset containing the climatology data.
    inputs_template : dict      
        A dictionary mapping variable names to numpy arrays, used to determine the expected shapes.
    
    Returns
    -------
    pytree : dict
        A dictionary where each key corresponds to a variable in `inputs_template`, and the values are
        JAX arrays containing the data from `xr_ds` or zeros if the variable is not present.
    """
    pytree = {}
    for key, arr in inputs_template.items():
        shape = arr.shape
        if key in xr_ds:
            val = xr_ds[key].values
            while len(val.shape) < len(shape):
                val = np.expand_dims(val, axis=0)
            for i, (v, s) in enumerate(zip(val.shape, shape)):
                if v > s:
                    val = val.take(indices=range(s), axis=i)
            try:
                val = np.broadcast_to(val, shape)
            except Exception:
                raise ValueError(
                    f"Shape mismatch for variable '{key}': "
                    f"climatology shape {val.shape}, input shape {shape}"
                )
            val = np.asarray(val)
            pytree[key] = jnp.array(val)
        else:
            pytree[key] = jnp.zeros(shape, dtype=arr.dtype)
    return pytree

# --------------------------------------------------------------------------

def dict_to_xr(inputs_dict, template):
    """
    Convert a dict of arrays to an xarray.Dataset with the same structure as the template.
    Ensures that coordinates match exactly for JAX compatibility.

    Parameters
    ----------
    inputs_dict : dict
        A dictionary where keys are variable names and values are numpy arrays or JAX arrays.
    template : xr.Dataset
        An xarray Dataset that serves as a template for the output structure.   
   
    Returns
    -------
    xr.Dataset
        An xarray Dataset with the same structure as the template, filled with data from inputs_dict
        and coordinates copied from the template.
    """
    data_vars = {}
    for k, v in inputs_dict.items():
        if k not in template:
            raise ValueError(
                f"Variable '{k}' is in inputs but not in the template Dataset. "
                "All variables must be present in the template with correct dims/coords."
            )
        
        da = template[k]
        
        data_vars[k] = xr.DataArray(v, dims=da.dims, coords={
            dim: template[dim] for dim in da.dims if dim in template.coords
        })
    
    ds = xr.Dataset(data_vars, attrs=template.attrs)
    
    if "level" in template.coords:
        ds = ds.assign_coords(level=template.level)
    
    for dim in ["time", "lat", "lon"]:
        if dim in template.coords:
            ds = ds.assign_coords({dim: template[dim]})
    
    return ds

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _prepare_graphcast(
    model_name: str,
    raw_ds: xr.Dataset,
    *,
    lead_hours: int,
) -> Tuple:
    """
    Prepare the GraphCast model for inference, loading the checkpoint and normalization data.
    This function sets up the model, extracts inputs and targets from the dataset,
    and returns a JIT-compiled function for running predictions.
    Parameters
    ----------
    model_name : str
        The name of the GraphCast model to use.
    raw_ds : xr.Dataset
        The raw dataset containing input data for the model.
    lead_hours : int
        The lead time in hours for the forecast.
    Returns
    -------
    run_jit : Callable
        A JIT-compiled function that takes inputs and returns model predictions.
    inputs_xr : xr.Dataset
        An xarray Dataset containing the model inputs.
    inputs_dict : dict
        A dictionary mapping variable names to JAX arrays, used as inputs for the model.
    targets : xr.Dataset
        An xarray Dataset containing the target values for the model.
    forcings : xr.Dataset
        An xarray Dataset containing the forcing variables for the model.
    """
    ckpt, bucket, dir_prefix = get_model_checkpoint(model_name)
    diffs_stddev, mean, stddev = load_normalization_data(bucket, dir_prefix)

    model_config, task_config = ckpt.model_config, ckpt.task_config
    params, state = ckpt.params, {}

    lead_slice = slice(f"{lead_hours}h", f"{lead_hours}h")
    inputs_xr, targets, forcings = data_utils.extract_inputs_targets_forcings(
        raw_ds, target_lead_times=lead_slice, **dataclasses.asdict(task_config)
    )
    inputs_dict = {k: jnp.asarray(v.values) for k, v in inputs_xr.items()}

    @hk.transform_with_state
    def forward(mconf, tconf, _inp, _tgt, _frc):
        net = graphcast.GraphCast(mconf, tconf)
        net = casting.Bfloat16Cast(net)
        net = normalization.InputsAndResiduals(net,
                                              diffs_stddev_by_level=diffs_stddev,
                                              mean_by_level=mean,
                                              stddev_by_level=stddev)
        predictor = autoregressive.Predictor(net, gradient_checkpointing=True)
        return predictor(_inp, targets_template=_tgt, forcings=_frc)

    apply_f = functools.partial(forward.apply, params, state, None,
                                model_config, task_config)
    run_jit = jax.jit(lambda inp, tgt, frc: apply_f(inp, tgt, frc)[0])

    return run_jit, inputs_xr, inputs_dict, targets, forcings

# -----------------------------------------------------------------------------

def _scalar_output_factory(run_jit, targets, forcings, *,
                           lat_idx: int, lon_idx: int, lead_idx: int,
                           input_template: xr.Dataset):
    """Factory producing scalar‑output function on *dict* inputs.

    The input *dict* is turned back into an ``xarray.Dataset`` using
    ``input_template`` so it matches the signature that GraphCast expects.
    The function computes the wind speed at the specified lat/lon and lead time.
    Parameters
    ----------
    run_jit : Callable
        A JIT-compiled function that takes inputs and returns model predictions.
    targets : xr.Dataset    
        An xarray Dataset containing the target values for the model.
    forcings : xr.Dataset

        An xarray Dataset containing the forcing variables for the model.
    lat_idx : int
        The index of the latitude in the input data.
    lon_idx : int
        The index of the longitude in the input data.
    lead_idx : int
        The index of the lead time in the input data.
    input_template : xr.Dataset
        An xarray Dataset that serves as a template for the input structure.
    Returns
    -------
    scalar : Callable
        A function that takes a dictionary of inputs and returns the wind speed
        at the specified latitude and longitude for the given lead time.
    """
    def scalar(inputs_dict):
        inp_ds = dict_to_xr(inputs_dict, input_template)
        preds = run_jit(inp_ds, targets * jnp.nan, forcings)
        u = preds["10m_u_component_of_wind"].isel(
            time=lead_idx, lat=lat_idx, lon=lon_idx).data.jax_array
        v = preds["10m_v_component_of_wind"].isel(
            time=lead_idx, lat=lat_idx, lon=lon_idx).data.jax_array
        return jnp.sqrt(u**2 + v**2).squeeze()
    return scalar

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_forecast_with_gradients(
    model_name: str,
    input_data_or_path: Union[str, xr.Dataset],
    output_name: str,
    *,
    target_lat: float = 54.0,
    target_lon: float = 9.0,
    lead_hours: int = 6,
    upload_to_gcs: bool = False,
    saliency_folder: str = "../data/saliency_maps",
) -> None:
    """
    Run a GraphCast forecast and compute gradients for a specific target location.
    
    Parameters
    ----------
    model_name : str
        The name of the GraphCast model to use.
    input_data_or_path : str or xr.Dataset
        Path to the input data file or an xarray.Dataset containing the input data.
    output_name : str
        The name for the output files (forecast, saliency, etc.).
    target_lat : float          
        Latitude of the target location for which to compute gradients.
    target_lon : float
        Longitude of the target location for which to compute gradients.
    lead_hours : int
        The lead time in hours for the forecast.
    upload_to_gcs : bool
        If True, upload the output files to Google Cloud Storage.
    saliency_folder : str
        Local folder where the predictions and saliency maps will be saved.     
    """
    input_ds = (xr.open_dataset(input_data_or_path).load()
                if isinstance(input_data_or_path, str) else input_data_or_path)

    run_jit, inputs_xr, inputs_dict, targets, forcings = _prepare_graphcast(
        model_name, input_ds, lead_hours=lead_hours)

    lats, lons = input_ds["lat"].values, input_ds["lon"].values
    lat_idx = int(jnp.argmin(jnp.abs(lats - target_lat)))
    lon_idx = int(jnp.argmin(jnp.abs(lons - target_lon)))

    scalar_output = _scalar_output_factory(
        run_jit, targets, forcings,
        lat_idx=lat_idx, lon_idx=lon_idx, lead_idx=0,
        input_template=inputs_xr
    )
    grad_fn = jax.grad(scalar_output)
    
    # Compute vanilla gradient saliency
    saliency = grad_fn(inputs_dict) 
    # Compute gradient × input saliency
    grad_x_input = compute_grad_times_input(grad_fn, inputs_dict)

    # Save data
    os.makedirs(saliency_folder, exist_ok=True)

    saliency_path = os.path.join(saliency_folder, f"{output_name}_saliency.nc")
    grad_xinput_path = os.path.join(saliency_folder, f"{output_name}_grad_x_input.nc")
    
    save_saliency_to_netcdf(saliency, inputs_xr, saliency_path, description="Vanilla Gradient Saliency")
    save_saliency_to_netcdf(grad_x_input, inputs_xr, grad_xinput_path, description="Gradient × Input Saliency")

    print(f"Saliency saved to {saliency_path}")
    print(f"Gradient × Input saliency saved to {grad_xinput_path}")

    if upload_to_gcs:
        gcs.upload_file(saliency_path, saliency_path)
        gcs.upload_file(grad_xinput_path, grad_xinput_path)
        print("Files uploaded to Google Cloud Storage.")

# -----------------------------------------------------------------------------

def run_forecast_with_integrated_gradients(
    model_name: str,
    input_data_or_path: Union[str, xr.Dataset],
    output_name: str,
    *,
    target_lat: float = 54.0,
    target_lon: float = 9.0,
    lead_hours: int = 6,
    m_steps: int = 50,
    baseline: Union[xr.Dataset, Dict[str, jnp.ndarray], None] = None,
    upload_to_gcs: bool = False,
    saliency_folder: str = "../data/saliency_maps",
) -> None:
    """
    Runs a GraphCast forecast for a specified model and input data,
    computes Integrated Gradients for a specific target location, and saves the results
    to NetCDF files. It can also upload the results to Google Cloud Storage if specified.
    It handles the model loading, input extraction, and normalization internally.
    It also supports using a baseline for Integrated Gradients, which can be an xarray.Dataset
    or a dictionary of arrays. If no baseline is provided, it defaults to a zero baseline.
    The function computes the Integrated Gradients using a fori_loop to accumulate gradients
    at multiple steps between the baseline and the inputs, then averages them.
    The results are saved to NetCDF files, and optionally uploaded to GCS.
    
    Parameters
    ----------
    model_name : str
        The name of the GraphCast model to use.
    input_data_or_path : str or xr.Dataset
        Path to the input data file or an xarray.Dataset containing the input data. 
    output_name : str
        The name for the output files (forecast, saliency, etc.).
    target_lat : float
        Latitude of the target location for which to compute gradients.
    target_lon : float
        Longitude of the target location for which to compute gradients.
    lead_hours : int
        The lead time in hours for the forecast.
    m_steps : int
        The number of steps to use for the integration. More steps will
        result in a more accurate estimate, but will also be more computationally expensive.
    baseline : xr.Dataset or dict, optional
        A baseline pytree or xarray.Dataset to use for integrated gradients.
        If None, defaults to a pytree of zeros with the same structure as `inputs`.
    upload_to_gcs : bool
        If True, upload the output files to Google Cloud Storage.
    saliency_folder : str
        Local folder where the predictions and saliency maps will be saved.
    """
    input_ds = (xr.open_dataset(input_data_or_path).load()
                if isinstance(input_data_or_path, str) else input_data_or_path)

    run_jit, inputs_xr, inputs_dict, targets, forcings = _prepare_graphcast(
        model_name, input_ds, lead_hours=lead_hours)

    lats, lons = input_ds["lat"].values, input_ds["lon"].values
    lat_idx = int(jnp.argmin(jnp.abs(lats - target_lat)))
    lon_idx = int(jnp.argmin(jnp.abs(lons - target_lon)))

    scalar_output = _scalar_output_factory(run_jit, targets, forcings,
                                           lat_idx=lat_idx, lon_idx=lon_idx, lead_idx=0, input_template=inputs_xr)
    grad_fn = jax.grad(scalar_output)

    if isinstance(baseline, xr.Dataset):
        baseline = convert_xr_to_pytree(baseline, inputs_dict)

    ig = compute_integrated_gradients(
        grad_fn, inputs_dict, baseline=baseline, m_steps=m_steps)

    os.makedirs(saliency_folder, exist_ok=True)
    out_path = os.path.join(saliency_folder, f"{output_name}_integrated.nc")
    save_saliency_to_netcdf(ig, inputs_xr, out_path, description="integrated saliency")
    print(f"Integrated gradients saved to {out_path}")
    if upload_to_gcs:
        gcs.upload_file(out_path, out_path)
        print("File uploaded to Google Cloud Storage.")