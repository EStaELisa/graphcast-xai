import dataclasses
import functools
import os

import xarray as xr
import numpy as np

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp

from graphcast import autoregressive, casting, data_utils, graphcast, normalization
from pkg.forecast.run_forecast import get_model_checkpoint, load_normalization_data
from pkg.gcs_utils import client as gcs

# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------

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

# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------

def run_forecast_with_gradients(
    model_name: str,
    input_data_or_path: str | xr.Dataset,
    output_name: str,
    target_lat: float = 54.0,
    target_lon: float = 9.0,
    lead_hours: int = 6,
    upload_to_gcs: bool = False,
    saliency_folder: str = "../data/saliency_maps",
):
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
    # Load model checkpoint and normalization data
    ckpt, bucket, dir_prefix = get_model_checkpoint(model_name)
    diffs_stddev, mean, stddev = load_normalization_data(bucket, dir_prefix)
    model_config, task_config = ckpt.model_config, ckpt.task_config
    params, state = ckpt.params, {}

    # Load input data
    if isinstance(input_data_or_path, str):
        input_data = xr.open_dataset(input_data_or_path).load()
    else:
        input_data = input_data_or_path

    lead_slice = slice(f"{lead_hours}h", f"{lead_hours}h")  # single step
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        input_data, target_lead_times=lead_slice, **dataclasses.asdict(task_config)
    )

    # GraphCast forward
    @hk.transform_with_state
    def run_forward(mconf, tconf, _inputs, _targets, _forcings):
        pred = graphcast.GraphCast(mconf, tconf)
        pred = casting.Bfloat16Cast(pred)
        pred = normalization.InputsAndResiduals(
            pred,
            diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean,
            stddev_by_level=stddev
        )
        predictor = autoregressive.Predictor(pred, gradient_checkpointing=True)
        return predictor(_inputs, targets_template=_targets, forcings=_forcings)

    # bind and jit
    run_fn = functools.partial(
        run_forward.apply,
        params, state, None, model_config, task_config
    )
    run_jitted = jax.jit(lambda inp, tgt, frc: run_fn(inp, tgt, frc)[0])

    # Compute the target indices for the specified lat/lon
    lats = input_data["lat"].values
    lons = input_data["lon"].values
    lat_idx = int(jnp.argmin(jnp.abs(lats - target_lat)))
    lon_idx = int(jnp.argmin(jnp.abs(lons - target_lon)))
    lead_idx = 0 

    # define scalar_output
    def scalar_output(_inputs):
        preds = run_jitted(_inputs, targets * jnp.nan, forcings)

        da_u = preds["10m_u_component_of_wind"].isel(
            time=lead_idx, lat=lat_idx, lon=lon_idx
        )
        da_v = preds["10m_v_component_of_wind"].isel(
            time=lead_idx, lat=lat_idx, lon=lon_idx
        )

        # grab the JAX arrays
        u_tr = da_u.data.jax_array
        v_tr = da_v.data.jax_array

        # compute wind speed
        speed = jnp.sqrt(u_tr**2 + v_tr**2)

        # ensure it's a pure scalar
        return speed.squeeze()

    # Compute gradients
    grad_fn = jax.grad(scalar_output)

    # vanilla gradient saliency
    saliency = grad_fn(inputs)

    # compute gradient × input saliency
    grad_x_input = compute_grad_times_input(grad_fn, inputs)

    # Forecast
    # predictions = run_jitted(inputs, targets * jnp.nan, forcings)

    # Save data
    os.makedirs(saliency_folder, exist_ok=True)

    saliency_path = os.path.join(saliency_folder, f"{output_name}_saliency.nc")
    grad_xinput_path = os.path.join(saliency_folder, f"{output_name}_grad_x_input.nc")
    
    save_saliency_to_netcdf(saliency, inputs, saliency_path, description="Vanilla Gradient Saliency")
    save_saliency_to_netcdf(grad_x_input, inputs, grad_xinput_path, description="Gradient × Input Saliency")

    print(f"Saliency saved to {saliency_path}")
    print(f"Gradient × Input saliency saved to {grad_xinput_path}")

    if upload_to_gcs:
        gcs.upload_file(saliency_path, saliency_path)
        gcs.upload_file(grad_xinput_path, grad_xinput_path)
        print("Files uploaded to Google Cloud Storage.")

# --------------------------------------------------------------------------

def run_forecast_with_integrated_gradients(
    model_name: str,
    input_data_or_path: str | xr.Dataset,
    output_name: str,
    target_lat: float = 54.0,
    target_lon: float = 9.0,
    lead_hours: int = 6,
    m_steps: int = 50,
    baseline: xr.Dataset | dict | None = None,
    upload_to_gcs: bool = False,
    saliency_folder: str = "../data/saliency_maps",
):
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
    def ensure_time_features(ds):
    # Add missing time/seasonal/forcing variables with correct dims
        time = ds['time']
        batch_dim = ds['batch'].shape[0] if 'batch' in ds.dims else 1
        time_dim = time.shape[0]
        lat_dim = ds['lat'].shape[0]
        lon_dim = ds['lon'].shape[0]

        # Add year_progress_sin/cos
        if 'year_progress_sin' not in ds:
            ds['year_progress_sin'] = (('batch', 'time'), np.zeros((batch_dim, time_dim)))
        if 'year_progress_cos' not in ds:
            ds['year_progress_cos'] = (('batch', 'time'), np.ones((batch_dim, time_dim)))
        # Add day_progress_sin/cos
        if 'day_progress_sin' not in ds:
            ds['day_progress_sin'] = (('batch', 'time', 'lon'), np.zeros((batch_dim, time_dim, lon_dim)))
        if 'day_progress_cos' not in ds:
            ds['day_progress_cos'] = (('batch', 'time', 'lon'), np.ones((batch_dim, time_dim, lon_dim)))
        # Add toa_incident_solar_radiation
        if 'toa_incident_solar_radiation' not in ds:
            ds['toa_incident_solar_radiation'] = (('batch', 'time', 'lat', 'lon'), np.zeros((batch_dim, time_dim, lat_dim, lon_dim)))
        return ds

    def align_climatology_to_input(climatology, input):
        # Align all coords present in both
        for coord in ['level', 'time', 'lat', 'lon']:
            if coord in climatology and coord in input:
                climatology = climatology.assign_coords({coord: input[coord].values})
        return climatology


    # Load model & normalization statics
    ckpt, bucket, dir_prefix = get_model_checkpoint(model_name)
    diffs_stddev, mean, stddev = load_normalization_data(bucket, dir_prefix)
    model_config, task_config = ckpt.model_config, ckpt.task_config
    params, state = ckpt.params, {}

    # Load input example
    if isinstance(input_data_or_path, str):
        input_data = xr.open_dataset(input_data_or_path).load()
    else:
        input_data = input_data_or_path
    input_data = ensure_time_features(input_data)

    # Extract inputs / targets / forcings
    lead_slice = slice(f"{lead_hours}h", f"{lead_hours}h")
    inputs_xr, targets, forcings = data_utils.extract_inputs_targets_forcings(
        input_data,
        target_lead_times=lead_slice,
        **dataclasses.asdict(task_config),
    )

    # Convert all input DataArrays to JAX arrays for compatibility
    inputs = {k: jnp.array(v.values) if hasattr(v, "values") else jnp.array(v) for k, v in inputs_xr.items()}

    # FORCE baseline → dict of arrays
    if isinstance(baseline, xr.Dataset):
        baseline = convert_xr_to_pytree(baseline, inputs)
    elif baseline is None:
        # default to zero baseline
        baseline = jax.tree_map(lambda x: jnp.zeros_like(x), inputs)
    elif not isinstance(baseline, dict):
        raise ValueError(f"`baseline` must be an xarray.Dataset or dict, got {type(baseline)}")

    baseline = align_climatology_to_input(baseline, input_data)

    # Build GraphCast forward
    @hk.transform_with_state
    def run_forward(mconf, tconf, _inputs, _targets, _forcings):
        pred = graphcast.GraphCast(mconf, tconf)
        pred = casting.Bfloat16Cast(pred)
        pred = normalization.InputsAndResiduals(
            pred,
            diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean,
            stddev_by_level=stddev,
        )
        predictor = autoregressive.Predictor(pred, gradient_checkpointing=True)
        return predictor(_inputs, targets_template=_targets, forcings=_forcings)

    run_fn = functools.partial(
        run_forward.apply, params, state, None, model_config, task_config
    )
    run_jitted = jax.jit(lambda inp, tgt, frc: run_fn(inp, tgt, frc)[0])

    # Find grid indices for the target
    lats = input_data["lat"].values
    lons = input_data["lon"].values
    lat_idx = int(jnp.argmin(jnp.abs(lats - target_lat)))
    lon_idx = int(jnp.argmin(jnp.abs(lons - target_lon)))
    lead_idx = 0

    # Define scalar output (wind speed at specified point)
    def scalar_output(_inputs):
        # Convert dict of arrays to xarray.Dataset using a fixed template
        input_xr_from_dict = dict_to_xr(_inputs, inputs_xr)
        preds = run_jitted(input_xr_from_dict, targets * jnp.nan, forcings)
        u = preds["10m_u_component_of_wind"].isel(
            time=lead_idx, lat=lat_idx, lon=lon_idx
        ).data.jax_array
        v = preds["10m_v_component_of_wind"].isel(
            time=lead_idx, lat=lat_idx, lon=lon_idx
        ).data.jax_array
        return jnp.sqrt(u**2 + v**2).squeeze()

    grad_fn = jax.grad(scalar_output)

    # Check baseline type and convert if necessary
    if isinstance(baseline, xr.Dataset):
        baseline = convert_xr_to_pytree(baseline, inputs)
    elif baseline is None:
        baseline = jax.tree_map(lambda x: jnp.zeros_like(x), inputs)
    elif not isinstance(baseline, dict):
        raise ValueError(f"`baseline` must be an xarray.Dataset or dict, got {type(baseline)}")

    # Defensive: If baseline is still not a dict, convert it
    if not isinstance(baseline, dict):
        raise ValueError(f"Baseline must be a dict after conversion, got {type(baseline)}")

    # Compute Integrated Gradients
    ig_attributions = compute_integrated_gradients(
        grad_fn,
        inputs,
        baseline=baseline,
        m_steps=m_steps,
    )

    # Save to NetCDF (and optionally GCS)
    os.makedirs(saliency_folder, exist_ok=True)
    out_path = os.path.join(saliency_folder, f"{output_name}_intgrad.nc")

    save_saliency_to_netcdf(ig_attributions, inputs_xr, out_path, description="Integrated Gradients Saliency")
    if upload_to_gcs:
        gcs.upload_file(out_path, out_path)

    print(f"✅ Integrated gradients saved to {out_path}")