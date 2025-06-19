import dataclasses
import functools
from graphcast import autoregressive
from graphcast import casting
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
import haiku as hk
import jax
import xarray as xr
import os
from pkg.gcs_utils import client as gcs
import numpy as _np
from jax import lax

import jax.numpy as jnp
from pkg.forecast.run_forecast import get_model_checkpoint, load_normalization_data

# --------------------------------------------------------------------------

def save_saliency_to_netcdf(saliency_pytree, template_inputs, path):
    """Save a pytree of saliency gradients to a NetCDF file.
    Parameters
    ----------
    saliency_pytree : pytree
        A pytree of gradients, e.g. the output of jax.grad.
    template_inputs : dict
        A dict mapping variable names to JAX arrays, used to determine dimensions.
    path : str
        The path where the NetCDF file will be saved.
    """
    data_vars = {}
    for name, grad in saliency_pytree.items():
        # if this is a DataArray or wrapper, get its raw array first:
        arr = _np.asarray(getattr(grad, "data", grad)).astype("float32")
        data_vars[f"{name}_saliency"] = (template_inputs[name].dims, arr)
    xr.Dataset(data_vars, attrs={"description": "Vanilla gradient saliency"}).to_netcdf(path)
    print(f"✅ Saliency saved to: {path}")
    
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
    Compute Integrated Gradients for a given grad_fn and inputs.

    Parameters
    ----------
    grad_fn : Callable
        Function mapping inputs-pytree → scalar output (e.g. wind speed).
    inputs : pytree of arrays
        Your model inputs (same structure as for grad_fn).
    baseline : pytree of arrays, optional
        Baseline/reference input. If None, uses zeros of same shape as `inputs`.
    m_steps : int
        Number of Riemann steps to approximate the integral (higher → more accurate).

    Returns
    -------
    ig_attributions : pytree of arrays
        Same structure as `inputs`, each leaf is the integrated–gradient attribution.
    """
    # 1. set baseline = zeros if not provided
    if baseline is None:
        baseline = jax.tree_map(lambda x: jnp.zeros_like(x), inputs)

    # 2. make scaled inputs: for k = 0..m_steps, alpha = k/m_steps
    #    inputs_k = baseline + alpha * (inputs - baseline)
    def interpolate(alpha):
        return jax.tree_map(lambda b, x: b + alpha * (x - b), baseline, inputs)

    # 3. compute gradients at each interpolated input
    def grad_at_alpha(k):
        alpha = k / m_steps
        inp_k = interpolate(alpha)
        return grad_fn(inp_k)

    # vectorize over k
    ks = jnp.arange(1, m_steps + 1)  # we skip k=0 to avoid zero gradient at baseline
    grads = jax.vmap(grad_at_alpha)(ks)  # this returns a pytree with leading dim=m_steps

    # 4. average gradients across the path
    #    shape: for each leaf, grads has shape [m_steps, ...]
    avg_grads = jax.tree_map(lambda g: g.mean(axis=0), grads)

    # 5. multiply by (inputs - baseline)
    ig = jax.tree_map(lambda avgg, x, b: (x - b) * avgg, avg_grads, inputs, baseline)

    return ig

# --------------------------------------------------------------------------

def compute_integrated_gradients_loop(grad_fn, inputs, baseline=None, m_steps=50):
    # 1. zero baseline if None
    if baseline is None:
        baseline = jax.tree_map(lambda x: jnp.zeros_like(x), inputs)

    # 2. accumulator for sum of grads
    init_sum = jax.tree_map(lambda x: jnp.zeros_like(x), inputs)

    def body_fun(i, carry):
        sum_grads, = carry
        alpha = (i + 1) / m_steps
        # interpolate inputs
        inp_k = jax.tree_map(lambda b, x: b + alpha * (x - b), baseline, inputs)
        gk = grad_fn(inp_k)
        sum_grads = jax.tree_map(lambda s, g: s + g, sum_grads, gk)
        return (sum_grads,)

    (sum_grads,) = lax.fori_loop(0, m_steps, body_fun, (init_sum,))

    # average and scale
    avg_grads = jax.tree_map(lambda s: s / m_steps, sum_grads)
    ig = jax.tree_map(lambda ag, x, b: (x - b) * ag, avg_grads, inputs, baseline)
    return ig

# --------------------------------------------------------------------------
def convert_xr_to_pytree(xr_ds: xr.Dataset, inputs_template: dict) -> dict:
    """
    Convert an xarray.Dataset to the same dict‐pytree structure as `inputs_template`.
    Any variable missing in `xr_ds` is filled with zeros of the right shape.
    """
    pytree = {}
    for key, arr in inputs_template.items():
        shape = arr.shape
        if key in xr_ds:
            # use climatology value
            pytree[key] = jnp.array(xr_ds[key].values)
        else:
            # fill missing variable with zeros
            pytree[key] = jnp.zeros(shape, dtype=arr.dtype)
    return pytree


# --------------------------------------------------------------------------


def run_forecast_with_gradients(
    model_name: str,
    input_data_or_path: str | xr.Dataset,
    output_name: str,
    target_lat: float = 54.0,
    target_lon: float = 9.0,
    lead_hours: int = 6,
    m_steps: int = 50,
    baseline: xr.Dataset | None = None,  
    upload_to_gcs: bool = False,
    predictions_folder: str = "data/predictions",
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
    baseline : xr.Dataset or None
        Optional baseline dataset for integrated gradients. If None, zeros will be used.
    upload_to_gcs : bool
        If True, upload the output files to Google Cloud Storage.
    predictions_folder : str
        Local folder where the predictions and saliency maps will be saved.     
    """
    # ------------------------------------------------------------------ LOAD MODEL & STATS ---
    ckpt, bucket, dir_prefix = get_model_checkpoint(model_name)
    diffs_stddev, mean, stddev = load_normalization_data(bucket, dir_prefix)
    model_config, task_config = ckpt.model_config, ckpt.task_config
    params, state = ckpt.params, {}

    # ----------------------------------------------------------- LOAD INPUT EXAMPLE ---
    if isinstance(input_data_or_path, str):
        example_batch = xr.open_dataset(input_data_or_path).load()
    else:
        example_batch = input_data_or_path

    lead_slice = slice(f"{lead_hours}h", f"{lead_hours}h")  # single step
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=lead_slice, **dataclasses.asdict(task_config)
    )

    if baseline is not None and isinstance(baseline, xr.Dataset):
        baseline = convert_xr_to_pytree(baseline, inputs)

    # --------------------------------------------------------- GRAPHCAST FORWARD ---
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

    # ---------------------------------------------- COMPUTE GRID INDICES ---
    lats = example_batch["lat"].values
    lons = example_batch["lon"].values
    lat_idx = int(jnp.argmin(jnp.abs(lats - target_lat)))
    lon_idx = int(jnp.argmin(jnp.abs(lons - target_lon)))
    lead_idx = 0  # single step

    # ------------------------------------------------ DEFINE scalar_output ---
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

    # ------------------------------------------------------------- COMPUTE GRADIENTS

    grad_fn = jax.grad(scalar_output)

    # vanilla gradient saliency
    saliency = grad_fn(inputs)

    # compute gradient × input saliency
    grad_x_input = compute_grad_times_input(grad_fn, inputs)

    # integrated gradients
    ig_attributions = compute_integrated_gradients_loop(
        grad_fn,
        inputs,
        baseline=baseline,
        m_steps=m_steps     
    )

    # -------------------------------------------------------------------- FORECAST
    predictions = run_jitted(inputs, targets * jnp.nan, forcings)

    # ----------------------------------------------------------- SAVE TO DISK/GCS ---
    os.makedirs(predictions_folder, exist_ok=True)
    forecast_path = os.path.join(predictions_folder, f"{output_name}.nc")
    predictions.to_netcdf(forecast_path)

    saliency_path = os.path.join(predictions_folder, f"{output_name}_saliency.nc")
    grad_xinput_path = os.path.join(predictions_folder, f"{output_name}_grad_x_input.nc")
    ig_path = os.path.join(predictions_folder, f"{output_name}_intgrad.nc")
    save_saliency_to_netcdf(saliency, inputs, saliency_path)
    save_saliency_to_netcdf(grad_x_input, inputs, grad_xinput_path)
    save_saliency_to_netcdf(ig_attributions, inputs, ig_path)

    if upload_to_gcs:
        gcs.upload_file(forecast_path, forecast_path)
        gcs.upload_file(saliency_path, saliency_path)
        gcs.upload_file(grad_xinput_path, grad_xinput_path)
        gcs.upload_file(ig_path, ig_path)
