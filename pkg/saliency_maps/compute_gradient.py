import dataclasses
import functools
from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
import haiku as hk
import jax
import xarray as xr
import os
from pkg.gcs_utils import client as gcs
import numpy as _np

import jax.numpy as jnp


def _get_model_checkpoint(model_name: str):
    """
    Retrieves the model checkpoint for the given model name from the public GraphCast GCS bucket.

    Args:
        model_name (str): One of "graphcast", "graphcast_operational", or "graphcast_small".

    Returns:
        Tuple: (checkpoint object, bucket object, prefix str for stats files)

    Raises:
        FileNotFoundError: If no matching checkpoint is found.
    """
    gcs_client = storage.Client.create_anonymous_client()
    bucket = gcs_client.bucket("dm_graphcast")
    prefix = "graphcast/params/"

    target_prefix = model_name.lower()

    matching = [
        blob.name for blob in bucket.list_blobs(prefix=prefix)
        if blob.name.lower().startswith(prefix + target_prefix)
    ]

    if not matching:
        raise FileNotFoundError(f"❌ No matching checkpoint found for model '{model_name}'.")

    blob = bucket.blob(matching[0])
    print(f"📦 Using checkpoint: {matching[0]}")

    with blob.open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    return ckpt, bucket, "graphcast/"



def _load_normalization_data(bucket, dir_prefix):
    """
    Loads normalization statistics (mean, stddev, diffs_stddev) from the given GCS bucket.

    Args:
        bucket (google.cloud.storage.Bucket): GCS bucket object.
        dir_prefix (str): Path prefix to the stats files inside the bucket.

    Returns:
        Tuple[xr.Dataset, xr.Dataset, xr.Dataset]: diffs_stddev, mean, stddev datasets.
    """
    with bucket.blob(dir_prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev = xr.load_dataset(f).compute()
    with bucket.blob(dir_prefix + "stats/mean_by_level.nc").open("rb") as f:
        mean = xr.load_dataset(f).compute()
    with bucket.blob(dir_prefix + "stats/stddev_by_level.nc").open("rb") as f:
        stddev = xr.load_dataset(f).compute()
    return diffs_stddev, mean, stddev


def _build_model(ckpt, diffs_stddev, mean, stddev):
    """
    Builds the GraphCast model wrapped with normalization and autoregressive logic.

    Args:
        ckpt: Loaded model checkpoint.
        diffs_stddev (xr.Dataset): Normalization dataset for residuals.
        mean (xr.Dataset): Mean values for normalization.
        stddev (xr.Dataset): Stddev values for normalization.

    Returns:
        Callable: A function that constructs the predictor when given model and task configs.
    """
    def construct(model_config, task_config):
        pred = graphcast.GraphCast(model_config, task_config)
        pred = casting.Bfloat16Cast(pred)
        pred = normalization.InputsAndResiduals(
            pred,
            diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean,
            stddev_by_level=stddev
        )
        return autoregressive.Predictor(pred, gradient_checkpointing=True)

    return construct



# --------------------------------------------------------------------------

def save_saliency_to_netcdf(saliency_pytree, template_inputs, path):
    data_vars = {}
    for name, grad in saliency_pytree.items():
        # if this is a DataArray or wrapper, get its raw array first:
        arr = _np.asarray(getattr(grad, "data", grad)).astype("float32")
        data_vars[f"{name}_saliency"] = (template_inputs[name].dims, arr)
    xr.Dataset(data_vars, attrs={"description": "Vanilla gradient saliency"}).to_netcdf(path)
    print(f"✅ Saliency saved to: {path}")
    
# --------------------------------------------------------------------------
def run_forecast_with_gradients(
    model_name: str,
    input_data_or_path: str | xr.Dataset,
    output_name: str,
    target_lat: float = 54.0,
    target_lon: float = 9.0,
    lead_hours: int = 6,
    upload_to_gcs: bool = False,
    predictions_folder: str = "data/predictions",
):
    # ------------------------------------------------------------------ LOAD MODEL & STATS ---
    ckpt, bucket, dir_prefix = _get_model_checkpoint(model_name)
    diffs_stddev, mean, stddev = _load_normalization_data(bucket, dir_prefix)
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

    # ------------------------------------------------ Debugging forward pass ---
    # Run one pass outside of grad to inspect the wrapper
    preds = run_jitted(inputs, targets * jnp.nan, forcings)
    da_u10 = preds["10m_u_component_of_wind"]
    print("DataArray type:", type(da_u10))
    print("Attributes on da_u10.variable:", dir(da_u10.variable))
    print("Wrapper dir:", [a for a in dir(da_u10.data) if not a.startswith("_")])
    wrapper = da_u10.data
    for name in dir(wrapper):
        try:
            val = getattr(wrapper, name)
        except Exception:
            continue
        if "jax" in type(val).__module__ or "ArrayImpl" in type(val).__name__:
            print(f"  wrapper.{name!r}: {type(val)}")

    # … after you obtain preds and da_u10 …

    wrapper = da_u10.variable.data
    print("Wrapper type:", type(wrapper))
    print("Wrapper dir:", [a for a in dir(wrapper) if not a.startswith("_")])

    # Look for any attribute that points to a JAX array or tracer
    for name in dir(wrapper):
        try:
            val = getattr(wrapper, name)
        except Exception:
            continue
        # Print those that look like JAX arrays/tracers by type name
        if "jax" in type(val).__module__ or "DeviceArray" in type(val).__name__:
            print(f"  wrapper.{name!r} -> type: {type(val)}")


    # ------------------------------------------------ DEFINE scalar_output after debug ---
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
    saliency = grad_fn(inputs)  # this will now run after you fill in <ATTR>

    # -------------------------------------------------------------------- FORECAST
    predictions = run_jitted(inputs, targets * jnp.nan, forcings)

    # ----------------------------------------------------------- SAVE TO DISK/GCS ---
    os.makedirs(predictions_folder, exist_ok=True)
    forecast_path = os.path.join(predictions_folder, f"{output_name}.nc")
    predictions.to_netcdf(forecast_path)
    print(f"✅ Prediction saved to: {forecast_path}")

    saliency_path = os.path.join(predictions_folder, f"{output_name}_saliency.nc")
    save_saliency_to_netcdf(saliency, inputs, saliency_path)

    if upload_to_gcs:
        gcs.upload_file(forecast_path, forecast_path)
        gcs.upload_file(saliency_path, saliency_path)

    return forecast_path, saliency_path
