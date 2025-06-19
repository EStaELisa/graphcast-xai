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


def get_model_checkpoint(model_name: str):
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



def load_normalization_data(bucket, dir_prefix):
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


def run_forecast(
    model_name: str,
    input_data_or_path: str | xr.Dataset,
    output_name: str,
    upload_to_gcs: bool = False,
    predictions_folder: str = "data/predictions",
    forecast_hours: int = 6
) -> str:
    """
    Run an autoregressive GraphCast forecast.

    Parameters:
    - model_name: One of "graphcast", "graphcast_operational", "graphcast_small"
    - input_data_or_path : str | xr.Dataset
        Either a path to a NetCDF file or an in-memory xarray.Dataset
    - output_name: Base name of output prediction file (saved to `predictions/` locally and optionally uploaded)

    Returns:
    - Local path to the saved prediction NetCDF file
    """
    ckpt, bucket, dir_prefix = get_model_checkpoint(model_name)
    if bucket and dir_prefix:
        diffs_stddev, mean, stddev = load_normalization_data(bucket, dir_prefix)
    else:
        raise RuntimeError("Normalization statistics not found.")

    model_config = ckpt.model_config
    task_config = ckpt.task_config
    params = ckpt.params
    state = {}

    # Load input dataset (if needed)
    if isinstance(input_data_or_path, str):
        example_batch = xr.open_dataset(input_data_or_path).load()
    else:
        example_batch = input_data_or_path

    # Extract data for one-step prediction
    lead_time_slice = slice("6h", f"{forecast_hours}h")

    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=lead_time_slice,
        **dataclasses.asdict(task_config))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=lead_time_slice,
        **dataclasses.asdict(task_config))

    construct_fn = _build_model(ckpt, diffs_stddev, mean, stddev)

    # Define model call
    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_fn(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

    # Run forecast
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * float('nan'),
        forcings=eval_forcings
    )

    # Save prediction
    os.makedirs(predictions_folder, exist_ok=True)
    output_path = os.path.join(predictions_folder, f"{output_name}.nc")
    predictions.to_netcdf(output_path)
    print(f"✅ Prediction saved to: {output_path}")

    if upload_to_gcs:
        # blob_path = f"predictions/{output_name}.nc"
        gcs.upload_file(output_path, output_path)

    return output_path
