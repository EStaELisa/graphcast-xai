import pytest
import numpy as np
import xarray as xr
from pkg.mask import apply_mask


def make_dummy_dataset(var_names, shape=(1, 2, 2), dims=("time", "lat", "lon")):
    data = {}
    coords = {}
    for dim, size in zip(dims, shape):
        coords[dim] = list(range(size))

    for var in var_names:
        data[var] = (dims, np.ones(shape, dtype=np.float32))
    return xr.Dataset(data, coords=coords)


def test_ensure_required_climatology_vars_adds_missing():
    input_ds = make_dummy_dataset(["var1", "var2"])
    climatology = make_dummy_dataset(["var1"])  # missing var2

    result = apply_mask.ensure_required_climatology_vars(climatology, input_ds)

    assert "var1" in result
    assert "var2" in result
    np.testing.assert_array_equal(result["var2"].values, 0)


def test_apply_mask_replaces_with_climatology():
    input_ds = make_dummy_dataset(["temp"])
    climatology = make_dummy_dataset(["temp"])
    climatology["temp"][:] = 999  # unique value to spot replacement
    mask = make_dummy_dataset(["temp"])
    mask["temp"][:] = [[0, 1], [1, 0]]  # mask some elements

    result = apply_mask.apply_mask_to_input(input_ds, climatology, mask)
    out = result["temp"].values

    # Where mask == 0 -> original value (1)
    # Where mask == 1 -> climatology value (999)
    expected = np.array([[[1, 999], [999, 1]]], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)


def test_apply_mask_raises_on_variable_mismatch():
    input_ds = make_dummy_dataset(["temp"])
    climatology = make_dummy_dataset(["temp"])
    mask = make_dummy_dataset(["humidity"])  # different var name

    with pytest.raises(ValueError, match="same variables"):
        apply_mask.apply_mask_to_input(input_ds, climatology, mask)


def test_apply_mask_raises_on_shape_mismatch():
    input_ds = make_dummy_dataset(["temp"], shape=(1, 2, 2))
    climatology = make_dummy_dataset(["temp"])
    mask = make_dummy_dataset(["temp"], shape=(1, 1, 1))  # wrong shape

    with pytest.raises(ValueError, match="Shape mismatch"):
        apply_mask.apply_mask_to_input(input_ds, climatology, mask)


def test_apply_mask_raises_on_dim_mismatch():
    input_ds = make_dummy_dataset(["temp"], dims=("time", "lat", "lon"))
    climatology = make_dummy_dataset(["temp"])
    mask = make_dummy_dataset(["temp"], dims=("lat", "lon", "time"))  # wrong dim order

    with pytest.raises(ValueError, match="Dimension mismatch"):
        apply_mask.apply_mask_to_input(input_ds, climatology, mask)


def test_apply_mask_handles_time_mismatch():
    input_ds = make_dummy_dataset(["temp"])
    climatology = make_dummy_dataset(["temp"])
    climatology["time"] = [12345]  # different time
    mask = make_dummy_dataset(["temp"])

    result = apply_mask.apply_mask_to_input(input_ds, climatology, mask)

    # Should not raise, and time coord should match input
    assert result["time"].equals(input_ds["time"])
