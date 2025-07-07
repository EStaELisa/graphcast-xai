import pytest
import numpy as np
import xarray as xr
import jax.numpy as jnp
from pkg.saliency_maps import compute_gradient


def test_compute_grad_times_input():
    inputs = {
        "temp": jnp.array([1.0, 2.0, 3.0]),
        "v": jnp.array([4.0, 5.0, 6.0])
    }

    def dummy_grad_fn(inputs):
        return {
            "temp": jnp.array([0.1, 0.2, 0.3]),
            "v": jnp.array([0.4, 0.5, 0.6])
        }

    result = compute_gradient.compute_grad_times_input(dummy_grad_fn, inputs)
    assert np.allclose(result["temp"], jnp.array([0.1, 0.4, 0.9]))
    assert np.allclose(result["v"], jnp.array([1.6, 2.5, 3.6]))


def test_compute_integrated_gradients_zero_baseline():
    inputs = {
        "temp": jnp.array([1.0, 2.0]),
    }

    def dummy_grad_fn(x):
        return {"temp": x["temp"]}

    ig = compute_gradient.compute_integrated_gradients(dummy_grad_fn, inputs, m_steps=10)
    # Integral approximation: sum_{i=1}^{10} (i/10) = 0.55, 1.1
    expected = jnp.array([0.55, 2.2])  # because input = [1.0, 2.0] and mean gradient = same
    assert np.allclose(ig["temp"], expected, rtol=1e-2)


def test_convert_xr_to_pytree_handles_missing():
    inputs_template = {"temp": jnp.ones((2, 2)), "v": jnp.ones((2, 2))}
    xr_ds = xr.Dataset({"temp": (("x", "y"), np.array([[1, 2], [3, 4]]))})

    result = compute_gradient.convert_xr_to_pytree(xr_ds, inputs_template)
    assert "temp" in result and "v" in result
    assert np.allclose(result["temp"], [[1, 2], [3, 4]])
    assert np.all(result["v"] == 0)


def test_dict_to_xr_raises_on_missing_key():
    template = xr.Dataset({
        "temp": xr.DataArray(np.zeros((2, 2)), dims=("x", "y"), coords={"x": [0, 1], "y": [0, 1]})
    })
    inputs_dict = {"wrong_var": jnp.zeros((2, 2))}

    with pytest.raises(ValueError):
        compute_gradient.dict_to_xr(inputs_dict, template)
