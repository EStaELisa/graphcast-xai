import pytest
import numpy as np
import xarray as xr
from pkg.mask import generate_mask


def make_tiny_ds():
    """Creates a tiny synthetic dataset with lat/lon/time/level for testing."""
    lat = [45.0, 50.0, 55.0, 60.0]
    lon = [0.0, 5.0, 350.0, 355.0]
    time = [0, 1]
    level = [300, 500]
    shape = (1, 2, 2, 4, 4)  # batch, time, level, lat, lon

    u = np.ones(shape)
    v = np.ones(shape) * 2

    ds = xr.Dataset(
        {
            "u_component_of_wind": (("batch", "time", "level", "lat", "lon"), u),
            "v_component_of_wind": (("batch", "time", "level", "lat", "lon"), v),
        },
        coords={
            "batch": [0],
            "time": time,
            "level": level,
            "lat": lat,
            "lon": lon,
        }
    )
    return ds


# --- get_region_mask -----------------------------------------------------------------------

def test_get_region_mask_valid():
    ds = make_tiny_ds()
    mask = generate_mask.get_region_mask(ds, "north_germany")
    assert isinstance(mask, xr.DataArray)
    assert mask.dims == ("lat", "lon")


def test_get_region_mask_invalid():
    ds = make_tiny_ds()
    with pytest.raises(ValueError, match="Unknown region"):
        generate_mask.get_region_mask(ds, "invalid_region")


# --- compute_wind_speed --------------------------------------------------------------------

def test_compute_wind_speed():
    ds = make_tiny_ds()
    result = generate_mask.compute_wind_speed(ds)
    assert "wind_speed" in result
    expected = np.sqrt(1.0**2 + 2.0**2)
    np.testing.assert_allclose(result["wind_speed"].values, expected, rtol=1e-6)


# --- get_uljs_coordinates_by_time ----------------------------------------------------------

def test_get_uljs_coordinates_by_time():
    ds = make_tiny_ds()
    ds = generate_mask.compute_wind_speed(ds)
    coords = generate_mask.get_uljs_coordinates_by_time(ds, wind_threshold=1.0)
    assert isinstance(coords, dict)
    assert all(isinstance(v, list) for v in coords.values())


# --- create_feature_mask -------------------------------------------------------------------

def test_create_feature_mask_variable_filtering():
    ds = make_tiny_ds()
    mask = generate_mask.create_feature_mask(ds, variables=["u_component_of_wind"])
    assert "u_component_of_wind" in mask
    assert "v_component_of_wind" in mask
    assert (mask["v_component_of_wind"] == 0).all()


def test_create_feature_mask_pressure_filter():
    ds = make_tiny_ds()
    mask = generate_mask.create_feature_mask(ds, pressure_levels=[300])
    u_mask = mask["u_component_of_wind"]
    assert np.all((u_mask.sum("level") < u_mask.shape[1]))  # some levels masked


def test_create_feature_mask_time_masking():
    ds = make_tiny_ds()
    mask = generate_mask.create_feature_mask(ds, time_steps=[0])
    u_mask = mask["u_component_of_wind"]
    assert (u_mask.sel(time=1) == 0).all()


def test_create_feature_mask_static_region():
    ds = make_tiny_ds()
    mask = generate_mask.create_feature_mask(ds, regions=["north_germany"])
    assert "u_component_of_wind" in mask
    assert np.issubdtype(mask["u_component_of_wind"].dtype, np.integer)


def test_create_feature_mask_unknown_region():
    ds = make_tiny_ds()
    with pytest.raises(ValueError, match="Unknown region"):
        generate_mask.create_feature_mask(ds, regions=["fake_region"])
