import os
import numpy as np
import pandas as pd
import xarray as xr
import shap

from pkg.mask.apply_mask import apply_mask_to_input
from pkg.mask.generate_mask import create_feature_mask, compute_wind_speed
from pkg.forecast.run_forecast import run_forecast

def run_kernel_shap(
    features: list,
    input_ds: xr.Dataset,
    climatology_ds: xr.Dataset,
    shap_folder: str = "../data/shap",
    target_region: dict = {"lat_min":53.0,"lat_max":55.0,"lon_min":8.0,"lon_max":10.0},
    model_name: str = "graphcast_small",
    nsamples: int | str = "auto"
) -> pd.DataFrame:
    """
    Runs KernelSHAP on GraphCast’s mean 10 m wind‐speed prediction,
    saving all SHAP forecasts into `shap_folder`.
    """
    # ensure output folder exists
    os.makedirs(shap_folder, exist_ok=True)

    # align climatology and input in time
    if "time" in climatology_ds.dims:
        climatology_ds = climatology_ds.copy()
        climatology_ds["time"] = input_ds["time"]

    def predictor(masks: np.ndarray) -> np.ndarray:
        masks = np.atleast_2d(masks)
        out = np.zeros(masks.shape[0], dtype=float)

        for i, mask_vec in enumerate(masks):
            # 1) build list of features to KEEP
            keep = []
            for bit, feat in zip(mask_vec, features):
                if bit == 1:
                    if isinstance(feat, str):
                        keep.append(feat)
                    else:
                        keep.extend(feat)

            # 2) create keep‐mask (1=keep,0=mask) and invert it
            raw_keep = create_feature_mask(
                ds=input_ds,
                variables=keep,
                pressure_levels=None,
                regions=None,
                time_steps=None
            )
            mask_ds = (raw_keep == 0).astype(int)

            # 3) apply mask → climatology where mask_ds==1
            masked = apply_mask_to_input(input_ds, climatology_ds, mask_ds)

            # 4) run and SAVE forecast into shap_folder
            out_name = f"shap_pred_{i}"
            run_forecast(
                model_name=model_name,
                input_data_or_path=masked,
                output_name=out_name,
                predictions_folder=shap_folder,
                upload_to_gcs=False
            )
            pred = xr.open_dataset(os.path.join(shap_folder, out_name + ".nc"))

            # 5) compute region‐mean wind speed at last timestep
            ws = compute_wind_speed(
                pred,
                u_name="10m_u_component_of_wind",
                v_name="10m_v_component_of_wind",
                new_name="10m_wind_speed"
            )["10m_wind_speed"]
            leaf = (
                ws.isel(time=-1)
                  .where(
                      (ws.lat>=target_region["lat_min"]) &
                      (ws.lat<=target_region["lat_max"]) &
                      (ws.lon>=target_region["lon_min"]) &
                      (ws.lon<=target_region["lon_max"]),
                      drop=True
                  )
                  .mean()
                  .item()
            )
            out[i] = leaf

        return out

    # 1) background = all‐masked → zeros
    Xb = np.zeros((1, len(features)))
    expl = shap.KernelExplainer(predictor, Xb, link="identity")

    # 2) evaluation = all‐kept → ones
    Xe = np.ones((1, len(features)))
    shap_vals = expl.shap_values(Xe, nsamples=nsamples)[0]

    # 3) return DataFrame
    names = [f if isinstance(f, str) else "+".join(f) for f in features]
    return pd.DataFrame({"feature": names, "shap_value": shap_vals})

