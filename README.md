# GraphCast XAI

This project explores the physical interpretability of GraphCast, a state-of-the-art deep learning model for global weather forecasting developed by DeepMind. Using explainable AI (XAI) techniques, the focus is on understanding how GraphCast predicts near-surface wind speeds during Storm Zeynep (18 February 2022) in northern Germany.

The analysis combines multiple attribution methods, including feature occlusion, permutation feature importance, KernelSHAP, and integrated gradients, to identify which atmospheric variables and regions most influence the model's forecasts. The findings suggest that GraphCast's predictions are guided by physically meaningful patterns, demonstrating alignment with meteorological theory despite its black-box architecture.

This work highlights the potential of XAI to enhance transparency and trust in AI-based weather models, especially for extreme weather events.

---

# Requirements & Installation

The repository uses separate requirements files for different environments:

- **`requirements/base.txt`**: Core dependencies for running notebooks and scripts, including Jupyter, numpy, Google Cloud Storage, and GraphCast.
- **`requirements/local.txt`**: Extends `base.txt` with local (CPU/GPU) dependencies, such as `jax`.
- **`requirements/server.txt`**: Extends `base.txt` with server/TPU-specific dependencies, such as `jax[tpu]`.

## Install dependencies locally

```sh
pip install -r requirements/local.txt
```

## Install dependencies on server/TPU

```sh
pip install -r requirements/server.txt
```

You can also install the base requirements only:

```sh
pip install -r requirements/base.txt
```

---

**Note:**  

- Make sure to use the appropriate requirements file for your environment.
- Some dependencies (e.g., `shapely`) are installed from source (`--no-binary shapely`).
- The GraphCast package is installed directly from GitHub.

# Project Notebooks, Package Structure, Tests

---

## `notebooks_xai/`

This folder contains the main Jupyter notebooks for running and analyzing the XAI experiments:

- **0_run_initial_forecast.ipynb**  
  Run an initial GraphCast forecast and save results.

- **1_make_climatology.ipynb**  
  Compute ERA5 climatology datasets for baseline comparison.

- **2_feature_occlusion.ipynb**  
  Perform feature occlusion experiments and visualize their impact.

- **3_permutation_feature_importance.ipynb**  
  Run permutation feature importance (PFI) experiments and analyze results.

- **4_shap.ipynb**  
  Compute and visualize SHAP values for model interpretability.

- **5_saliency_maps.ipynb**  
  Generate and analyze saliency maps for model predictions.

---

## `pkg/`

This is the main Python package containing reusable modules for data processing, model evaluation, and XAI methods. Subfolders include:

- **attention_based/**  
  Code for attention-based interpretability (e.g., Grad-CAM).

- **feature_occlusion/**  
  Utilities for feature occlusion experiments.

- **forecast/**  
  Scripts for preparing inputs, running forecasts, and postprocessing.

- **gcs_utils/**  
  Utilities for interacting with Google Cloud Storage.

- **mask/**  
  Scripts for downloading, processing, and handling climatology and mask data.

- **pfi/**  
  Permutation feature importance computation and evaluation.

- **plots/**  
  Plotting utilities for visualizing results (feature occlusion, PFI, SHAP, etc.).

- **saliency_maps/**  
  Methods for computing and saving saliency maps.

- **shap/**  
  SHAP value computation and utilities.

---

## `tests/`

This folder contains unit and integration tests for all major modules in the `pkg/` package. The structure mirrors the main package and covers:

- **feature_occlusion/**: Tests for feature occlusion logic and forecast masking.
- **forecast/**: Tests for input preparation, model checkpointing, and forecast execution.
- **gcs_utils/**: Tests for Google Cloud Storage utilities (upload, download, listing files).
- **mask/**: Tests for mask creation, climatology computation, and data downloading.
- **pfi/**: Tests for permutation feature importance routines.
- **saliency_maps/**: Tests for saliency map and gradient computation.
- **shap/**: Tests for SHAP value computation and integration.

To run all tests, use:

```sh
pytest tests/
```

---
---

# Cloud Computing Setup

This project supports automated cloud infrastructure deployment on Google Cloud Platform (GCP) using [Pulumi](https://www.pulumi.com/). You can provision TPUs and storage buckets for large-scale or accelerated experiments.

## 1. Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`, `gsutil`)
- [Pulumi CLI](https://www.pulumi.com/docs/get-started/install/)
- GCP project with billing enabled
- Sufficient IAM permissions to create TPUs and storage buckets

## 2. Enable Required GCP APIs

Enable the Cloud TPU API for your project:

[Enable TPU API](https://console.cloud.google.com/apis/library/tpu.googleapis.com?project={GRAPHCAST_PROJECT_ID}&pli=1)

## 3. Authenticate and Set Up SSH

Authenticate with Google Cloud and set up your SSH key for OS Login:

```sh
gcloud auth login
gcloud compute os-login ssh-keys add --key-file=$HOME/.ssh/id_ed25519.pub
```

*(Adapt the path to your SSH key if needed.)*

## 4. Configure Environment Variables

Create a `.env` file in your project root with at least:

```
GRAPHCAST_BUCKET_NAME=your-bucket-name
GRAPHCAST_PROJECT_ID=your-gcp-project-id
SSH_USERNAME=your-gcp-username
SSH_KEY_PATH=/path/to/your/private/key
```

## 5. Deploy Infrastructure with Pulumi

Choose your desired configuration (TPU or bucket) by editing the appropriate `Pulumi.*.yaml` file.  
Then run:

```sh
pulumi up
```

This will:

- Provision the TPU and/or storage bucket
- Upload setup scripts, requirements, and your Python package to the server
- Run the setup scripts in order
- Install Python dependencies

## 6. Connect to the Server

After deployment, connect to your TPU (adjust zone/project as needed):

```sh
gcloud compute tpus tpu-vm ssh --zone us-east5-a graphcast-tpu --project ${GRAPHCAST_PROJECT_ID} -- -L 8081:localhost:8081
```

## 7. Access Jupyter Notebook

Start Jupyter Lab on the server (see `setup.sh` for details), then open [http://127.0.0.1:8081/tree](http://127.0.0.1:8081/tree) in your browser.

## 8. Interact with Your GCS Bucket

List your bucket contents locally:

```sh
gsutil ls gs://${GRAPHCAST_BUCKET_NAME}
```

---

**Note:**  

- All setup scripts are located in the `scripts/` directory and are executed automatically during deployment.
- Infrastructure configuration is controlled via the Pulumi YAML files (`Pulumi.tpu.yaml`, `Pulumi.bucket.yaml`).
- See `main.go` for Pulumi automation logic.

---

## Setup Scripts (`scripts/`)

This repository provides several helper scripts in the [`scripts/`](scripts/) directory to automate environment setup and system preparation, especially for cloud or TPU environments:

- **00_update.sh**  
  Updates and upgrades system packages using `apt`.

- **01_install_dependencies.sh**  
  Installs essential system and Python build dependencies, including Python 3.12 and required libraries.

- **02_create_venv.sh**  
  Creates a Python 3.12 virtual environment in `.venv` and upgrades `pip`.

- **03_optimize_for_tpu.sh**  
  Applies system settings to optimize performance for TPU usage.

### Usage

These scripts are automatically uploaded and executed on the server as part of the Pulumi deployment process (see [`main.go`](main.go)).  
You can also run them manually in order if setting up a local or cloud environment by executing:

```sh
cd scripts
bash 00_update.sh
bash 01_install_dependencies.sh
bash 02_create_venv.sh
# (activate the venv if needed)
bash 03_optimize_for_tpu.sh   # Only if using a TPU
```

**Note:**  

- Make sure you have the necessary permissions (e.g., `sudo`) to run these scripts.
- The scripts are designed for Ubuntu-based systems.

---

## Third-Party Licenses and Attribution

This project uses code and model weights from [GraphCast](https://github.com/deepmind/graphcast) by DeepMind.

- The **code** is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- The **model weights** are made available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

> This is not an officially supported Google product.  
> Copyright 2024 DeepMind Technologies Limited.
