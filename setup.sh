sudo apt update
sudo apt upgrade -y

# general python build dependencies
sudo apt-get update && sudo apt-get install -y \
  libgeos-dev \
  python3-dev \
  build-essential \
  pkg-config

# install python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update && sudo apt-get install -y \
  python3.12 \
  python3.12-venv \
  python3.12-dev

# create python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# install jupyter dependencies
pip install jax[tpu]==0.5.3
pip install jupyter==1.1.1 

# install graphcast dependencies
pip install --upgrade git+https://github.com/deepmind/graphcast.git@v0.1.1
# optimize for TPU
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

# required dependencies
pip install ipywidgets==8.1.6
pip install google-cloud-storage==3.1.0 
pip install numpy==2.2.4 
pip install --no-binary shapely shapely==2.1.0


# start jupityer lab kernel
.venv/bin/python3 -m notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8081 --NotebookApp.port_retries=0 --no-browser
