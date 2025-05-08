#!/bin/bash

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