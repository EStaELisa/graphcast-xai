#!/bin/bash

# optimize for TPU
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"