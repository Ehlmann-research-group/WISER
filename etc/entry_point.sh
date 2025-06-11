#!/usr/bin/env bash
set -e

# Start Xvfb in the background
Xvfb :1 -screen 0 1024x768x16 &

# Set DISPLAY to use the Xvfb server
export DISPLAY=:1

# Wait a moment for Xvfb to initialize
sleep 2

# Run your commands within the conda environment
conda run -n wiser-source /bin/bash -c "
cd /WISER
make generated
cd src/tests"
