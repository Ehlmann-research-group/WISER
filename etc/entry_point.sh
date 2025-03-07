#!usr/bin/env bash
set -e

conda run -n wiser-source /bin/bash -c "
cd /WISER
make generated
cd src/tests
pytest ."
