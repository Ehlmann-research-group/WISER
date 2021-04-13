#!/bin/sh

# PYTHON_OPTS="-m cProfile -o profile_stats"
PYTHON_OPTS=

python $PYTHON_OPTS -m wiser $@
