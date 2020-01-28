#!/bin/sh

BUILD_DIR=../build
DIST_DIR=../dist

python setup.py py2app -A --bdist-base $BUILD_DIR --dist-dir $DIST_DIR \
  --resources ./resources
