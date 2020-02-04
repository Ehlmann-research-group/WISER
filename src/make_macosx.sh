#!/bin/sh

BUILD_DIR=../build
DIST_DIR=../dist

APP_PATH="$DIST_DIR/Imaging Spectroscopy Workbench.app"
PKG_PATH=$DIST_DIR/ISWB.pkg

set -e

python setup.py py2app -A --bdist-base $BUILD_DIR --dist-dir $DIST_DIR \
	--resources ./resources

sudo pkgbuild --install-location /Applications --component "$APP_PATH" $PKG_PATH
