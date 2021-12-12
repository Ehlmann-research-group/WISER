#!/bin/bash

INPUT=wiser.png
INPUT_SM=wiser_sm.png
OUTPUT_DIR=wiser.iconset

convert $INPUT_SM -resize 16x16 $OUTPUT_DIR/icon_16x16.png
convert $INPUT_SM -resize 32x32 $OUTPUT_DIR/icon_16x16@2x.png

convert $INPUT_SM -resize 32x32 $OUTPUT_DIR/icon_32x32.png
convert $INPUT_SM -resize 64x64 $OUTPUT_DIR/icon_32x32@2x.png

convert $INPUT -resize 128x128 $OUTPUT_DIR/icon_128x128.png
convert $INPUT -resize 256x256 $OUTPUT_DIR/icon_128x128@2x.png

convert $INPUT -resize 256x256 $OUTPUT_DIR/icon_256x256.png
convert $INPUT -resize 512x512 $OUTPUT_DIR/icon_256x256@2x.png

convert $INPUT -resize 512x512 $OUTPUT_DIR/icon_512x512.png
convert $INPUT -resize 1024x1024 $OUTPUT_DIR/icon_512x512@2x.png

iconutil -c icns wiser.iconset

