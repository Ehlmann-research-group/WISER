#!/bin/bash

INPUT=wiser.png
INPUT_SM=wiser_sm.png
OUTPUT_DIR=wiser-ico

convert $INPUT_SM -resize 16x16 $OUTPUT_DIR/wiser_16.bmp
convert $INPUT_SM -resize 32x32 $OUTPUT_DIR/wiser_32.bmp
convert $INPUT_SM -resize 64x64 $OUTPUT_DIR/wiser_64.bmp

convert $INPUT -resize 128x128 $OUTPUT_DIR/wiser_128.bmp
convert $INPUT -resize 256x256 $OUTPUT_DIR/wiser_256.bmp

convert $OUTPUT_DIR/wiser_16.bmp $OUTPUT_DIR/wiser_32.bmp \
	$OUTPUT_DIR/wiser_64.bmp $OUTPUT_DIR/wiser_128.bmp \
	$OUTPUT_DIR/wiser_256.bmp $OUTPUT_DIR/wiser.ico

