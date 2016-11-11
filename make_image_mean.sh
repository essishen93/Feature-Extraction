#!/usr/bin/env sh
# Convert original images to data with format, lmdb
# Compute the mean image from the lmdb

TEMP_OUT=data/linmao_temp
DATA=examples/linmao
TOOLS=build/tools

IMG_SRC_ROOT=/nfs/dataset/LinMao_WebCrawl/

file_name=$1

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$IMG_SRC_ROOT" ]; then
  echo "Error: IMG_SRC_ROOT is not a path to a directory: $IMG_SRC_ROOT"
  echo "Set the IMG_SRC_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating lmdb data…"

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $IMG_SRC_ROOT \
    $DATA/${file_name} \
    $TEMP_OUT/linmao_lmdb

echo "Computing image mean…"

$TOOLS/compute_image_mean $TEMP_OUT/linmao_lmdb \
  $TEMP_OUT/linmao_mean.binaryproto

echo "Changing to .npy…"
python $DATA/bin_to_npy.py $TEMP_OUT/linmao_mean.binaryproto $TEMP_OUT/linmao_mean_12000.npy

echo "Deleting temporary files…"
rm -r $TEMP_OUT/linmao_lmdb
rm $TEMP_OUT/linmao_mean.binaryproto

echo "Done."
