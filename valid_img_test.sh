#!/usr/bin/env sh
# Extract features from source image file

DATA=examples/linmao
IMG_SRC_ROOT=/nfs/dataset/LinMao_WebCrawl/

file_name=$1

python $DATA/is_valid_image.py $DATA/${file_name} $IMG_SRC_ROOT