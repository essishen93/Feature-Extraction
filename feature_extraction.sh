#!/usr/bin/env sh
# Extract features from source image file

TEMP_OUT=data/linmao_temp
DATA=examples/linmao
IMG_SRC_ROOT=/nfs/dataset/LinMao_WebCrawl/

BATCH=113

file_name=$1
TOTAL=$2

python $DATA/fea_extract.py $DATA/${file_name} $IMG_SRC_ROOT $DATA/LinMao_VggFC7_PPS_1280.bin ${TOTAL} $BATCH
