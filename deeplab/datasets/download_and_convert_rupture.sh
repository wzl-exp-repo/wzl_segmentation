#!/bin/bash

# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to download and preprocess the PASCAL VOC 2012 dataset.
#
# Usage:
#   bash ./download_and_convert_voc2012.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_voc2012_data.py
#     - download_and_convert_voc2012.sh
#     - remove_gt_colormap.py
#     + pascal_voc_seg
#       + VOCdevkit
#         + VOC2012
#           + JPEGImages
#           + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./WZL_rupture_seg"
#WORK_DIR="./pascal_voc_seg"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack VOC 2012 dataset.

download_and_uncompress(){
	local BASE_URL=${1}
	local FILENAME=${2}
	
	if [ ! -f "${FILENAME}" ]; then
        	echo "Downloading ${FILENAME} to ${WORK_DIR}"
	 	wget -nd -c "${BASE_URL}/${FILENAME}"
	fi
	echo "Uncompressing ${FILENAME}"
	tar -xf "${FILENAME}"


}

copy_and_uncompress(){

  local SRC_BASE_URL=${1}
  local SRC_FILENAME=${2}
  local DEST_BASE_URL=${3}
  
  
  
  if [ ! -f "${SRC_FILENAME}" ]; then
    echo "Coping ${SRC_FILENAME} to ${WORK_DIR}"
  
    cp -R "${SRC_BASE_URL}/${SRC_FILENAME}" .
  fi
   echo "Uncompressing ${SRC_FILENAME} in directory ${WORK_DIR}" 
  
  
   tar -xf "${SRC_FILENAME}"  

}





# Download the images.

SRC_BASE_URL="/home/doniyor/Desktop/WZL_Einarbeitung/Daten/image"
SRC_FILENAME="WZL_rupture_dataset.tar"

DEST_BASE_URL="/home/doniyor/Desktop/WZL_Einarbeitung/tensorflow/research/deeplab/datasets"



copy_and_uncompress "${SRC_BASE_URL}" "${SRC_FILENAME}" "${DEST_BASE_URL}"
#download_and_uncompress "${BASE_URL}" "${FILENAME}"

#echo "Test----------------------------"

cd "${CURRENT_DIR}"

# Create train, val and train_val txt files

python ./create_train_test_lists.py

# Root path for  dataset.
RUPTURE_ROOT="${WORK_DIR}/WZL_rupture_dataset/Rupture" 

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${RUPTURE_ROOT}/SegmentationClass"
SEMANTIC_SEG_FOLDER="${RUPTURE_ROOT}/SegmentationClassRaw"

echo "Removing the color map in ground truth annotations..."
python ./remove_gt_colormap.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"



# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${RUPTURE_ROOT}/JPEGImages"
LIST_FOLDER="${RUPTURE_ROOT}/ImageSets/Segmentation"


echo "Converting Rupture 2019 dataset..."
python ./build_rupture_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="png" \
  --output_dir="${OUTPUT_DIR}"
echo "Finish Download and Convert part"

