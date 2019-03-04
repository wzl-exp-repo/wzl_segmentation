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
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_ test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/WZ_model_test.py -v

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
    DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"
sh WZ_download_and_convert_rupture.sh





# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
RUPTURE_FOLDER="WZL_rupture_seg"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${RUPTURE_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${RUPTURE_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${RUPTURE_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${RUPTURE_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${RUPTURE_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained  checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"                #Heavy pretrained model

#TF_INIT_CKPT="deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz"                               #Light pretrained model
cd "${INIT_FOLDER}"
wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"



RUPTURE_DATASET="${WORK_DIR}/${DATASET_DIR}/${RUPTURE_FOLDER}/tfrecord"

echo "=========================== Begin the training ============================"

# Train 10 iterations.
NUM_ITERATIONS=1 #10


python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --dataset="rupture" \
  --model_variant="xception_65" \
  --output_stride=16 \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --fine_tune_batch_norm=False \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=4 \
  --decoder_output_stride=4 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${RUPTURE_DATASET}" \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True




#--model_variant="xception_65" \

#--atrous_rates=6 \
#  --atrous_rates=12 \     only
#  --atrous_rates=18 \     for
#                          xception65

#--fine_tune_batch_norm=False \
#--last_layers_contain_logits_only = False
#--decoder_output_stride=4 \
#--tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \               #Heavy pretrained model

echo "=========================== End of  the training ============================"


# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.

echo "=========================== Begin of the evaluation ============================"

python "${WORK_DIR}"/WZ_eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --decoder_output_stride=4 \
  --eval_crop_size=2454 \
  --eval_crop_size=2454 \
  --fine_tune_batch_norm=False \
  --dataset="rupture" \
  --checkpoint_dir="${TRAIN_LOGDIR3}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${RUPTURE_DATASET}" \
  --max_number_of_evaluations=1




#--model_variant="xception_65" \
#--atrous_rates=6 \
#--atrous_rates=12 \
#--atrous_rates=18 \
# --decoder_output_stride=4 \
#--eval_batch_size=1 \
echo "=========================== End of the evaluation ============================"

echo "============================Begin of the Visualization=========================="

# Visualize the results.
python "${WORK_DIR}"/WZ_vis.py \
  --logtostderr \
  --vis_split="val" \
  --dataset="rupture" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=4 \
  --output_stride=16 \
  --vis_crop_size=2454 \
  --vis_crop_size=2454 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${RUPTURE_DATASET}" \
  --max_number_of_iterations=1


#--model_variant="xception_65" \
#--atrous_rates=6 \
#--atrous_rates=12 \
#--atrous_rates=18 \
# --decoder_output_stride=4 \
echo "=========================== End of the Visualization =================-========================"


echo "=========================== Begin of the export of retrained model ============================"

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/WZ_export_model.py \
  --logtostderr \
  --dataset="rupture" \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=2 \
  --crop_size=2454 \
  --crop_size=2454 \
  --inference_scales=1.0


#--model_variant="xception_65" \
#--atrous_rates=6 \
#--atrous_rates=12 \
#--atrous_rates=18 \
# --decoder_output_stride=4 \
echo "=========================== End of the exporting of retrained model ============================"


# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.


