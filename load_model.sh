##!/bin/bash

model_name=$1
model_dir=$(pwd)

echo $model_dir 

cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader

python downloader.py --name $model_name -o $model_dir/resources/models

# save only FP16 version of model

cd $model_dir/resources/models/intel/$model_name/FP16
mkdir $model_dir/resources/models/$model_name

mv * $model_dir/resources/models/$model_name

cd $model_dir/resources/models/
rm -rf intel/