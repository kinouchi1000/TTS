#!/bin/bash

# This script will download below
# - TTS checkpoints
#   - config.json
#   - best_model.pth.tar
# - Speaker encoder 
#   - config_se.json
#   - SE_checkpoint.pth.tar
set -e

pip install gdown 

download_path="download"

mkdir $download_path
cd $download_path

gdown --id 1nKDMwl-HvsQsKc_in_F8nLLImoFhl4cH -O config.json
gdown --id 1ASxl9SODWEjkQVM0TtU2Bhz4TmIVxtCz -O best_model.pth.tar

gdown --id  19cDrhZZ0PfKf2Zhr_ebB-QASRw844Tn1 -O config_se.json
gdown --id   17JsW6h6TIh7-LkU2EvB_gnNrPcdBxt7X -O SE_checkpoint.pth.tar

cd ..
