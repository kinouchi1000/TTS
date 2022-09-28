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

gdown --id  1Izs9dWYS1ENwgq4Sok5MCz5fxPF6zgZq -O jvs_vctk_speakers.json
gdown --id  1ZInyIG-_WUv4eMxLZh-Z_7AGZZG7SxVU -O config_se.json
gdown --id  1_goS_TYOe_JWn1_rBq01wQAgDPwNFhVu -O SE_checkpoint.pth.tar
gdown --id  1WPuF-oHREVUi_b3JeWZH8mXBPT9Td9uh -O jvs_spekaers.json

cd ..
