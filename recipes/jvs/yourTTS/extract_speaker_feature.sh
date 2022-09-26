#!/bin/bash

set -e
python ../../../TTS/bin/compute_embeddings.py \
    --model_path ${PWD}/download/SE_checkpoint.pth.tar \
    --config_path ${PWD}/download/config_se.json \
    --dataset_name jvs \
    --formatter_name jvs \
    --dataset_path ${PWD}/../jvs_ver1 \
    --output_path exp/spekaers.json