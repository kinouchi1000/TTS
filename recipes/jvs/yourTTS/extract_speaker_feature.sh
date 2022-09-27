#!/bin/bash

set -e
python ../../../TTS/bin/compute_embeddings.py \
    --model_path ${PWD}/download/SE_checkpoint.pth.tar \
    --config_path ${PWD}/download/config_se.json \
    --config_dataset_path ${PWD}/download/dataset_config.json \
    --output_path exp/jvs_vctk_speakers.json
    # --disable_cuda true