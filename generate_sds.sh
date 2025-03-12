#!/bin/bash

case="generate_sds"
device=0
mkdir -p ./data/$case

# modify the text.yaml config file control the experiment
CUDA_VISIBLE_DEVICES=$device python main.py \
        --config configs/text.yaml \
        outdir="../data/$case" \
        test=False \
        save_path=$case 

CUDA_VISIBLE_DEVICES=$device python main.py \
        --config configs/text.yaml \
        outdir="../data/$case" \
        load=../data/${case}/${case}_model.ply \
        save_path=$case \
        gui=False \
        test=True \