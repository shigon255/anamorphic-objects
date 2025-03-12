#!/bin/bash

case=$1
angle=$(echo $case | grep -oE '[0-9]+$')
echo "Visualizing $case"
echo "Angle: $angle"

# SDS
echo "Visualizing SDS"
SDS_video_path="$case/generate_sds/generate_sds_gs_360.mp4"
SDS_output_path="$case/generate_sds/visualize_sds.png"
python visualize_result.py \
    --input_video $SDS_video_path \
    --reverse \
    --angle $angle \
    --output_path $SDS_output_path
