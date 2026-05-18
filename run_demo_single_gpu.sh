#!/bin/bash

#PROJECT_DIR="$HOME/Project"
#LTX2V_DIR="$PROJECT_DIR/LightX2V"
#export PYTHONPATH="$LTX2V_DIR:$PYTHONPATH"

#VENV_DIR=.venv
#LAUNCH_BIN="$VENV_DIR/bin/torchrun"
MODEL_PATH="./models/LiveAct"

USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 --master_port=$(shuf -n 1 -i 10000-65535)  \
    demo.py \
    --size 416*720 \
    --ckpt_dir "$MODEL_PATH" \
    --wav2vec_dir ./models/chinese-wav2vec2-base \
    --video_save_path ./generated_videos \
    --fp8_kv_cache \
#    --block_offload \
#    --t5_cpu
#    --fps 20 \
#    --dura_print \
#    --input_json examples/example.json \
#    --steam_audio
#    --video_save_path ./generated_videos