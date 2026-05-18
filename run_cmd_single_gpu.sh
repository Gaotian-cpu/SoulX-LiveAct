#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 --master_port=$(shuf -n 1 -i 10000-65535) \
    generate.py \
    --ckpt_dir ./models/LiveAct \
    --wav2vec_dir ./models/chinese-wav2vec2-base \
    --size 416*720 \
    --fps 20 \
    --audio_cfg 0.8 \
    --input_json examples/ghb_test/example.json \
    --fp8_kv_cache \
    --block_offload \
    --t5_cpu