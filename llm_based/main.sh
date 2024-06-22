#!/bin/bash

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0



# "70B", "8B", "01-34B", "4k", "128k", "llama-3-sft", "gemma-7b", "glm"

#
#python vllm_run.py \
#    --model_name='8B' \
#    --prompt_mode="single" \
#    --output_path="result/tmp" \
#    --max_new_tokens=1 \
#    --batch_size=32 \
#    --task="gossip" \
#    --VLLM_GPU_MEMORY_UTILIZATION=0.95
#
#
python vllm_run.py \
    --model_name='qwen-sft' \
    --prompt_mode="single" \
    --output_path="result/tmp" \
    --max_new_tokens=1 \
    --batch_size=32 \
    --task="gossip,all" \
    --VLLM_GPU_MEMORY_UTILIZATION=0.75

python vllm_run.py \
    --model_name='qwen' \
    --prompt_mode="single" \
    --output_path="result/tmp" \
    --max_new_tokens=1 \
    --batch_size=32 \
    --task="gossip,all" \
    --VLLM_GPU_MEMORY_UTILIZATION=0.75

#python vllm_run.py \
#    --model_name='qwen' \
#    --prompt_mode="cot" \
#    --output_path="result/tmp" \
#    --max_new_tokens=1000 \
#    --batch_size=2 \
#    --task="all" \
#    --VLLM_GPU_MEMORY_UTILIZATION=0.95


# python vllm_run.py \
#     --model_name='glm' \
#     --input_path="dataset/mer2024/emotion_pred.csv" \
#     --output_path="result/tmp_glm" \
#     --max_new_tokens=1 \
#     --batch_size=16 \
#     --task="all" \
#     --VLLM_GPU_MEMORY_UTILIZATION=0.95
# ,all
