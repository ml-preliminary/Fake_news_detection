#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,4,5,6,8
eval "$(conda shell.bash hook)"


# 定义参数变量
MODEL_NAME_OR_PATH="Qwen/Qwen2-7B-Instruct"  # 或 "NousResearch/Meta-Llama-3-8B-Instruct"
DATASET_NAME="/home/hyt/project/fake_news_detect/dataset/trainingset/"
OUTPUT_DIR="./0622_1522"
LORA_R=64
RESPONSE_TEMPLATE="<|im_start|>assistant"       #Llama3: "<|start_header_id|>assistant<|end_header_id|>"  qwen: "<|im_start|>assistant"
MAX_SEQ_LENGTH=5000
NUM_TRAIN_EPOCHS=1

SAVE_DIR="./qwen-sft"

conda activate myenv
cd sft/
#chmod +x ./sft.sh
./sft.sh "$MODEL_NAME_OR_PATH" "$DATASET_NAME" "$LORA_R" "$RESPONSE_TEMPLATE" "$MAX_SEQ_LENGTH" "$NUM_TRAIN_EPOCHS" "$OUTPUT_DIR"
