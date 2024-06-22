#!/bin/bash
# 使用传入的参数
MODEL_NAME_OR_PATH=${1}
DATASET_NAME=${2}
LORA_R=${3}
RESPONSE_TEMPLATE=${4}
MAX_SEQ_LENGTH=${5}
NUM_TRAIN_EPOCHS=${6}
OUTPUT_DIR=${7}

accelerate launch --config_file "deepspeed_config.yaml" train.py \
--seed 100 \
--model_name_or_path "$MODEL_NAME_OR_PATH" \
--dataset_name "$DATASET_NAME" \
--response_template "$RESPONSE_TEMPLATE" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_length "$MAX_SEQ_LENGTH" \
--num_train_epochs "$NUM_TRAIN_EPOCHS" \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "epoch" \
--evaluation_strategy "epoch" \
--save_strategy "steps" \
--push_to_hub no \
--hub_private_repo no \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-5 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.1 \
--max_grad_norm 1.0 \
--output_dir "$OUTPUT_DIR" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r "$LORA_R" \
--lora_alpha 16 \
--lora_dropout 0.05 \
--lora_target_modules "all-linear" \
--use_4bit_quantization False
