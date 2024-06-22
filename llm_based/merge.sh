eval "$(conda shell.bash hook)"

ADAPTER_MODEL_NAME_1="/home/hyt/project/fake_news_detect/llm_based/sft/0622_1522/checkpoint-7000"
MODEL_NAME_OR_PATH="Qwen/Qwen2-7B-Instruct"
SAVE_DIR="./qwen-sft"
#MODEL_NAME_OR_PATH2="NousResearch/Meta-Llama-3-8B-Instruct"
#ADAPTER_MODEL_NAME_2="../sft/result/0619/checkpoint-20000"
#SAVE_DIR_2="./llama3-sft"


conda activate myenv

cd sft/
python merge_lora.py --model_path "$MODEL_NAME_OR_PATH" --save_dir "$SAVE_DIR" --adapter_model_name "$ADAPTER_MODEL_NAME_1"
#python tools.py --model_path "$MODEL_NAME_OR_PATH2" --save_dir "$SAVE_DIR_2" --adapter_model_name "$ADAPTER_MODEL_NAME_2"

