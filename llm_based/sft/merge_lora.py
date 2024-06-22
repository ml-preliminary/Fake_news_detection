import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def set_working_directory_to_script_location():
    # 设置当前工作目录为脚本所在目录
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)

def run_(model_path, save_dir, adapter_model_name):
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_path), adapter_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    new_base_model = model.merge_and_unload()
    tokenizer.save_pretrained(save_dir)
    new_base_model.save_pretrained(save_dir, max_shard_size='10GB')

def main():
    set_working_directory_to_script_location()

    parser = argparse.ArgumentParser(description='Run model merging and saving.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the merged model')
    parser.add_argument('--adapter_model_name', type=str, required=True, help='Name of the adapter model')

    args = parser.parse_args()

    run_(args.model_path, args.save_dir, args.adapter_model_name)

if __name__ == "__main__":
    main()
