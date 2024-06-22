import os
import random
from typing import Any, Dict, List
import vllm
import pandas as pd
import argparse
import utils
from utils import create_custom_prompt as create_custom_prompt
from tqdm import tqdm
from datetime import datetime

RUN_SEED = int(os.getenv("RUN_SEED", 773815))


class LlamaModel:
    def __init__(self, model_name: str, max_seq_len: int = 8100, dtype: str = "bfloat16", VLLM_TENSOR_PARALLEL_SIZE=1,
                 VLLM_GPU_MEMORY_UTILIZATION=0.95):
        random.seed(RUN_SEED)
        self.model_name = model_name
        self.llm = vllm.LLM(
            self.model_name,
            max_model_len=max_seq_len,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            dtype=dtype,
            enforce_eager=False
        )
        self.tokenizer = self.llm.get_tokenizer()

    def delete(self):
        del self.llm
        del self.tokenizer

    def batch_predict(self, prompts: List[str], max_new_tokens: int = 1000) -> List[str]:
        responses = self.llm.generate(
            prompts,
            vllm.SamplingParams(
                n=1,
                top_p=0.9,
                temperature=0.9,
                seed=RUN_SEED,
                skip_special_tokens=True,
                max_tokens=max_new_tokens,
                stop="</answer>"
            ),
            use_tqdm=False
        )

        batch_response = [self.process_response(response.outputs[0].text) for response in responses]
        return batch_response

    def process_response(self, generation: str) -> str:
        generation = self.extract_content(generation).strip()
        return str(generation)

    @staticmethod
    def extract_content(s):
        start_tag = "<answer>"
        start_index = s.find(start_tag)
        if start_index == -1:
            return s
        else:
            return s[start_index + len(start_tag):]


class DataLoader:
    def __init__(self,
                 batch_size: int = None):
        self.batch_size = batch_size
        self.data_df = pd.read_csv("/home/hyt/project/fake_news_detect/dataset/raw/dataset/val/val.csv", encoding='utf-8')
        # self.data_df = pd.read_csv("/home/hyt/project/fake_news_detect/dataset/raw/dataset/val/same_distribute.csv",encoding='utf-8')
        # self.data_df = pd.read_csv("/home/hyt/project/fake_news_detect/dataset/raw/dataset/val/raw_distribute.csv", encoding='utf-8')
        # self.data_df = pd.read_csv("/home/hyt/project/fake_news_detect/dataset/raw/gossipcop_v3_origin.csv",
        #                            encoding='utf-8')
    def get_task_data(self, task):
        if 'task_name' in self.data_df.columns:
            grouped_data = self.data_df.groupby('task_name')
            if task in grouped_data.groups:
                return grouped_data.get_group(task).reset_index(drop=True)
            else:
                raise ValueError(f"No data found for task: {task}")
        else:
            raise ValueError("The 'task_name' column is not present in the data.")

    def switch_task(self, task):
        self.task_data_df = self.get_task_data(task)

    def get_batches(self):
        for i in range(0, len(self.task_data_df), self.batch_size):
            yield self.task_data_df.iloc[i:i + self.batch_size]

    def get_data(self):
        return self.task_data_df


def analyse_task(task_name: str, prompt_mode: str):
    task_list = task_name.lower().replace(' ', '').split(',')
    result_list = []
    max_new_tokens_list = []

    task_names = [
        "content_based_fake",
        "integration_based_fake",
        "story_based_fake",
        "style_based_legitimate",  # 高分
        "style_based_fake",
        "integration_based_legitimate"  # 高分
    ]
    if prompt_mode == 'cot':
        tokens_limit = 1048
    else:
        tokens_limit = 1
    if 'all' in task_list:
        task_list.remove('all')
        task_list.extend(task_names)

    for task_name in task_list:
        task_name = task_name.strip()
        if "gossip" in task_name:
            task_name="gossipcop_v3_origin"
        max_new_tokens_list.append(tokens_limit)
        result_list.append(task_name)
    return result_list, max_new_tokens_list


def main():
    parser = argparse.ArgumentParser(description="Run the pipeline model")
    parser.add_argument("--model_name", type=str, default='meta-llama/Meta-Llama-3-70B-Instruct',
                        help="Model name or path")
    parser.add_argument("--prompt_mode", type=str, default="cot", help="prompt_mode")
    parser.add_argument("--output_path", type=str, default="target_result/test", help="Path for the output path")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum number of new tokens")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--VLLM_GPU_MEMORY_UTILIZATION", type=float, default=0.94, help="VLLM_GPU_MEMORY_UTILIZATION")
    parser.add_argument("--task", type=str, default="PELD_emotion", help="task name")
    args = parser.parse_args()
    utils.set_directory(args.output_path)
    task_list, max_new_tokens_list = analyse_task(args.task, args.prompt_mode)
    print(task_list)
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    args.VLLM_TENSOR_PARALLEL_SIZE = len(cuda_visible_devices.split(',')) if cuda_visible_devices else 1

    model = LlamaModel(
        model_name=utils.get_model_path(args.model_name),
        VLLM_TENSOR_PARALLEL_SIZE=args.VLLM_TENSOR_PARALLEL_SIZE,
        VLLM_GPU_MEMORY_UTILIZATION=args.VLLM_GPU_MEMORY_UTILIZATION,
        dtype="bfloat16"
    )
    report = {}

    # Get the current time at the beginning
    time1 = datetime.now()

    data_loader = DataLoader(batch_size=args.batch_size)
    for task, max_new_tokens in zip(task_list, max_new_tokens_list):
        data_loader.switch_task(task)
        output_path = os.path.join(args.output_path, task + '.csv')

        batches = data_loader.get_batches()
        all_df = pd.DataFrame()
        for batch_df in tqdm(batches):
            if isinstance(batch_df, pd.DataFrame):
                prompts = [create_custom_prompt(args.prompt_mode, row) for _, row in batch_df.iterrows()]
                model_output = model.batch_predict(prompts, max_new_tokens=max_new_tokens)
                output_df = batch_df.copy()
                output_df["model_output"] = model_output
                all_df = pd.concat([all_df, output_df])
                utils.save_df_data(output_df, output_path=output_path)
            else:
                print(f"Batch is not a DataFrame: {batch_df}")
        result = utils.evaluate(all_df, task)
        report.update(result)

    # Get the current time at the end
    time2 = datetime.now()

    # Calculate the time cost
    time_cost = (time2 - time1).total_seconds()

    # Append the report and time information to the text file
    with open('report.txt', 'a') as f:
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y/%m/%d/%H:%M')
        f.write(f'------model: {args.model_name}----task: {task_list}-----------\n')
        f.write(f'------end_time: {formatted_time}-----all time cost: {time_cost}s------\n')
        for key, value in report.items():
            f.write(f'{key}: {value}\n')
        f.write('\n')

    print(report)


if __name__ == "__main__":
    main()
