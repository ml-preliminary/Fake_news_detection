import numpy as np
import xml.etree.ElementTree as ET
import shutil
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
import re
def calculate_euclidean_distance(values1, values2):
    """计算两个列表之间的欧氏距离"""
    return np.sqrt(np.sum((np.array(values1) - np.array(values2)) ** 2))

def convert_score_to_eq(x, mean=2.79, sd=0.822, target_mean=100, target_sd=15):
    """将原始分数转换为eq值"""
    z_score = (x - mean) / sd
    # 将Z分数转换为目标正态分布上的分数（这里以100为均值，15为标准差的正态分布为例）
    eq_score = - z_score * target_sd + target_mean
    return eq_score

def get_model_path(model_name: str) -> str:
    username = os.getlogin()
    if username == 'hyt':
        paths = {
            "70B": "/home/hyt/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-70B-"
                                                    "Instruct/snapshots/84cbdcd4bcccad50126c29ec7f7a476dec014fcf",
            "8B": "/home/hyt/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-"
                                                   "Instruct/snapshots/3cf58932fb9b7257157a1a7e4b4cf0a469b069ba",
            "01-34B": "/home/hyt/.cache/huggingface/hub/models--01-ai--Yi-34B-Chat/snapshots/9879dbaa4ba7030faefafed1866d1ca5a8c091f1",
            "4k": "/home/hyt/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/4b6011c87354f042a116f4567ecaac71e144afe4",
            "128k": "/home/hyt/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/ebee18c488086b396dde649f2aa6548b9b8d2404",
            "llama-3-sft": "/home/hyt/EI_explore/functions/llama3-sft",
            "gemma-7b": "/home/hyt/.cache/huggingface/hub/models--google--gemma-7b-it/snapshots/18329f019fb74ca4b24f97371785268543d687d2",
            "glm": "/home/hyt/.cache/huggingface/hub/models--THUDM--glm-4-9b-chat/snapshots/b84dc74294ccd507a3d78bde8aebf628221af9bd",
            "qwen": "/home/hyt/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/41c66b0be1c3081f13defc6bdf946c2ef240d6a6",
            "qwen-sft": "/home/hyt/project/fake_news_detect/llm_based/sft/qwen-sft"
        }
        return paths.get(model_name, model_name)
    else:
        # Return the model name itself if the username is not 'hyt'
        return model_name


def set_directory(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        try:
            os.makedirs(path)
            print(f'Created new directory: {path}')
        except Exception as e:
            print(f'Failed to create directory {path}. Reason: {e}')
prompt_set = {
    "single": (
        "You are a trained fake news detection expert. Your task is to detect whether a news fragment is real or fake."
        " Below you will be given a news fragment:\n\n"
        "News:\n{content}\n"
        "Instruction:\n\n"
        "Please use four labels to classify the type of this news sample. 0 represents real news, "
        "1 represents fake news. Your answer should be between 0 and 1."
        "Now please organize your judgment using the following format:"
        "<answer>put only one number here</answer>\nAnswer:  <answer>"
    ),
    "cot": (
        "You are a trained fake news detection expert. Your task is to detect whether a news fragment is real or fake."
        " Below you will be given a news fragment:\n\n"
        "News:\n{content}\n"
        "Instruction:\n\n"
        "Please use four labels to classify the type of this news sample. 0 represents real news, "
        "1 represents fake news. Your answer should be between 0 and 1."
        "Think step by step, you might find clues from some details. Write down your idea and explaination."
        "You answer should be organized within the following format:"
        "<explain></explain><answer>put only one number here</answer>\n\nAnswer:  <explain>"
    )    
}

def create_custom_prompt(prompt_mode, row) -> str:
    prompt = prompt_set[prompt_mode].format(content=row['question'])
    length = len(prompt)
    prompt = prompt[length-8095:]
    return prompt






def evaluate(df: pd.DataFrame, task):
    err_parse = 0
    total_predictions = []

    for _, row in df.iterrows():
        try:
            number = int(row['model_output'])
        except ValueError:
            err_parse += 1
            total_predictions.append(-1)
            continue

        if number not in range(0, 2):
            err_parse += 1
            total_predictions.append(-1)
            continue

        total_predictions.append(number)

    # Remove invalid predictions
    valid_indices = [i for i, pred in enumerate(total_predictions) if pred != -1]
    try:
        valid_labels = df['answer'].iloc[valid_indices].astype(int)
    except ValueError:
        # If direct conversion fails, use regex to extract the number
        valid_labels = df['answer'].iloc[valid_indices].apply(
            lambda x: int(re.search(r'(\d+)</answer>', x).group(1)))
    valid_predictions = [total_predictions[i] for i in valid_indices]

    # Calculate overall accuracy
    total_acc = accuracy_score(valid_labels, valid_predictions)

    # Calculate real_acc (label is 0 and prediction is correct)
    real_correct = sum((valid_labels == 0) & (valid_labels == valid_predictions))
    real_total = sum(valid_labels == 0)
    real_acc = real_correct / real_total if real_total > 0 else 0

    # Calculate fake_acc (label is 1 and prediction is correct)
    fake_correct = sum((valid_labels == 1) & (valid_labels == valid_predictions))
    fake_total = sum(valid_labels == 1)
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0

    # Calculate other metrics
    total_f1 = f1_score(valid_labels, valid_predictions, average='weighted', zero_division=0)
    total_recall = recall_score(valid_labels, valid_predictions, average='weighted', zero_division=0)

    result = {
        task: {
            'all_acc': round(total_acc, 3),
            'real_acc': round(real_acc, 3),
            'fake_acc': round(fake_acc, 3),
            'f1_score': round(total_f1, 3),
            'recall': round(total_recall, 3),
            'error_parse': err_parse
        }
    }
    print(result)
    return result

        
def save_df_data(df: pd.DataFrame, output_path: str):
    columns_to_keep = ['answer', 'model_output']
    df = df[columns_to_keep]
    if os.path.isfile(output_path):
        existing_df = pd.read_csv(output_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(output_path, index=False, encoding='utf-8')