import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


# Function to process CSV files and convert them to question-answer format
def process_csv_files(filepath):
    df = pd.read_csv(filepath)
    print(f"Processed {filepath}")
    return df


def stratified_split(df, test_size, random_state):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[['answer', 'task_name']])


def split_data_per_task(df, random_state=42):
    train_frames = []
    test_frames = []
    val_frames = []

    for task_name, group in df.groupby('task_name'):
        # 按照任务名称分组并获取每个任务的正负样本比例
        print(f"\nTask: {task_name}")
        label_distribution = group['answer'].value_counts(normalize=True).to_dict()
        print(f"Label distribution: {label_distribution}")

        # 根据比例计算需要的样本数
        train_data, temp_data = train_test_split(group, test_size=0.85, random_state=random_state,
                                                 stratify=group['answer'])
        test_data, val_data = train_test_split(temp_data, test_size=0.95, random_state=random_state,
                                               stratify=temp_data['answer'])

        train_frames.append(train_data)
        test_frames.append(test_data)
        val_frames.append(val_data)

    train_data = pd.concat(train_frames).reset_index(drop=True)
    test_data = pd.concat(test_frames).reset_index(drop=True)
    val_data = pd.concat(val_frames).reset_index(drop=True)
    print(f"len(train,test,val):{len(train_data)},{len(test_data)},{len(val_data)}")
    return train_data, test_data, val_data


# Main function to process all files and save the splits
def main_split():
    input_filepath = './raw.csv'
    output_directory = 'dataset'

    all_data = process_csv_files(input_filepath)
    # 计算原始数据中的比例
    all_data['numeric_answer'] = all_data['answer'].apply(lambda x: int(x[0]))
    original_label_ratio = all_data['numeric_answer'].mean()
    original_task_distribution = all_data['task_name'].value_counts(normalize=True)
    all_data = all_data[["news_id", "question", "answer", "task_name"]]
    all_data['answer'] = all_data['answer'].apply(lambda x: str(x)[0]+"</answer>")

    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    all_data = all_data.sample(frac=1, random_state=26).reset_index(drop=True)
    all_data = all_data.sample(frac=1, random_state=58).reset_index(drop=True)
    # 进行多次打乱
    train_data, test_data, val_data = split_data_per_task(all_data)
    train_data = train_data.sample(frac=1, random_state=73).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=74).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=86).reset_index(drop=True)

    # 验证分割后的数据集是否保持了原始比例
    for dataset, name in zip([train_data, test_data, val_data], ['Train', 'Test', 'Validation']):
        label_ratio = dataset['answer'].apply(lambda x: int(x.strip('</answer>'))).mean()
        task_distribution = dataset['task_name'].value_counts(normalize=True)

        print(f"\n{name} Dataset:")
        print(f"Label ratio (0/1): {label_ratio:.2f}")
        print("Task distribution:")
        print(task_distribution)

        # 检查比例是否接近原始数据
        assert abs(label_ratio - original_label_ratio) < 0.01, f"{name} label ratio differs significantly"
        for task in original_task_distribution.index:
            assert abs(task_distribution[task] - original_task_distribution[task]) < 0.01, \
                f"{name} task distribution for {task} differs significantly"

    os.makedirs(os.path.join(output_directory, 'trainingset/data'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'val'), exist_ok=True)

    train_data.to_csv(os.path.join(output_directory, 'trainingset', 'data', 'train.csv'), index=False)
    test_data.to_csv(os.path.join(output_directory, 'trainingset', 'data', 'test.csv'), index=False)
    val_data.to_csv(os.path.join(output_directory, 'val', 'val.csv'), index=False)

    print("All files have been processed and saved with balanced distributions.")

def combine_csv_files(input_directory, output_filepath):
    # 定义要合并的文件列表
    file_list = [
        os.path.join(input_directory, 'trainingset', 'data', 'train.csv'),
        os.path.join(input_directory, 'trainingset', 'data', 'test.csv'),
        os.path.join(input_directory, 'val', 'val.csv')
    ]

    # 读取并合并所有文件
    combined_df = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)
    # 保存合并后的数据到新的 CSV 文件
    combined_df.to_csv(output_filepath, index=False)
    print(f"Combined data saved to {output_filepath}")

# 主函数
def main():
    input_directory = '/home/hyt/project/fake_news_detect/dataset/raw/dataset/'
    output_filepath = 'raw.csv'
    combine_csv_files(input_directory, output_filepath)


if __name__ == "__main__":
    # main()
    # print("convert ok")
    main_split()
