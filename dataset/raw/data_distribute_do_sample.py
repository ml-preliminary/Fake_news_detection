import pandas as pd
import numpy as np

# 读取val/val.csv
val_df = pd.read_csv('dataset/val/val.csv')


# 打印每个任务的分布信息
task_distribution_info = val_df['task_name'].value_counts(normalize=True)
print(task_distribution_info)

val_df['answer'] = val_df['answer'].apply(lambda x: x.strip('</answer>'))

# 定义任务分布
task_distribution = {
    'style_based_fake': 0.207397,
    'story_based_fake': 0.203334,
    'style_based_legitimate': 0.157521,
    'content_based_fake': 0.157474,
    'integration_based_legitimate': 0.117219,
    'gossipcop_v3_origin': 0.103722,
    'integration_based_fake': 0.053332
}

num_samples_per_task = 500

# 创建两个空的DataFrame用于存储结果
proportional_distribute_df = pd.DataFrame()
fixed_distribute_df = pd.DataFrame()

for task, proportion in task_distribution.items():
    task_df = val_df[val_df['task_name'] == task]

    # 计算所需的样本数量
    num_samples_label1 = int(num_samples_per_task * proportion)
    num_samples_label0 = num_samples_per_task - num_samples_label1
    # 按照任务分布比例采样
    label1_df = task_df[task_df['answer'] == "1"]
    label0_df = task_df[task_df['answer'] == "0"]
    print(task_df.describe(include='all'))
    sampled_label1_df = label1_df.sample(n=num_samples_label1, random_state=42) if len(label1_df) >= num_samples_label1 else label1_df
    sampled_label0_df = label0_df.sample(n=num_samples_label0, random_state=42) if len(label0_df) >= num_samples_label0 else label0_df
    print(len(sampled_label1_df), len(sampled_label0_df))

    proportional_task_df = pd.concat([sampled_label1_df, sampled_label0_df])
    proportional_distribute_df = pd.concat([proportional_distribute_df, proportional_task_df])

    # 固定比例采样（250正250负）
    sampled_label1_fixed_df = label1_df.sample(n=250, random_state=42) if len(label1_df) >= 250 else label1_df
    sampled_label0_fixed_df = label0_df.sample(n=250, random_state=42) if len(label0_df) >= 250 else label0_df

    fixed_task_df = pd.concat([sampled_label1_fixed_df, sampled_label0_fixed_df])
    fixed_distribute_df = pd.concat([fixed_distribute_df, fixed_task_df])

# 检查并确保每个任务的采样结果数量正确
if len(proportional_distribute_df) > num_samples_per_task * len(task_distribution):
    proportional_distribute_df = proportional_distribute_df.sample(n=num_samples_per_task * len(task_distribution), random_state=42)
if len(fixed_distribute_df) > num_samples_per_task * len(task_distribution):
    fixed_distribute_df = fixed_distribute_df.sample(n=num_samples_per_task * len(task_distribution), random_state=42)

# 打乱数据顺序
proportional_distribute_df = proportional_distribute_df.sample(frac=1, random_state=42).reset_index(drop=True)
fixed_distribute_df = fixed_distribute_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存结果
proportional_distribute_df.to_csv('dataset/val/raw_distribute.csv', index=False)
fixed_distribute_df.to_csv('dataset/val/same_distribute.csv', index=False)

print("Proportional sampling result:")
print(proportional_distribute_df)
print("\nFixed sampling result:")
print(fixed_distribute_df)
