import pandas as pd

def count_unique_names_per_task(filepath, task_column='task_name', unique_column='news_id', answer_column='answer'):
    # 读取CSV文件
    df = pd.read_csv(filepath)

    # 获取每个任务中唯一值的数量
    unique_counts = df.groupby(task_column)[unique_column].nunique()

    # 准备结果列表
    task_list = df[task_column].unique()
    results = ["label," + ",".join(task_list)]

    # 打印每个任务中唯一值的数量以及answer列的唯一值及其数量
    for label in df[answer_column].unique():
        row = [f"'{label}'"]
        for task in task_list:
            task_df = df[df[task_column] == task]
            answer_counts = task_df[answer_column].value_counts()
            row.append(str(answer_counts.get(label, 0)))
        results.append(",".join(row))

    # 输出结果
    for line in results:
        print(line)

# 使用函数
filepath = 'raw.csv'
# count_unique_names_per_task(filepath)

# filepath = '/home/hyt/project/fake_news_detect/dataset/raw/dataset/val/val.csv'
# df = pd.read_csv(filepath)
# print(df["answer"])