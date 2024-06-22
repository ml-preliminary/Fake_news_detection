import os
import pandas as pd

# 定义目录路径
directory = 'tmp/'

# 要保留的列
columns_to_keep = ['label', 'model_output']

# 遍历目录中的所有CSV文件
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # 只保留指定的列
        df = df[columns_to_keep]
        # 覆盖写回CSV文件
        df.to_csv(file_path, index=False)
        print(f"Processed and saved {filename}")

print("All files have been processed.")
