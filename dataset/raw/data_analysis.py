import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def load_datasets(directory, filename):
    filepath = os.path.join(directory, filename)
    df = pd.read_csv(filepath)
    print(f"Loaded {filename}")
    return df

def plot_question_length_distribution(df, output_directory, prefix):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['task_name'].unique())))
    for color, task_name in zip(colors, df['task_name'].unique()):
        task_df = df[df['task_name'] == task_name].copy()
        task_df['length'] = task_df['question'].apply(len)
        density = gaussian_kde(task_df['length'])
        x = np.linspace(task_df['length'].min(), task_df['length'].max(), 100)
        plt.hist(task_df['length'], bins=30, alpha=0.3, label=task_name, density=True, color=color)
        plt.plot(x, density(x), label=f"{task_name} (curve)", color=color)
        plt.fill_between(x, 0, density(x), alpha=0.1, color=color)
    plt.title(f'Question Length Distribution for {prefix}')
    plt.xlabel('Length')
    plt.ylabel('Density')
    plt.legend()
    output_path = os.path.join(output_directory, f'{prefix}_question_length.png')
    plt.savefig(output_path)
    plt.close()

def plot_label_distribution_by_task(df, output_directory, prefix):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['task_name'].unique())))
    bar_width = 0.1  # Adjusted width for better separation
    tasks = df['task_name'].unique()
    unique_answers = df['answer'].apply(lambda x: int(x.strip('<answer></answer>'))).unique()
    indices = np.arange(len(unique_answers) * len(tasks))  # Ensure sufficient length for indices

    for i, (color, task_name) in enumerate(zip(colors, tasks)):
        task_df = df[df['task_name'] == task_name].copy()
        task_df['answer'] = task_df['answer'].apply(lambda x: int(x.strip('<answer></answer>')))
        counts = task_df['answer'].value_counts().reindex(unique_answers, fill_value=0).sort_index()
        bar_positions = indices[i * len(unique_answers):(i + 1) * len(unique_answers)]  # Correct bar positions
        plt.bar(bar_positions, counts, bar_width, alpha=0.7, label=task_name, color=color)

    plt.title(f'Label Distribution for {prefix}')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(indices[::len(tasks)], unique_answers)  # Set x-ticks to display labels correctly
    plt.legend()
    output_path = os.path.join(output_directory, f'{prefix}_label_distribution.png')
    plt.savefig(output_path)
    plt.close()


def analyze_data(df, output_directory, prefix):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Question length distribution
    plot_question_length_distribution(df, output_directory, prefix)

    # Label distribution
    plot_label_distribution_by_task(df, output_directory, prefix)


def main():
    output_directory = 'data_analysis'  # Output directory for the analysis results

    # Process train.csv
    directory = '/home/hyt/project/fake_news_detect/dataset/raw/dataset/trainingset/data'
    train_df = load_datasets(directory, 'train.csv')
    analyze_data(train_df, output_directory, 'train')

    # Process val.csv
    directory = '/home/hyt/project/fake_news_detect/dataset/raw/dataset/val'
    val_df = load_datasets(directory, 'val.csv')
    analyze_data(val_df, output_directory, 'val')

    # Process test.csv
    directory = '/home/hyt/project/fake_news_detect/dataset/raw/dataset/trainingset/data'
    test_df = load_datasets(directory, 'test.csv')
    analyze_data(test_df, output_directory, 'test')

    print("Data analysis completed and plots saved.")

if __name__ == "__main__":
    main()
