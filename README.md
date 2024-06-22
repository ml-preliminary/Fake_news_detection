### Fake News Detection LLM-Based

#### Overview

This project leverages LLM for fake news detection. It includes modules for data loading, model initialization, prediction generation, and evaluation. The project utilizes Qwen and Llama models for text analysis and dynamically allocates resources based on available GPUs.

本项目利用大模型进行假新闻检测，以及进行lora微调。项目包含数据加载、模型初始化、生成预测和评估，模型训练等模块。该项目使用Qwen和Llama模型进行文本分析，并根据可用的GPU动态分配资源。

#### Project Structure 项目结构

##### dataset/

- **raw**: Contains raw data and related processing scripts. 包含原始数据和相关处理脚本。
  - **data_analysis**: Directory for data analysis scripts and results. 数据分析脚本和结果目录。
  - **dataset**: Directory for dataset processing scripts and datasets. 数据集处理脚本和数据集目录。
    - `data_analysis.py`: Analyzes datasets, generates plots for question length distribution and label distribution. 分析数据集，生成问题长度分布和标签分布的图表。
    - `data_distribute_do_sample.py`: Processes and samples datasets according to specified distribution ratios. 根据指定的分布比例处理和抽样数据集。
    - `process_all.py`: Script for processing all datasets. 处理所有数据集的脚本。
    - `raw.csv`: Raw dataset file. 原始数据集文件。
    - `whyy.py`: Additional dataset processing script. 额外的数据集处理脚本。

- **trainingset**: Contains training datasets. 包含训练数据集。
  - **data**: Directory for training data files. 训练数据文件目录。
    - `test.csv`: Test dataset file. 测试数据集文件。
    - `train.csv`: Training dataset file. 训练数据集文件。

##### llm_based/

- **vllm_run.py**: Initializes the model, processes input data, generates predictions, and evaluates results. 
  初始化模型，处理输入数据，生成预测并评估结果。

- **utils.py**: Contains utility functions for distance calculation, score conversion, directory management, and creating custom prompts.
  包含距离计算、分数转换、目录管理和创建自定义提示的实用功能。

- **main.sh**: Bash script for setting CUDA devices and running the `vllm_run.py` script with specific parameters.
  用于设置CUDA设备并运行`vllm_run.py`脚本的Bash脚本。

  这部分代码用于运行 `vllm_run.py` 脚本，并传递以下参数：
  - `--model_name='qwen'`：指定使用的模型名称为 `qwen`。
  - `--prompt_mode="single"`：指定提示模式为 `single`。
  - `--output_path="result/tmp"`：指定输出路径为 `result/tmp`。
  - `--max_new_tokens=1`：指定最大新生成的tokens数量为1。
  - `--batch_size=32`：指定批处理大小为32。
  - `--task="gossip,all"`：指定要处理的任务为 `gossip` 和 `all`。
  - `--VLLM_GPU_MEMORY_UTILIZATION=0.75`：指定GPU内存利用率为0.75。

  这个脚本的目的是设置环境变量并调用 `vllm_run.py` 脚本，确保指定的模型、任务和配置正确地运行和处理数据。

- **train.sh**: Script for using SFTTrainer to train models.
  用于使用SFTTrainer训练模型的脚本。包括以下传入参数：
  - `MODEL_NAME_OR_PATH`: 模型的名称或路径，例如 "Qwen/Qwen2-7B-Instruct" 或 "NousResearch/Meta-Llama-3-8B-Instruct"。
  - `DATASET_NAME`: 数据集的路径，例如 "/home/hyt/project/fake_news_detect/dataset/trainingset/"。
  - `OUTPUT_DIR`: 输出目录，例如 "./0622_1522"。
  - `LORA_R`: LoRA 参数，例如 64。
  - `RESPONSE_TEMPLATE`: 响应模板，例如 "assistant"（Llama3: "assistant", Qwen: "assistant"）。
  - `MAX_SEQ_LENGTH`: 最大序列长度，例如 5000。
  - `NUM_TRAIN_EPOCHS`: 训练周期数，例如 1。
  - `SAVE_DIR`: 保存目录，例如 "./qwen-sft"。

- **merge.sh**: Script for merging lora models.
  用于合并lora模型的脚本。