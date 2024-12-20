# 文本二分类项目

## 🚀 项目概述

本项目基于BERT实现了一个二分类模型，旨在分类一个财政文本句子，是否包含我所需的目标指标数值，如果这个句子包含该指标为分类为1，反之为0。

## ✨ 主要特性

- **基于BERT的分类**：利用预训练的BERT模型进行高精度句子分类
- **可定制的模型**：支持冻结BERT层以提高微调效率
- **训练技巧**:
  - 用于处理类别不平衡的Focal Loss
  - 随机过采样
  - 早停机制
- **全面的模型评估**: 
  - 分类报告
  - ROC-AUC得分
  - 混淆矩阵

## 🛠 环境依赖

### 依赖包
- Python 3.7+
- PyTorch 1.4.0
- Transformers
- Scikit-learn
- Pandas
- NumPy
- tqdm
- imbalanced-learn
- TensorBoard

通过以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 📊 项目结构

```
budget-classification/
│
├── src/
│   ├── binary_classify.py      # 主训练脚本
│   ├── BudgetSentence_cls.py   # 推理脚本
│
├── model/                      # 保存的模型权重bert-base-chinese后续可以放在这个文件夹中
├── logging/                    # TensorBoard日志，保存训练过程
├── requirements.txt            # 项目依赖
└── README.md                   # 项目文档
```

## 🔧 配置参数

关键配置参数：
- 预训练模型路径
- 数据集路径
- 分词器设置

### 模型超参数
- **批次大小**：16
- **最大序列长度**：256
- **学习率**：2e-5
- **训练轮次**：5
- **冻结BERT层数**：8

## 🚀 快速开始

### 下载bert-chinese-base模型

```
通过网盘分享的文件：bert-base-chinese
链接: https://pan.baidu.com/s/1rX2QU7stVN4g6WNZNRRx4g?pwd=4eam 提取码: 4eam
```

### 修改binary_classify.py中的参数配置（模型、tokenizer路径和训练数据集路径）

```
Config.pretrained_model_name_or_path
Config.budget_dataset_path
```

### 准备好自己的数据集

```
仿照budget_dataset.xlsx格式准备自己的数据集
```

### 训练（5个epoch、4000条数据集，cpu训练耗时8h左右）

```bash
python binary_classify.py
```

### 推理
```python
from BudgetSentence_cls import batch_sentence_cls

sentences = ["一个句子"]
filtered_sentences = batch_sentence_cls(sentences)
```

## 📈 性能指标

模型典型性能如下：
- **精确率**：~0.98
- **召回率**：~0.85
- **F1分数**：~0.91
- **准确率**：~0.88

## 🧠 模型架构

- **基础模型**：BERT
- **分类头**：带dropout的线性层
- **损失函数**：Focal Loss
- **优化器**：带线性学习率调度的AdamW

## 📝 关键技术

1. **Focal Loss**：通过降低简单样本权重来处理类别不平衡
2. **随机过采样**：平衡少数类和多数类
3. **早停**：防止过拟合
4. **层冻结**：降低计算复杂度
