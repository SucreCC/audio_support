# 增强版文本相似度模型训练脚本

## 概述

这个增强版训练脚本在原始TF-IDF + LR baseline的基础上，新增了两种先进的算法：

1. **BERT微调** - 利用预训练语言模型进行语义建模
2. **Word2Vec + LSTM** - 结合词向量和序列模型进行文本理解

## 算法说明

### 1. TF-IDF + LR (原始baseline)
- **特点**: 快速、轻量级，适合快速验证
- **原理**: 使用TF-IDF特征提取，逻辑回归分类
- **适用场景**: 数据量较大，需要快速baseline

### 2. BERT微调
- **特点**: 强大的语义理解能力，适合小数据集
- **原理**: 基于预训练BERT模型，在目标数据集上微调
- **优势**: 
  - 自动处理中文分词
  - 强大的语义表示能力
  - 适合处理复杂的语义关系

### 3. Word2Vec + LSTM
- **特点**: 中间方案，结合词向量和序列建模
- **原理**: 
  - 使用Word2Vec训练词向量
  - 双向LSTM处理序列信息
  - 适合理解文本的序列特征
- **优势**: 
  - 比BERT更轻量
  - 比TF-IDF有更强的语义理解
  - 适合中等规模数据

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行完整训练（推荐）

```bash
python enhanced_train_and_test.py
```

这将依次训练三个模型并进行集成。

### 2. 单独运行某个模型

如果需要单独测试某个模型，可以修改脚本中的相应部分。

## 输出文件

训练完成后，会在`prediction_result/`目录下生成以下文件：

- `lr_result.csv` - TF-IDF + LR模型预测结果
- `bert_result.csv` - BERT模型预测结果  
- `lstm_result.csv` - Word2Vec + LSTM模型预测结果
- `ensemble_result.csv` - 集成模型预测结果
- `final_result.csv` - 最终提交结果（与原始格式一致）

## 模型性能对比

脚本运行完成后会显示各个模型的性能对比：

```
模型性能总结
==================================================
TF-IDF + LR:     0.xxxx ± 0.xxxx
BERT 微调:       0.xxxx ± 0.xxxx
Word2Vec + LSTM: 0.xxxx ± 0.xxxx
集成模型:        已保存到 prediction_result/final_result.csv
```

## 参数调优建议

### BERT模型
- `max_len`: 文本最大长度（默认128）
- `batch_size`: 批次大小（默认16）
- `epochs`: 训练轮数（默认2）
- `lr`: 学习率（默认2e-5）

### LSTM模型
- `embedding_dim`: 词向量维度（默认100）
- `hidden_dim`: LSTM隐藏层维度（默认128）
- `num_layers`: LSTM层数（默认2）
- `max_len`: 序列最大长度（默认50）
- `batch_size`: 批次大小（默认32）
- `epochs`: 训练轮数（默认5）

### 集成策略
当前使用简单平均集成，可以根据验证集性能调整权重：

```python
# 加权平均集成示例
weights = [0.3, 0.4, 0.3]  # 根据各模型性能调整权重
ensemble_predictions = (weights[0] * lr_predictions[:, 1] + 
                       weights[1] * bert_predictions[:, 1] + 
                       weights[2] * lstm_predictions[:, 1])
```

## 注意事项

1. **内存要求**: BERT模型需要较大内存，建议使用GPU
2. **训练时间**: 完整训练可能需要几小时，BERT模型最耗时
3. **中文处理**: 脚本自动检测并使用中文BERT模型
4. **数据格式**: 确保数据文件路径正确

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 减少max_len
   - 使用CPU训练

2. **BERT模型下载失败**
   - 检查网络连接
   - 手动下载模型到本地

3. **分词错误**
   - 确保jieba库正确安装
   - 检查文本编码格式

## 扩展建议

1. **数据增强**: 添加同义词替换、回译等数据增强技术
2. **模型改进**: 
   - 使用RoBERTa、ALBERT等更先进的预训练模型
   - 尝试BERT-CNN、BERT-LSTM等混合架构
3. **特征工程**: 添加额外的统计特征
4. **集成优化**: 使用Stacking、Blending等高级集成方法 