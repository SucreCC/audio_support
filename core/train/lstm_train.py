#!/usr/bin/env python3
"""
Word2Vec + LSTM 模型训练脚本
"""

import sys
import os

from core.models import LstmModelWrapper

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    """主函数"""
    print("开始训练Word2Vec + LSTM模型...")
    
    # 创建模型实例
    model = LstmModelWrapper(
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        max_len=50,
        batch_size=32,
        epochs=5,
        lr=0.001,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1
    )
    
    # 训练模型
    auc, predictions = model.train()
    
    # 保存结果
    model.save_results(predictions)
    
    print(f"\nWord2Vec + LSTM 模型训练完成！")
    print(f"平均 AUC: {auc:.4f}")

if __name__ == "__main__":
    main() 