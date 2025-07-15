#!/usr/bin/env python3
"""
TF-IDF + LR 模型训练脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models.tfidf_lr_model import TfidfLRModel

def main():
    """主函数"""
    print("开始训练TF-IDF + LR模型...")
    
    # 创建模型实例
    model = TfidfLRModel(
        ngram_range=(1, 5),
        n_components=100,
        C=1,
        n_jobs=20
    )
    
    # 训练模型
    auc, predictions = model.train()
    
    # 保存结果
    model.save_results(predictions)
    
    print(f"\nTF-IDF + LR 模型训练完成！")
    print(f"平均 AUC: {auc:.4f}")

if __name__ == "__main__":
    main() 