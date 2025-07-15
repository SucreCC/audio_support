#!/usr/bin/env python3
"""
BERT 模型训练脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.bert_model import BertModelWrapper

def main():
    """主函数"""
    print("开始训练BERT模型...")
    
    # 创建模型实例
    model = BertModelWrapper(
        bert_model_name='bert-base-chinese',
        max_len=128,
        batch_size=16,
        epochs=2,
        lr=2e-5
    )
    
    # 训练模型
    auc, predictions = model.train()
    
    # 保存结果
    model.save_results(predictions)
    
    print(f"\nBERT 模型训练完成！")
    print(f"平均 AUC: {auc:.4f}")

if __name__ == "__main__":
    main() 