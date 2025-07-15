#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.utils import config

def test_config():
    """测试配置加载"""
    try:
        # 强制重新加载配置
        configer = config.get_config(force_reload=True)
        
        print("配置加载成功!")
        print(f"BERT学习率: {configer.bert.learning_rate}")
        print(f"BERT学习率类型: {type(configer.bert.learning_rate)}")
        print(f"LSTM学习率: {configer.lstm.learning_rate}")
        print(f"LSTM学习率类型: {type(configer.lstm.learning_rate)}")
        
        # 测试是否可以用于数学运算
        lr = configer.bert.learning_rate
        print(f"学习率 * 2: {lr * 2}")
        print(f"学习率 > 0: {lr > 0}")
        
        return True
    except Exception as e:
        print(f"配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config() 