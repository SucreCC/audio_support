#!/usr/bin/env python3
"""
文本相似度模型训练系统启动脚本
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='文本相似度模型训练系统')
    parser.add_argument('--model', type=str, choices=['tfidf_lr', 'bert', 'lstm', 'all'], 
                       default=None, help='选择要训练的模型')
    parser.add_argument('--skip', type=str, nargs='+', 
                       choices=['tfidf_lr', 'bert', 'lstm'],
                       help='跳过的模型（仅在model=all时有效）')
    parser.add_argument('--ensemble-only', action='store_true',
                       help='只运行集成，不训练模型')
    parser.add_argument('--test', action='store_true',
                       help='运行快速测试')
    
    args, unknown = parser.parse_known_args()

    # 如果没有输入任何参数，则进入交互式选择
    if len(sys.argv) == 1:
        print("请选择要训练的模型：")
        print("1. tfidf_lr")
        print("2. bert")
        print("3. lstm")
        print("4. all（全部模型）")
        choice = input("请输入数字选择（1/2/3/4）：").strip()
        if choice == '1':
            args.model = 'tfidf_lr'
        elif choice == '2':
            args.model = 'bert'
        elif choice == '3':
            args.model = 'lstm'
        elif choice == '4':
            args.model = 'all'
        else:
            print("无效选择，程序退出。")
            sys.exit(1)
    
    # 检查core目录是否存在
    if not os.path.exists('core'):
        print("错误：找不到core目录，请确保在正确的项目根目录下运行")
        sys.exit(1)
    
    # 切换到core目录
    os.chdir('core')
    
    if args.test:
        print("运行快速测试...")
        subprocess.run([sys.executable, 'quick_test.py'])
    elif args.ensemble_only:
        print("只运行模型集成...")
        subprocess.run([sys.executable, 'run_all_models.py', '--ensemble-only'])
    elif args.model == 'all':
        if args.skip:
            skip_args = ['--skip'] + args.skip
            subprocess.run([sys.executable, 'run_all_models.py'] + skip_args)
        else:
            subprocess.run([sys.executable, 'run_all_models.py'])
    else:
        # 运行单个模型
        model_scripts = {
            'tfidf_lr': 'train/tfidf_lr_train.py',
            'bert': 'train/bert_train.py',
            'lstm': 'train/lstm_train.py'
        }
        
        if args.model in model_scripts:
            script = model_scripts[args.model]
            print(f"运行 {args.model} 模型训练...")
            subprocess.run([sys.executable, script])
        else:
            print(f"不支持的模型: {args.model}")

if __name__ == "__main__":
    main() 