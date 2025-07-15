#!/usr/bin/env python3
"""
自动安装依赖包脚本
根据用户环境自动选择合适的PyTorch版本
"""

import os
import sys
import subprocess
import platform

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"✓ 检测到CUDA {cuda_version}")
            return True, cuda_version
        else:
            print("✗ 未检测到CUDA")
            return False, None
    except ImportError:
        print("✗ PyTorch未安装，无法检测CUDA")
        return False, None

def check_system():
    """检查系统信息"""
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"架构: {platform.machine()}")

def install_requirements(requirements_file):
    """安装指定的requirements文件"""
    print(f"\n正在安装 {requirements_file}...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {requirements_file} 安装成功")
            return True
        else:
            print(f"✗ {requirements_file} 安装失败")
            print("错误信息:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ 安装过程出错: {e}")
        return False

def main():
    print("="*60)
    print("文本相似度模型训练系统 - 依赖安装向导")
    print("="*60)
    
    # 检查系统信息
    check_system()
    
    # 检查CUDA
    cuda_available, cuda_version = check_cuda()
    
    print("\n" + "="*60)
    print("请选择安装方式:")
    print("1. CPU版本 (推荐用于快速测试)")
    print("2. GPU版本 (推荐用于完整训练)")
    print("3. 自动检测 (根据环境自动选择)")
    print("4. 退出")
    print("="*60)
    
    while True:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            print("\n选择CPU版本安装...")
            success = install_requirements("requirements_cpu.txt")
            break
        elif choice == "2":
            if not cuda_available:
                print("\n⚠️  警告: 未检测到CUDA，但您选择了GPU版本")
                confirm = input("是否继续安装GPU版本? (y/N): ").strip().lower()
                if confirm != 'y':
                    continue
            
            print("\n选择GPU版本安装...")
            success = install_requirements("requirements_gpu.txt")
            break
        elif choice == "3":
            print("\n自动检测环境...")
            if cuda_available:
                print("检测到CUDA，安装GPU版本")
                success = install_requirements("requirements_gpu.txt")
            else:
                print("未检测到CUDA，安装CPU版本")
                success = install_requirements("requirements_cpu.txt")
            break
        elif choice == "4":
            print("退出安装")
            return
        else:
            print("无效选择，请输入1-4")
    
    if success:
        print("\n" + "="*60)
        print("✓ 依赖安装完成！")
        print("="*60)
        print("\n接下来您可以:")
        print("1. 运行快速测试: cd core && python quick_test.py")
        print("2. 开始训练: cd core && python run_all_models.py")
        print("3. 查看使用说明: cat 使用说明.md")
    else:
        print("\n" + "="*60)
        print("✗ 依赖安装失败")
        print("="*60)
        print("\n请尝试:")
        print("1. 检查网络连接")
        print("2. 更新pip: pip install --upgrade pip")
        print("3. 手动安装: pip install -r requirements_cpu.txt")

if __name__ == "__main__":
    main() 