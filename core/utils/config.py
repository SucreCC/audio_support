import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    import yaml
except ImportError:
    print("请安装PyYAML: pip install PyYAML")
    yaml = None

try:
    import torch
except ImportError:
    print("PyTorch未安装，设备检测功能将受限")
    torch = None


@dataclass
class DataConfig:
    """数据配置"""
    train_file: str
    test_file: str
    columns: dict
    text_separator: str
    fill_na: str


@dataclass
class TrainingConfig:
    """训练配置"""
    n_folds: int
    random_state: int
    shuffle: bool
    results_dir: str


@dataclass
class TfidfLRConfig:
    """TF-IDF + LR 模型配置"""
    ngram_range: List[int]
    n_components: int
    C: float
    n_jobs: int


@dataclass
class BertConfig:
    """BERT 模型配置"""
    model_name: str
    max_len: int
    batch_size: int
    epochs: int
    learning_rate: float
    dropout: float
    num_classes: int


@dataclass
class LSTMConfig:
    """Word2Vec + LSTM 模型配置"""
    # Word2Vec参数
    vector_size: int
    window: int
    min_count: int
    workers: int
    sg: int
    
    # LSTM参数
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    max_len: int
    
    # 训练参数
    batch_size: int
    epochs: int
    learning_rate: float
    
    # 模型结构
    dropout: float
    bidirectional: bool
    num_classes: int


@dataclass
class EnsembleConfig:
    """集成配置"""
    methods: List[str]
    default_weights: dict


@dataclass
class SystemConfig:
    """系统配置"""
    device: str
    show_progress: bool
    
    def get_device(self):
        """获取设备"""
        if torch is None:
            return "cpu"  # 默认返回字符串
            
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')  # Apple Silicon
            else:
                return torch.device('cpu')
        elif self.device == "cpu":
            return torch.device('cpu')
        elif self.device == "cuda":
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("CUDA不可用，使用CPU")
                return torch.device('cpu')
        elif self.device == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("MPS不可用，使用CPU")
                return torch.device('cpu')
        else:
            return torch.device('cpu')


@dataclass
class ModelConfig:
    """模型配置总类"""
    data: DataConfig
    training: TrainingConfig
    tfidf_lr: TfidfLRConfig
    bert: BertConfig
    lstm: LSTMConfig
    ensemble: EnsembleConfig
    system: SystemConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """从YAML文件加载配置"""
        if yaml is None:
            raise RuntimeError("PyYAML未安装，无法加载配置文件")
            
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            # 创建各个配置对象
            data_cfg = DataConfig(**config_dict['data'])
            training_cfg = TrainingConfig(**config_dict['training'])
            tfidf_lr_cfg = TfidfLRConfig(**config_dict['tfidf_lr'])
            bert_cfg = BertConfig(**config_dict['bert'])
            lstm_cfg = LSTMConfig(**config_dict['lstm'])
            ensemble_cfg = EnsembleConfig(**config_dict['ensemble'])
            system_cfg = SystemConfig(**config_dict['system'])

            return cls(
                data=data_cfg,
                training=training_cfg,
                tfidf_lr=tfidf_lr_cfg,
                bert=bert_cfg,
                lstm=lstm_cfg,
                ensemble=ensemble_cfg,
                system=system_cfg
            )
        except FileNotFoundError:
            print(f"未找到配置文件: {yaml_path}")
            return None
        except Exception as e:
            import traceback
            print("加载配置文件失败:")
            print(traceback.format_exc())
            raise e

    def get_model_config(self, model_name: str):
        """获取指定模型的配置"""
        if model_name == "tfidf_lr":
            return self.tfidf_lr
        elif model_name == "bert":
            return self.bert
        elif model_name == "lstm":
            return self.lstm
        else:
            raise ValueError(f"未知的模型名称: {model_name}")

    def print_config(self):
        """打印配置信息"""
        print("✅ 配置文件加载成功")
        print("\n📊 数据配置:")
        print(f"  训练文件: {self.data.train_file}")
        print(f"  测试文件: {self.data.test_file}")
        print(f"  文本分隔符: '{self.data.text_separator}'")
        print(f"  填充值: '{self.data.fill_na}'")
        
        print("\n🎯 训练配置:")
        print(f"  交叉验证折数: {self.training.n_folds}")
        print(f"  随机种子: {self.training.random_state}")
        print(f"  结果目录: {self.training.results_dir}")
        
        print("\n🤖 模型配置:")
        print(f"  TF-IDF+LR: ngram_range={self.tfidf_lr.ngram_range}, n_components={self.tfidf_lr.n_components}")
        print(f"  BERT: model_name={self.bert.model_name}, max_len={self.bert.max_len}, batch_size={self.bert.batch_size}")
        print(f"  LSTM: embedding_dim={self.lstm.embedding_dim}, hidden_dim={self.lstm.hidden_dim}, batch_size={self.lstm.batch_size}")
        
        print("\n🔧 系统配置:")
        device = self.system.get_device()
        print(f"  设备: {device}")
        print(f"  显示进度条: {self.system.show_progress}")
        
        print("\n📈 集成配置:")
        print(f"  可用方法: {self.ensemble.methods}")
        print(f"  默认权重: {self.ensemble.default_weights}")


# 全局配置实例
_config = None

def get_config(config_path: str = None) -> ModelConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        if config_path is None:
            # 默认配置文件路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            config_path = os.path.join(parent_dir, "config.yaml")
        
        _config = ModelConfig.from_yaml(config_path)
        if _config is None:
            raise RuntimeError("配置加载失败")
    
    return _config


if __name__ == '__main__':
    # 测试配置加载
    config = get_config()
    config.print_config()
    
    # 测试设备获取
    device = config.system.get_device()
    print(f"\n🎮 当前设备: {device}")
    
    # 测试模型配置获取
    bert_config = config.get_model_config("bert")
    print(f"\n🤖 BERT配置: {bert_config}") 