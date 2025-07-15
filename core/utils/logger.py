import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path

class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 创建logs目录
        log_dir = Path(__file__).resolve().parent.parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)

        # 创建文件处理器，使用按天轮转
        file_handler = TimedRotatingFileHandler(
            filename=log_dir / f"{name}.log",
            when="midnight",
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str, exc_info=True):
        self.logger.error(message, exc_info=exc_info)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
