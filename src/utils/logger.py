"""
日志工具模块
提供统一的日志记录功能
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logger(name: str = None, log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，None表示不保存到文件
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    global logger
    if name is not None:
        logger.name = name

    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 设置日志级别
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件Handler（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def create_timestamped_log(log_dir: str = 'logs', prefix: str = 'train') -> str:
    """
    创建带时间戳的日志文件路径（并确保目录存在）
    
    Args:
        log_dir: 日志目录
        prefix: 文件名前缀
        
    Returns:
        str: 日志文件路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    log_file = log_path / f"{prefix}_{timestamp}.log"
    return str(log_file)
