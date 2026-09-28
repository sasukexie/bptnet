"""
配置管理模块
负责管理和验证系统配置，支持YAML配置文件
"""

import yaml
import copy
from pathlib import Path
from typing import Dict, Any
from src.utils.ConfigDict import ConfigDict
from src.utils.logger import logger


class ConfigManager:
    """
    配置管理器
    
    支持YAML配置文件和命令行参数覆盖
    """
    
    def __init__(self, init_config):
        """
        初始化配置管理器
        
        Args:
            model_name: 模型名称（用于自动加载对应配置文件）
            config_path: 配置文件路径（如果指定，则忽略model_name）
            overrides: 覆盖配置的字典（可选）
        """
        self.config = self._load_config(init_config)
        
    def _load_config(self, init_config) -> Dict[str, Any]:
        """
        加载配置（从YAML文件）
        
        Returns:
            dict: 完整配置
        """
        self.config_dir = Path(__file__).parent.parent.parent / 'src/config'
        self.common_config_file = self.config_dir / 'common.yml'
        self.model_config_file = self.config_dir / f'{init_config["model"]["name"].lower()}.yml'

        # 检查文件是否存在
        if not self.common_config_file.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {self.common_config_file}\n"
            )
        
        logger.info(f"加载配置文件: {self.common_config_file}")
        
        # 读取YAML
        with open(self.common_config_file, 'r', encoding='utf-8') as f:
            common_config = yaml.safe_load(f)

        if self.model_config_file.exists():
            with open(self.model_config_file, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
        else:
            model_config = {}

        # 创建自定义Dict Config
        config = ConfigDict()
        config.update(common_config)

        # 汇总配置
        self.tranfer_dict(config, model_config)
        self.tranfer_dict(config, init_config)
        # 处理*结尾参数为最优先
        self.first_priority(config)

        # 处理配置
        self._handling_config(config)

        return config

    
    def _handling_config(self, config: Dict):
        """
        处理配置
        
        Args:
            config: 配置字典
        """
        # 验证设备配置
        if config['train.device'].startswith('cuda'):
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA不可用，切换到CPU")
                config['train.device'] = 'cpu'

        if 'name' not in config['base'] or config['base.name'] is None:
            config['base.name'] = f"model/{config['model.name']}/{config['data.dataset']}/{config['data.split_mode']}_{config['data.frame_type']}"

        # 路径处理
        project_root = Path(__file__).parent.parent.parent
        workspace_root = project_root.parent                       # mer/ 层：共享数据集所在目录
        # 数据目录：解析到 workspace_root/data_dir，使 mer/ 下的多个模型共用同一份数据集，节省磁盘空间
        data_dir = Path(config['data.data_dir'])
        if not data_dir.is_absolute():
            config['data.data_dir'] = str(workspace_root / data_dir)
            logger.debug(f"数据目录: {config['data.data_dir']}")

        # 数据集缓存文件
        cache_file = f"{config['base.base_dir']}/dataset/{config['data.dataset']}/{config['data.split_mode']}_{config['data.frame_type']}.pkl"

        # 标注文件
        if 'annotation_data' in config['data']:
            annotation_data = Path(config['data.annotation_data'])
            if not annotation_data.is_absolute():
                config['data.annotation_data'] = str(workspace_root / data_dir / annotation_data)
                logger.debug(f"标注文件: {config['data.annotation_data']}")

        base_name = config['base.name']
        output_dir = Path(config['base.base_dir']) / base_name # 第一次使用 saved 目录

        # 更新配置
        config['base.output_dir'] = str(output_dir)
        config['base.checkpoint.checkpoint_dir'] = str(output_dir / config['base.checkpoint.checkpoint_dir'])
        config['base.results'] = str(output_dir / 'results')
        config['data.cache_file'] = str(cache_file)
        base_dir = [
            output_dir,
            Path(config['base.checkpoint.checkpoint_dir']),
            Path(config['log.log_dir']),
            Path(config['base.results']),
            Path(cache_file).parent,
        ]

        # 如果启用可视化，创建图片目录
        if config.get('visualization.save_figures', False):
            base_dir.append(output_dir / config['visualization.figure_dir'])

        for dir_path in base_dir:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"实验输出目录: {output_dir}")

    def tranfer_dict(self, config, temp_config):
        # 将 config 中的参数覆盖 temp_config 中参数
        if temp_config:
            for key in temp_config.keys():
                if key in config and type(config[key]) is dict:
                    self.tranfer_dict(config[key], temp_config[key])
                else:
                    config[key] = temp_config[key]
        return config

    def first_priority(self, config):
        # 深度 copy config
        temp_config = copy.deepcopy(config)
        for key in temp_config.keys():
            value = config[key]
            if type(value) is dict:
                self.first_priority(value)
            elif key.__contains__('*'): # 参数里包含*
                config[key.replace('*', '')] = value
                config.pop(key) # 删除key

        return config

    def get_config(self) -> Dict[str, Any]:
        """
        获取配置字典
        
        Returns:
            dict: 配置字典
        """
        return self.config
    
    def print_config(self):
        """打印当前配置"""
        print("\n" + "="*60)
        print("系统配置")
        print("="*60)
        for key, value in sorted(self.config.items()):
            print(f"{key:30s}: {value}")
        print("="*60 + "\n")
    
    def save_config(self, filepath: str = 'config.json'):
        """
        保存配置到文件
        
        Args:
            filepath: 保存路径
        """
        import json
        
        # 转换不可序列化的类型
        serializable_config = {}
        for key, value in self.config.items():
            if isinstance(value, tuple):
                serializable_config[key] = list(value)
            else:
                serializable_config[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已保存到: {filepath}")
