"""
VGG16模型 - 深度CNN架构
基于ImageNet预训练，用于微表情识别

参考: Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, ICLR 2015
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

from .utils.nn_util import adapt_conv2d_channels


class VGG16(nn.Module):
    """
    VGG16模型（迁移学习版本）
    
    Args:
        config (dict): 配置字典
            model:
                num_classes: 分类数量
                pretrained: 是否加载预训练权重
                image_size: 输入图像尺寸（VGG需要>=32，推荐224）
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        data_config = config['data']
        
        self.num_classes = data_config.get('num_classes', 3)
        self.pretrained = model_config.get('pretrained', True)
        self.image_size = data_config.get('image_size', 224)
        
        # 使用配置文件中的 frame_type_config
        ft_config = data_config.get('frame_type_config').get(data_config.get('frame_type'))
        input_channels = ft_config['channels']
        self.is_multimodal = ft_config.get('is_multimodal', False)
        
        # 加载预训练VGG16
        weights = 'DEFAULT' if self.pretrained else None
        base_model = models.vgg16(weights=weights)
        
        # 如果通道数不是3，需要修改第一个卷积层
        if input_channels != 3:
            base_model.features[0] = adapt_conv2d_channels(base_model.features[0], input_channels)
        
        self.model = base_model
        
        # 修改分类头，适配微表情分类
        self.model.classifier[6] = nn.Linear(4096, self.num_classes)
        
        # 添加resize层，确保输入尺寸正确
        self.resize = transforms.Resize((self.image_size, self.image_size))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据
               - 单模态: [B, 3, H, W] (RGB图像)
               - 多模态: tuple/list ([B, 3, H, W], [B, 2, H, W]) (RGB + Flow)
        
        Returns:
            logits: 分类输出 [B, num_classes]
        """
        # 处理多模态输入
        if isinstance(x, (list, tuple)) and self.is_multimodal:
            rgb, flow = x
            # 拼接通道: [B, 5, H, W]
            x = torch.cat([rgb, flow], dim=1)
        
        # 如果输入尺寸不匹配，先resize
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = self.resize(x)
        return self.model(x)
    
    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
