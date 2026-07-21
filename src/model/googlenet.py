"""
GoogLeNet模型 - Inception架构
基于ImageNet预训练，用于微表情识别

参考: Szegedy et al., Going Deeper with Convolutions, CVPR 2015
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

from .utils.nn_util import adapt_conv2d_channels


class GoogLeNet(nn.Module):
    """
    GoogLeNet模型（迁移学习版本）
    
    Args:
        config (dict): 配置字典
            model:
                num_classes: 分类数量
                pretrained: 是否加载预训练权重
                aux_logits: 是否使用辅助分类器（训练时有用）
                image_size: 输入图像尺寸（GoogLeNet需要>=224）
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        data_config = config['data']
        
        self.num_classes = data_config.get('num_classes', 3)
        self.pretrained = model_config.get('pretrained', True)
        self.aux_logits = model_config.get('aux_logits', False)
        self.image_size = data_config.get('image_size', 224)
        
        # 使用配置文件中的 frame_type_config
        ft_config = data_config.get('frame_type_config').get(data_config.get('frame_type'))
        input_channels = ft_config['channels']
        self.is_multimodal = ft_config.get('is_multimodal', False)
        
        # 加载预训练GoogLeNet
        weights = 'DEFAULT' if self.pretrained else None
        base_model = models.googlenet(
            weights=weights,
            aux_logits=self.aux_logits,
            transform_input=False
        )
        
        # 如果通道数不是3，需要修改第一个卷积层
        if input_channels != 3:
            base_model.conv1.conv = adapt_conv2d_channels(base_model.conv1.conv, input_channels)
        
        self.model = base_model
        
        # 修改主分类头，适配微表情分类
        self.model.fc = nn.Linear(1024, self.num_classes)
        
        # 如果使用辅助分类器，也修改
        if self.aux_logits:
            self.model.aux1.fc2 = nn.Linear(1024, self.num_classes)
            self.model.aux2.fc2 = nn.Linear(1024, self.num_classes)
        
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
            (aux1, aux2): 辅助分类器输出（训练时）
        """
        # 处理多模态输入
        if isinstance(x, (list, tuple)) and self.is_multimodal:
            rgb, flow = x
            # 拼接通道: [B, 5, H, W]
            x = torch.cat([rgb, flow], dim=1)
        
        # 如果输入尺寸不匹配，先resize
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = self.resize(x)
        
        if self.training and self.aux_logits:
            # 训练模式：返回主输出和辅助输出
            outputs = self.model(x)
            # outputs = (logits, aux_logits1, aux_logits2)
            return outputs[0]  # 只返回主分类输出
        else:
            # 测试模式：只返回主输出
            return self.model(x)
    
    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
