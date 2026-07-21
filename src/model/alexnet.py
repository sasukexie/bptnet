"""
AlexNet模型 - 经典CNN架构
基于ImageNet预训练，用于微表情识别

参考: Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks, NIPS 2012

支持多模态输入:
- 单模态: RGB图像 (3通道)
- 多模态: RGB + Flow (5通道) 
"""
import torch
import torch.nn as nn
from torchvision import models, transforms


class AlexNet(nn.Module):
    """
    AlexNet模型（迁移学习版本）
    
    Args:
        config (dict): 配置字典
            model:
                num_classes: 分类数量
                pretrained: 是否加载预训练权重
                image_size: 输入图像尺寸（AlexNet需要>=64，推荐224）
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.get('model')
        data_config = config['data']
        
        self.num_classes = data_config.get('num_classes', 3)
        self.pretrained = model_config.get('pretrained', True)
        self.image_size = data_config.get('image_size', 224)
        
        # 使用配置文件中的 frame_type_config
        ft_config = data_config.get('frame_type_config').get(data_config.get('frame_type'))
        input_channels = ft_config['channels']
        self.is_multimodal = ft_config.get('is_multimodal', False)
        
        # 加载预训练AlexNet
        weights = 'DEFAULT' if self.pretrained else None
        base_model = models.alexnet(weights=weights)
        
        # 如果通道数不是3，需要修改第一个卷积层
        if input_channels != 3:
            self._adapt_first_conv(base_model, input_channels)
        
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
    
    def _adapt_first_conv(self, model, input_channels):
        """
        适配第一个卷积层的通道数
        
        Args:
            model: 基础模型
            input_channels: 输入通道数
        """
        old_conv = model.features[0]
        new_conv = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
        
        with torch.no_grad():
            if input_channels > 3:
                # 多模态: 保留原有3通道，新增通道用均值
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, input_channels - 3, 1, 1)
            else:
                # flow only (2通道): 使用前2个RGB通道的均值
                new_conv.weight[:, :input_channels, :, :] = old_conv.weight[:, :input_channels, :, :].mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
            new_conv.bias = old_conv.bias
        
        model.features[0] = new_conv
    
    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
