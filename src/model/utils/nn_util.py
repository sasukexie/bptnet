"""
模型工具函数 - 提供通用的模型适配功能
"""
import torch
import torch.nn as nn


def adapt_conv2d_channels(conv_layer, input_channels):
    """
    适配Conv2d层的输入通道数
    
    Args:
        conv_layer: 原始卷积层
        input_channels: 新的输入通道数
    
    Returns:
        新的卷积层
    """
    out_channels = conv_layer.out_channels
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    bias = conv_layer.bias is not None
    
    new_conv = nn.Conv2d(
        input_channels, out_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding,
        bias=bias
    )
    
    with torch.no_grad():
        if input_channels >= 3:
            # 保留原有3通道权重
            new_conv.weight[:, :3, :, :] = conv_layer.weight
            # 新增通道使用均值初始化
            if input_channels > 3:
                new_conv.weight[:, 3:, :, :] = conv_layer.weight.mean(dim=1, keepdim=True).repeat(1, input_channels - 3, 1, 1)
        else:
            # 通道数少于3（如flow的2通道），使用前N个通道的均值
            new_conv.weight[:, :input_channels, :, :] = conv_layer.weight[:, :input_channels, :, :].mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
        
        if bias:
            new_conv.bias = conv_layer.bias
    
    return new_conv


