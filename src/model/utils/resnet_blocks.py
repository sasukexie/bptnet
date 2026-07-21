"""
ResNet基础组件实现

包含：
- BasicBlock: 基础残差块
- CABlock: 带通道注意力的残差块（EDMDBN原始实现）
- CBAMBlock: 带CBAM注意力的残差块
- ResNet: ResNet骨干网络

这些组件支持灵活的Block类型，可用于构建各种ResNet变体。
"""
import torch
import torch.nn as nn
from typing import Optional, Callable, List
from torch import Tensor

from .cbam_attention import CBAM


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    """1x1卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=groups)


class BasicBlock(nn.Module):
    """基础残差块"""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1, 
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes, groups=groups)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CABlock(nn.Module):
    """带通道注意力的残差块（原始EDMDBN实现）
    
    核心特性:
    1. 计算全局平均池化和最大池化
    2. 拼接后通过Conv2d(2→1)生成注意力图
    3. 跨层传递注意力作为先验知识
    4. 输出格式: (features, attention_map, if_attn)
    """
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(CABlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes, groups=groups)
        self.bn2 = norm_layer(planes)
        
        # 通道注意力机制（原始EDMDBN实现）
        self.attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 元组 (features, attn_last, if_attn)
                - features: 输入特征 [B, C, H, W]
                - attn_last: 上一层的注意力图（可选）
                - if_attn: 是否应用注意力
        
        Returns:
            tuple: (output_features, attention_map, if_attn)
        """
        x, attn_last, if_attn = x
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        
        # 生成通道注意力
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        attn = torch.cat((avg_out, max_out), dim=1)  # [B, 2, H, W]
        attn = self.attn(attn)  # [B, 1, H, W]
        
        # 跨层注意力传递（先验知识）
        if attn_last is not None:
            attn = attn_last * attn
        
        # 应用注意力到特征
        attn = attn.repeat(1, self.planes, 1, 1)
        if if_attn:
            out = out * attn

        # 返回: (features, attention_map_for_next_layer, if_attn)
        return out, attn[:, 0, :, :].unsqueeze(1), True


class CBAMBlock(nn.Module):
    """带CBAM注意力的残差块"""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, groups: int = 1,
                 base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(CBAMBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes, groups=groups)
        self.bn2 = norm_layer(planes)
        self.attn = CBAM(planes)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x: Tensor) -> Tensor:
        x, attn_last, if_attn = x
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        attn = self.attn(out)
        
        if attn_last is not None:
            attn = attn_last * attn
            
        if if_attn:
            out = out * attn

        return out, None, True


class ResNet(nn.Module):
    """ResNet backbone"""

    def __init__(self, block: nn.Module, layers: List[int], num_classes: int = 3,
                 zero_init_residual: bool = False, groups: int = 4,
                 width_per_group: int = 64, replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
            
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(180, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 128, layers[0], groups=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], groups=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], groups=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], groups=1)

        self.num_features = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, 
                    stride: int = 1, dilate: bool = False, groups: int = 1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        前向传播，兼容不同Block类型
        
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            Tensor: 输出特征
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 判断block类型，使用不同的forward方式
        # CABlock和CBAMBlock接受元组 (x, attn_last, if_attn)
        # BasicBlock接受普通Tensor
        first_block = self.layer1[0]
        
        if hasattr(first_block, 'attn') and not isinstance(first_block, BasicBlock):
            # CABlock或CBAMBlock：需要传递注意力图
            # 注意：BasicBlock也有attn属性但实际不使用，所以要排除
            x, attn1, _ = self.layer1((x, None, True))
            if attn1 is not None:
                attn1 = self.maxpool(attn1)
            
            x, attn2, _ = self.layer2((x, attn1, True))
            if attn2 is not None:
                attn2 = self.maxpool(attn2)
            
            x, attn3, _ = self.layer3((x, attn2, True))
            if attn3 is not None:
                attn3 = self.maxpool(attn3)
            
            x, attn4, _ = self.layer4((x, attn3, True))
        else:
            # BasicBlock：普通Tensor输入
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
