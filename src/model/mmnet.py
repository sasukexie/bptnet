"""
MMNet (Multi-Modal Network)
基于EDMDBN的多模态网络，使用通道注意力机制

该模型通过计算onset和apex帧的差异来捕捉微表情的动态变化
"""
import torch.nn as nn
from torchvision.transforms import Resize

from .utils.resnet_blocks import ResNet, CABlock


class MMNet(nn.Module):
    """
    多模态微表情识别网络（完整复现版）
    
    核心特性:
    1. Onset-Apex差值特征提取
    2. 带通道注意力(CABlock)的ResNet主干
    3. 跨层注意力传递机制
    4. 可选的ViT-POS位置校准模块
    
    Args:
        config (dict): 配置字典
            - num_classes: 分类数量
            - use_vit_pos: 是否使用ViT-POS位置校准
            - image_size: 输入图像尺寸
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        data_config = config['data']
        
        # 从配置中读取参数
        self.num_classes = data_config.get('num_classes', 3)
        self.use_vit_pos = model_config.get('use_vit_pos', False)
        self.image_size = data_config.get('image_size', 28)
        
        # 读取 frame_type_config，获取帧输入模式和通道数
        ft_config = data_config.get('frame_type_config', {}).get(
            data_config.get('frame_type', 'apex'), {}
        )
        self.frame_mode = ft_config.get('mode', 'single')  # 'single' / 'stacked' / 'multimodal'
        self.input_channels = ft_config.get('channels', 3)
        
        # 特征提取卷积层（原始EDMDBN配置）
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=180, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),
        )
        
        # 可选的位置校准模块（ViT-POS）
        if self.use_vit_pos:
            from .utils.vit_pos_embed import VisionTransformer_POS
            # 根据输入尺寸调整patch_size
            patch_size = 7 if self.image_size >= 28 else 4
            self.vit_pos = VisionTransformer_POS(
                img_size=self.image_size,
                patch_size=patch_size,
                embed_dim=512,
                depth=2,
                num_heads=4,
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.
            )
            # 计算resize后的尺寸
            feature_map_size = self.image_size // 2  # conv_act stride=2
            patch_num = feature_map_size // patch_size
            self.resize = Resize([patch_num, patch_num])
        
        # 使用带通道注意力的ResNet作为主干网络
        # CABlock实现了真正的通道注意力机制
        self.main_branch = ResNet(block=CABlock, layers=[1, 1, 1, 1], num_classes=self.num_classes)
        num_features = self.main_branch.num_features
        
        # 添加自适应池化，确保输出固定尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头（增加Dropout防止过拟合）
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, self.num_classes),
        )
        
    def forward_features(self, x):
        """
        提取特征
        
        对于单帧输入，直接使用；对于双帧/堆叠帧输入，计算onset→apex差异
        
        Args:
            x: 输入数据
                - 单帧: [B, C, H, W]
                - 双帧: (onset, apex) 或 [onset, apex]
                - 堆叠帧(stacked): [B, 9, H, W] (onset+apex+offset 通道堆叠)
        
        Returns:
            Tensor: 特征图 [B, num_features, H', W']
        """
        # 确定 onset_frame (用于 ViT-POS) 和 act (onset→apex 差异)
        if isinstance(x, (list, tuple)):
            onset_frame, apex = x
            act = apex - onset_frame
        elif self.frame_mode == 'stacked' and self.input_channels > 3:
            # 堆叠模式 (e.g., rgb_triplet: onset+apex+offset = 9 channels)
            # 取前3通道为onset，中间3通道为apex，计算差异
            onset_frame = x[:, :3, :, :]
            apex = x[:, 3:6, :, :]
            act = apex - onset_frame
        else:
            # 单帧输入，直接使用
            onset_frame = x
            act = x
        
        # 特征提取
        act = self.conv_act(act)  # [B, 180, H/2, W/2]
        
        # 可选的位置嵌入（ViT-POS）
        pos_embed = None
        if self.use_vit_pos:
            B = act.shape[0]
            # ViT-POS从onset帧提取位置信息
            pos_embed = self.vit_pos(self.resize(onset_frame))
            # 转换格式: [B, N, C] -> [B, C, H, W]
            patch_num = pos_embed.shape[1]
            pos_embed = pos_embed.transpose(1, 2).view(B, 512, patch_num, patch_num)
        
        # 主干网络（CABlock ResNet）
        # 注意：ResNet内部会处理CABlock的元组输入输出
        out = self.main_branch(act)
        
        # 可选：融合位置嵌入
        if pos_embed is not None:
            # 需要确保尺寸匹配
            if out.shape[-2:] != pos_embed.shape[-2:]:
                pos_embed = nn.functional.interpolate(
                    pos_embed, size=out.shape[-2:], mode='bilinear', align_corners=False
                )
            out = out + pos_embed
        
        return out
    
    def forward(self, x):
        """前向传播"""
        features = self.forward_features(x)
        # 自适应池化到1x1
        features = self.adaptive_pool(features)
        features = features.flatten(start_dim=1)
        output = self.head(features)
        return output
    
    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
