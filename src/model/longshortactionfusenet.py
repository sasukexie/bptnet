"""
Long-Short Action Fusion Network (LSAFN)
基于EDMDBN的长短时动作融合网络

核心特性:
1. 双分支架构：长时动作（apex-onset差值）+ 短时动作（帧序列）
2. 可选backbone：Swin Transformer 2D 或 CNN
3. ViT-POS位置校准模块（可选）
4. 灵活的特征融合策略（concat/add）
5. 多种池化方式（Flatten/GAP/GMP）

参考文献:
    EDMDBN原始实现: baseline/EDMDBN/models/long_short_fusenet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from .utils.nn_util import adapt_conv2d_channels


class LongShortActionFuseNet(nn.Module):
    """
    长短时动作融合网络
    
    核心特性:
    1. 双分支架构：长时动作（apex-onset差值）+ 短时动作（帧序列）
    2. 可选backbone：Swin Transformer 2D 或 CNN
    3. ViT-POS位置校准模块（可选）
    4. 灵活的特征融合策略（concat/add）
    5. 多种池化方式（Flatten/GAP/GMP）
    
    Args:
        config (dict): 配置字典
            model:
                num_classes: 分类数量
                image_size: 输入图像尺寸
                backbone_type: backbone类型 ('swin_2d', 'cnn')
                embed_dim: 嵌入维度
                use_long_action: 是否使用长时分支
                use_short_action: 是否使用短时分支
                use_pos_vit: 是否使用ViT-POS
                use_feature_concat: 是否使用concat融合
                use_feature_conv: concat后是否使用卷积
                pooling_type: 池化类型 ('flatten', 'gap', 'gmp')
                head_dropout: 分类头dropout
    
    Input Formats:
        1. 单帧: [B, C, H, W]
        2. 双帧: (onset, apex) 或 [onset, apex]
        3. 字典: {'onset': tensor, 'apex': tensor, 'frames': tensor}
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        data_config = config['data']

        # 基础配置
        self.num_classes = data_config.get('num_classes', 3)
        self.image_size = data_config.get('image_size', 28)
        self.backbone_type = model_config.get('backbone_type', 'cnn')  # 'swin_2d' or 'cnn'
        
        # 分支控制
        self.use_long_action = model_config.get('use_long_action', True)
        self.use_short_action = model_config.get('use_short_action', False)
        assert self.use_long_action or self.use_short_action, \
            "use_long_action and use_short_action can't be both False"
        
        # ViT-POS配置
        self.use_pos_vit = model_config.get('use_pos_vit', False)
        self.posvit_patch_size = model_config.get('posvit_patch_size', 7)
        
        # 特征融合配置
        self.concat = model_config.get('use_feature_concat', False)
        self.use_feature_conv = model_config.get('use_feature_conv', False)
        
        # 池化配置
        self.pooling_type = model_config.get('pooling_type', 'flatten')  # 'flatten', 'gap', 'gmp'
        self.head_dropout = model_config.get('head_dropout', 0.5)
        
        # Backbone配置
        self.embed_dim = model_config.get('embed_dim', 128)
        self.swin_depths = model_config.get('depths', [2, 2, 6, 2])
        self.swin_num_heads = model_config.get('num_heads', [4, 8, 16, 32])
        self.swin_window_size = model_config.get('window_size', 7)
        self.drop_path_rate = model_config.get('drop_path_rate', 0.2)
        
        # ========== 构建模型组件 ==========
        
        # 1. ViT-POS位置校准模块（可选）
        if self.use_pos_vit:
            from .utils.vit_pos_embed import VisionTransformer_POS
            # 计算resize目标尺寸
            feature_map_size = self.image_size // 2  # 假设backbone第一层stride=2
            target_size = max(14, feature_map_size)  # 至少14x14
            self.vit_pos = VisionTransformer_POS(
                img_size=target_size,
                patch_size=self.posvit_patch_size,
                embed_dim=512,
                depth=2,
                num_heads=4,
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.
            )
            self.resize = Resize([target_size, target_size])
            posvit_output_size = target_size // self.posvit_patch_size
            self.posvit_feature_dim = 512
            self.posvit_spatial_size = posvit_output_size
        
        # 2. 构建backbone
        if self.backbone_type == 'swin_2d':
            self._build_swin_backbone()
        else:  # 'cnn'
            self._build_cnn_backbone()
        
        # 3. 获取backbone输出特征维度
        num_features = self._get_backbone_output_channels()
        
        # 4. 特征融合层
        if self.use_long_action and self.use_short_action:
            if self.concat:
                concat_features = num_features * 2
                if self.use_feature_conv:
                    self.feature_conv = nn.Sequential(
                        nn.Conv2d(concat_features, concat_features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(concat_features),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2)
                    )
                    num_features = concat_features
                else:
                    num_features = concat_features
            else:
                # element-wise add，特征维度不变
                pass
        
        # 5. 池化层和分类头
        self._build_pooling_and_head(num_features)
        
        # 初始化权重
        self._init_weights()
    
    def _build_swin_backbone(self):
        """构建Swin Transformer 2D backbone"""
        try:
            from timm.models import create_model
            
            # 检查输入尺寸是否适合Swin
            # Swin要求: img_size >= patch_size * window_size
            # 默认patch_size=4, window_size=7，所以最小需要28x28
            if self.image_size < 28:
                raise ValueError(
                    f"Image size {self.image_size}x{self.image_size} is too small for Swin Transformer. "
                    f"Minimum required: 28x28 (patch_size=4 * window_size=7). "
                    f"Consider using CNN backbone or resizing images."
                )
            
            # 获取输入通道数
            input_channels = self._get_input_channels()
            
            # 根据embed_dim选择模型变体
            if self.embed_dim <= 96:
                model_name = 'swin_tiny_patch4_window7_224'
            elif self.embed_dim <= 128:
                model_name = 'swin_small_patch4_window7_224'
            else:
                model_name = 'swin_base_patch4_window7_224'
            
            # 计算合适的window_size（必须能被输入尺寸整除）
            # Swin要求: img_size % (patch_size * window_size) == 0
            # 默认patch_size=4，所以对于28x28: window_size最大为7
            # 但为了安全，使用较小的window_size
            if self.image_size < 56:
                # 小尺寸输入使用较小的window
                swin_window_size = min(7, self.image_size // 4)
            else:
                swin_window_size = self.swin_window_size
            
            # 确保window_size至少为1
            swin_window_size = max(1, swin_window_size)
            
            # 验证配置有效性
            patch_size = 4  # Swin默认patch_size
            if self.image_size % (patch_size * swin_window_size) != 0:
                print(f"[WARNING] Image size {self.image_size} is not divisible by "
                      f"patch_size({patch_size}) * window_size({swin_window_size}) = {patch_size * swin_window_size}")
                print(f"[WARNING] Adjusting window_size to ensure compatibility...")
                # 调整window_size使其能整除
                for ws in range(swin_window_size, 0, -1):
                    if self.image_size % (patch_size * ws) == 0:
                        swin_window_size = ws
                        break
            
            print(f"[LSAFN] Swin config: img_size={self.image_size}, "
                  f"window_size={swin_window_size}, embed_dim={self.embed_dim}, "
                  f"model={model_name}, in_chans={input_channels}")
            
            # 创建长时动作分支
            if self.use_long_action:
                self.long_action_model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0,  # 移除分类头
                    img_size=self.image_size,
                    in_chans=input_channels,
                    window_size=swin_window_size,  # 使用计算后的window_size
                    drop_path_rate=self.drop_path_rate,
                )
                # 记录num_features
                self.long_num_features = self.long_action_model.num_features
            
            # 创建短时动作分支
            if self.use_short_action:
                self.short_action_model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0,
                    img_size=self.image_size,
                    in_chans=input_channels,
                    window_size=swin_window_size,  # 使用计算后的window_size
                    drop_path_rate=self.drop_path_rate,
                )
                self.short_num_features = self.short_action_model.num_features
                
        except ImportError:
            raise ImportError(
                "timm is required for Swin Transformer. "
                "Please install it: pip install timm>=0.6.0"
            )
    
    def _get_input_channels(self):
        """从配置中获取输入通道数"""
        data_config = self.config.get('data', {})
        frame_type = data_config.get('frame_type', 'apex')
        ft_config = data_config.get('frame_type_config', {})
        
        if ft_config and frame_type in ft_config:
            channels = ft_config[frame_type].get('channels', 3)
        else:
            # 回退: 尝试从 frame_type_config 路径获取
            channels = data_config.get('frame_type_config', {}).get(frame_type, {}).get('channels', 3)
        
        print(f"[LSAFN] Input channels: {channels} (frame_type={frame_type})")
        return channels
    
    def _build_cnn_backbone(self):
        """构建CNN backbone（简化版，用于资源受限场景）"""
        input_channels = self._get_input_channels()
        if self.use_long_action:
            self.long_action_model = self._create_cnn_module(input_channels)
            self.long_num_features = self.embed_dim * 4
        
        if self.use_short_action:
            self.short_action_model = self._create_cnn_module(input_channels)
            self.short_num_features = self.embed_dim * 4
    
    def _create_cnn_module(self, input_channels=3):
        """创建CNN模块"""
        cnn_module = nn.Sequential(
            # Block 1: HxH -> H/2 x H/2
            nn.Conv2d(3, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: H/2 x H/2 -> H/4 x H/4
            nn.Conv2d(self.embed_dim, self.embed_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: H/4 x H/4 -> fixed size
            nn.Conv2d(self.embed_dim * 2, self.embed_dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim * 4),
            nn.ReLU(inplace=True),
        )
        
        # 如果输入通道数不是默认的3，适配第一个卷积层
        if input_channels != 3:
            cnn_module[0] = adapt_conv2d_channels(cnn_module[0], input_channels)
        
        return cnn_module
    
    def _get_backbone_output_channels(self):
        """获取backbone输出通道数"""
        if self.use_long_action:
            return self.long_num_features
        elif self.use_short_action:
            return self.short_num_features
        else:
            raise ValueError("No action branch enabled")
    
    def _build_pooling_and_head(self, num_features):
        """构建池化层和分类头"""
        # 根据backbone类型和输入尺寸计算spatial size
        if self.backbone_type == 'swin_2d':
            # Swin Transformer的输出尺寸计算
            # Swin使用patch_size=4，然后经过4个stage，每个stage缩小2倍
            # 总缩小倍数: 4 * 2^4 = 64（但实际取决于具体实现）
            # 对于timm的Swin，输出通常是 input_size / 32
            # 例如: 224/32=7, 28/32≈1 (向下取整)
            # 更准确的方式是根据实际测试确定
            if self.image_size >= 224:
                spatial_size = 7
            elif self.image_size >= 112:
                spatial_size = 4
            elif self.image_size >= 56:
                spatial_size = 2
            else:
                # 小尺寸输入，可能是1x1或需要adaptive pooling
                spatial_size = 1
        else:  # CNN
            # CNN backbone的实际输出尺寸计算
            # 28x28 -> 14x14 (MaxPool) -> 7x7 (MaxPool) -> 7x7 (no pool)
            # 通用公式: image_size / (2^num_maxpool)
            num_maxpool = 2  # CNN中有2个MaxPool2d
            spatial_size = self.image_size // (2 ** num_maxpool)
            # 确保至少为1
            spatial_size = max(1, spatial_size)
        
        # 池化层
        if self.pooling_type == 'gap':
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            head_input_dim = num_features
        elif self.pooling_type == 'gmp':
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
            head_input_dim = num_features
        else:  # 'flatten'
            self.pooling = nn.Identity()
            head_input_dim = num_features * spatial_size * spatial_size
        
        # 分类头
        self.head = nn.Sequential(
            nn.Dropout(p=self.head_dropout),
            nn.Linear(head_input_dim, self.num_classes)
        )
        
        # 保存spatial size用于forward
        self.spatial_size = spatial_size
        
        # 调试信息（可选）
        if self.pooling_type == 'flatten':
            expected_dim = num_features * spatial_size * spatial_size
            print(f"[LSAFN] Backbone: {self.backbone_type}, "
                  f"num_features: {num_features}, "
                  f"spatial_size: {spatial_size}x{spatial_size}, "
                  f"head_input_dim: {expected_dim}")
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        """
        提取特征
        
        Args:
            x: 输入数据，支持多种格式
                - 单帧: [B, C, H, W]
                - 双帧: (onset, apex) 或 [onset, apex]
                - 字典: {'onset': tensor, 'apex': tensor, 'frames': tensor}
        
        Returns:
            Tensor: 特征图 [B, C, H', W']
        """
        # ========== 解析输入 ==========
        onset, apex, frames = self._parse_input(x)
        
        # ========== ViT-POS位置嵌入 ==========
        pos_embed = None
        if self.use_pos_vit and onset is not None:
            B = onset.shape[0]
            pos_embed = self.vit_pos(self.resize(onset))  # [B, N, 512]
            # Reshape to [B, 512, H, W]
            pos_embed = pos_embed.transpose(1, 2).view(
                B, self.posvit_feature_dim, 
                self.posvit_spatial_size, self.posvit_spatial_size
            )
        
        # ========== 长时动作分支 ==========
        long_action = None
        if self.use_long_action:
            # 计算apex-onset差值
            if apex is not None and onset is not None:
                # 检查是否是同一张图（单帧情况）
                if apex is onset or torch.equal(apex, apex):
                    # 单帧输入：直接使用，不计算差值
                    long_input = onset
                else:
                    # 双帧输入：计算差值
                    long_input = apex - onset
            else:
                long_input = onset if onset is not None else torch.zeros_like(apex)
            
            # Backbone提取特征
            long_action = self.long_action_model(long_input)
            
            # 处理Swin输出格式
            if isinstance(long_action, (list, tuple)):
                long_action = long_action[-1]  # 取最后一个阶段的输出
            
            # 如果是5D（时序），时间维度池化
            if len(long_action.shape) == 5:
                long_action = long_action.mean(dim=2)
            
            # 融合位置嵌入
            if pos_embed is not None:
                if long_action.shape[-2:] != pos_embed.shape[-2:]:
                    pos_embed_resized = F.interpolate(
                        pos_embed, size=long_action.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                else:
                    pos_embed_resized = pos_embed
                long_action = long_action + pos_embed_resized
        
        # ========== 短时动作分支 ==========
        short_action = None
        if self.use_short_action and frames is not None:
            short_action = self.short_action_model(frames)
            
            if isinstance(short_action, (list, tuple)):
                short_action = short_action[-1]
            
            if len(short_action.shape) == 5:
                short_action = short_action.mean(dim=2)
            
            if pos_embed is not None:
                short_action = short_action + pos_embed_resized
        
        # ========== 特征融合 ==========
        if self.use_long_action and self.use_short_action:
            if self.concat:
                out = torch.cat((long_action, short_action), dim=1)
                if self.use_feature_conv:
                    out = self.feature_conv(out)
            else:
                # Element-wise addition
                out = long_action + short_action
        elif self.use_long_action:
            out = long_action
        elif self.use_short_action:
            out = short_action
        else:
            raise ValueError("No action branch enabled")
        
        return out
    
    def forward(self, x):
        """
        完整的前向传播
        
        Args:
            x: 输入数据
        
        Returns:
            Tensor: 分类logits [B, num_classes]
        """
        # 提取特征
        features = self.forward_features(x)
        
        # 池化
        features = self.pooling(features)
        
        # Flatten
        features = features.flatten(start_dim=1)
        
        # 分类
        output = self.head(features)
        
        return output
    
    def _parse_input(self, x):
        """
        解析输入数据
        
        Returns:
            tuple: (onset, apex, frames)
        """
        if isinstance(x, dict):
            # 字典格式
            onset = x.get('onset', None)
            apex = x.get('apex', None)
            frames = x.get('frames', None)
        elif isinstance(x, (list, tuple)):
            # 元组/列表格式
            if len(x) == 2:
                onset, apex = x
                frames = None
            elif len(x) == 3:
                onset, frames, apex = x
            else:
                raise ValueError(f"Unsupported tuple length: {len(x)}")
        else:
            # 单帧格式
            onset = x
            apex = x
            frames = None
        
        return onset, apex, frames
    
    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 向后兼容的别名
LSAFN = LongShortActionFuseNet
