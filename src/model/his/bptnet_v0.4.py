"""
BPTNet 核心模型实现
CNN Backbone (ResNet18 pretrained) + Lightweight Transformer for MER

核心设计：
  输入 → ResNet18 (ImageNet预训练) → 特征图 [B,512,H/32,W/32]
       → PosEmbed + CLS Token
       → 2层 TransformerEncoder
       → CLS → LayerNorm → Linear(num_classes)

设计原则：
1. ResNet18 ImageNet 预训练骨干——利用大规模视觉先验
2. 2层轻量 Transformer——避免小数据集过拟合
3. 强正则化（Dropout + 数据增强 + LR scheduling）
4. 多模态统一处理——所有输入通过预训练骨干提取特征后简单相加融合

frame_type 支持：
  - apex:         RGB → ResNet18 → Transformer → Head
  - flow:         光流 → 转3ch → ResNet18 → Transformer → Head
  - rgb_triplet:  三帧 → 分别ResNet18 → 平均 → Transformer → Head
  - rgb_flow:     RGB + 光流 → 双路ResNet18 → 相加融合 → Transformer → Head
  - rgb_dual_flow: RGB + 双向光流 → 三路ResNet18 → 相加融合 → Transformer → Head

版本历史：
  v0.2 → his/bptnet_v0.2.py   (原 BPTNet + BPTNetv2 双类版)
  v0.3 → 当前版本               (BPTNetv2 架构整合为 BPTNet)
"""

import torch
from torch import nn


class BPTNet(nn.Module):
    """
    BPTNet: CNN Backbone + Lightweight Transformer for MER

    Args:
        config: 配置字典
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        data_config = config.get('data', {})

        image_size = data_config.get('image_size', 224)
        num_classes = data_config.get('num_classes', 3)

        # Parse frame type
        ft = data_config.get('frame_type', 'apex')
        ft_config = data_config.get('frame_type_config', {}).get(ft, {})
        self.is_multimodal = ft_config.get('is_multimodal', False)
        self.is_dual_flow = ft_config.get('is_dual_flow', False)
        self.is_triplet = (ft == 'rgb_triplet')

        # Architecture params
        backbone_type = model_config.get('backbone', 'resnet18')
        pretrained = model_config.get('pretrained', True)
        dropout = model_config.get('dropout', 0.15)
        transformer_dim = model_config.get('transformer_dim', 512)
        transformer_layers = model_config.get('transformer_layers', 2)
        transformer_heads = model_config.get('transformer_heads', 8)

        # ---- ResNet18 Backbone (pretrained) ----
        if backbone_type == 'resnet18':
            backbone, out_channels = self._build_resnet18(pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        self.backbone = backbone
        self.backbone_out_channels = out_channels  # 512

        # ---- Project backbone output to transformer dim ----
        if out_channels != transformer_dim:
            self.proj = nn.Conv2d(out_channels, transformer_dim, 1)
        else:
            self.proj = nn.Identity()

        # ---- Position Embedding (2D, bilinear-interpolatable) ----
        fmap_size = image_size // 32
        self.pos_embed = nn.Parameter(
            torch.randn(1, transformer_dim, fmap_size, fmap_size) * 0.02
        )

        # ---- CLS Token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)

        # ---- Lightweight Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dropout=dropout,
            dim_feedforward=transformer_dim * 4,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # ---- Classification Head ----
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Linear(transformer_dim, num_classes)

        # 初始化 transformer 相关参数
        self._init_weights()

    # ------------------------------------------------------------------
    #  Build helpers
    # ------------------------------------------------------------------

    def _build_resnet18(self, pretrained: bool):
        """构建 ResNet18，返回 (backbone, out_channels)"""
        import torchvision.models as models

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # 移除最后的 avgpool 和 fc，保留到 layer4 输出
        # 输出: [B, 512, H/32, W/32]
        modules = list(resnet.children())[:-2]
        backbone = nn.Sequential(*modules)
        return backbone, 512

    def _init_weights(self):
        """初始化非预训练部分的权重"""
        for name, p in self.named_parameters():
            if 'backbone' in name:
                continue  # backbone 已有预训练权重
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ------------------------------------------------------------------
    #  Input conversion & feature extraction
    # ------------------------------------------------------------------

    def _to_rgb_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        将非 RGB 输入转换为 3 通道，以利用预训练骨干。

        - 2ch flow → 复制第0通道 → 3ch
        - 9ch triplet → reshape 为 [B*3, 3, H, W]
        """
        if x.shape[1] == 3:
            return x  # 已经是 RGB
        elif x.shape[1] == 2:
            # 光流: 复制 dx 通道补充为 3 通道
            return torch.cat([x, x[:, :1]], dim=1)  # [B, 2, H, W] → [B, 3, H, W]
        elif x.shape[1] == 9:
            # 三帧堆叠: [B, 9, H, W] → [B*3, 3, H, W]
            B, _, H, W = x.shape
            return x.reshape(B * 3, 3, H, W)
        else:
            return x

    def _extract_features(self, x: torch.Tensor, is_triplet: bool = False) -> torch.Tensor:
        """
        通过骨干网络提取特征。

        - 普通输入: [B, 3, H, W] → [B, C, H/32, W/32]
        - triplet 输入: [B*3, 3, H, W] → mean → [B, C, H/32, W/32]
        """
        feat = self.backbone(x)        # [B_out, C, H_f, W_f]
        feat = self.proj(feat)

        if is_triplet:
            # 恢复帧维度并平均池化
            B3, C, H_f, W_f = feat.shape
            B = B3 // 3
            feat = feat.reshape(B, 3, C, H_f, W_f).mean(dim=1)

        return feat

    # ------------------------------------------------------------------
    #  Transformer forward
    # ------------------------------------------------------------------

    def _forward_transformer(self, feat: torch.Tensor) -> torch.Tensor:
        """
        通过 Transformer + 分类头。

        Args:
            feat: [B, C, H, W] 特征图
        Returns:
            [B, num_classes] logits
        """
        B, C, H, W = feat.shape

        # 自适应插值位置编码
        if H != self.pos_embed.shape[2] or W != self.pos_embed.shape[3]:
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            pos_embed = self.pos_embed

        # [B, C, H, W] → [B, H*W, C]
        feat = feat.flatten(2).transpose(1, 2)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)

        # 位置编码 + CLS token
        feat = feat + pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls_tokens, feat], dim=1)  # [B, 1+H*W, C]

        # Transformer
        feat = self.transformer(feat)

        # CLS token 分类
        feat = self.norm(feat[:, 0])
        return self.head(feat)

    # ------------------------------------------------------------------
    #  Main forward
    # ------------------------------------------------------------------

    def forward(self, img):
        """
        前向传播。支持所有 frame_type，自动适配输入格式。

        Args:
            img:
              - apex:          [B, 3, H, W]
              - flow:          [B, 2, H, W]
              - rgb_triplet:   [B, 9, H, W]
              - rgb_flow:      ((B,3,H,W), (B,2,H,W))
              - rgb_dual_flow: ((B,3,H,W), (B,2,H,W), (B,2,H,W))
        Returns:
            [B, num_classes] logits
        """
        if isinstance(img, (list, tuple)):
            # ======== 多模态: 各路独立提取 + 求和融合 ========
            if self.is_dual_flow:
                rgb, flow_oa, flow_ao = img
                f_rgb = self._extract_features(
                    self._to_rgb_input(rgb), is_triplet=False)
                f_oa = self._extract_features(
                    self._to_rgb_input(flow_oa), is_triplet=False)
                f_ao = self._extract_features(
                    self._to_rgb_input(flow_ao), is_triplet=False)
                feat = f_rgb + f_oa + f_ao
            else:
                # rgb_flow: 双路
                rgb, flow = img
                f_rgb = self._extract_features(
                    self._to_rgb_input(rgb), is_triplet=False)
                f_flow = self._extract_features(
                    self._to_rgb_input(flow), is_triplet=False)
                feat = f_rgb + f_flow
        else:
            # ======== 单模态 ========
            if self.is_triplet:
                # Triplet: [B, 9, H, W] → [B*3, 3, H, W] → backbone → mean
                B, _, H, W = img.shape
                img = img.reshape(B * 3, 3, H, W).contiguous()
                feat = self._extract_features(img, is_triplet=True)
            else:
                img_rgb = self._to_rgb_input(img)
                feat = self._extract_features(img_rgb, is_triplet=False)

        return self._forward_transformer(feat)

    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
