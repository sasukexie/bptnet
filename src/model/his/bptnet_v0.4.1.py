"""
BPTNet v0.4 — Phase-Aware Temporal Attention Network for MER
=============================================================

核心创新：
  1. PhaseCrossAttention — 空间特征交叉注意力到运动相位，显式建模收缩/放松相
  2. CBAM — 轻量空间+通道注意力，弥补 apex 单帧空间感知短板
  3. DropPath — 随机深度正则化，抑制小数据集过拟合

架构概览：
  输入 → ResNet18 (ImageNet预训练) → CBAM → PhaseCrossAttention (多模态融合)
       → PosEmbed + CLS Token → 2层 Transformer (DropPath) → CLS → Head

frame_type 支持：
  - apex:         RGB → ResNet18 → CBAM → Transformer → Head
  - flow:         光流 → 转3ch → ResNet18 → CBAM → Transformer → Head
  - rgb_triplet:  三帧 → 分别ResNet18 → 平均 → CBAM → Transformer → Head
  - rgb_flow:     RGB + 光流 → 双路ResNet18 → PhaseCrossAttention 融合 → Transformer → Head
  - rgb_dual_flow: RGB + 双向光流 → 三路ResNet18 → PhaseCrossAttention (收缩+放松相感知) → Transformer → Head

版本历史：
  v0.2 → his/bptnet_v0.2.py  (原 BPTNet + BPTNetv2 双类版)
  v0.3 → his/bptnet_v0.3.py  (BPTNetv2 架构整合为 BPTNet, ResNet18 + 简单相加融合)
  v0.4 → 当前版本              (PhaseCrossAttention + CBAM + DropPath)
"""

import torch
from torch import nn


# ------------------------------------------------------------------
#  DropPath (Stochastic Depth) — 仅训练时生效
# ------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float, training: bool):
    """随机丢弃整个残差路径，参考 Deep Networks with Stochastic Depth (Huang et al.)"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ------------------------------------------------------------------
#  CBAM — Convolutional Block Attention Module
#  轻量级空间+通道注意力，弥补纯单帧空间感知短板
# ------------------------------------------------------------------

class CBAM(nn.Module):
    """
    CBAM: 先通道注意力，再空间注意力。
    对 224→7×7 特征图用 kernel_size=7，对小特征图自适应降为 3。
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        # ---- 通道注意力 ----
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # ---- 空间注意力 (kernel_size 自适应) ----
        ks = max(3, kernel_size)
        if ks % 2 == 0:
            ks += 1
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, ks, padding=ks // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca = self.channel_attn(x)  # [B, C, 1, 1]
        x = x * ca
        # 空间注意力
        sa_input = torch.cat([x.mean(dim=1, keepdim=True),
                              x.max(dim=1, keepdim=True)[0]], dim=1)  # [B, 2, H, W]
        sa = self.spatial_attn(sa_input)
        return x * sa


# ------------------------------------------------------------------
#  PhaseCrossAttention — 相位感知交叉注意力 (核心创新)
# ------------------------------------------------------------------

class PhaseCrossAttention(nn.Module):
    """
    相位感知交叉注意力融合模块。

    设计动机：
      微表情的核心在"动"，而非"静"。onset→apex（肌肉收缩）与 apex→offset（肌肉放松）
      是两个非对称相位，其运动强度、方向、持续时间均不同。
      本模块让空间特征 (RGB) 通过 cross-attention 主动查询运动相位特征，
      由可学习的相位权重 + 门控机制自适应融合。

    使用场景：
      - rgb_flow (单相位): RGB 交叉注意力到 onset→apex 光流
      - rgb_dual_flow (双相位): RGB 分别交叉注意力到收缩相和放松相

    Args:
        dim: 特征通道数
        num_phases: 运动相位数 (1 for rgb_flow, 2 for rgb_dual_flow)
        num_heads: 交叉注意力头数
        dropout: 注意力 dropout
    """

    def __init__(self, dim: int, num_phases: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_phases = num_phases

        # 可学习的相位重要性权重 (如收缩相 vs 放松相)
        self.phase_weights = nn.Parameter(torch.ones(num_phases) / num_phases)

        # Cross-Attention: 空间特征 (query) 关注运动特征 (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 门控: 自适应决定融入多少相位信息
        gate_dim = dim * (num_phases + 1)  # RGB + 各相位特征
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(gate_dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, spatial_feat: torch.Tensor,
                *phase_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_feat: [B, C, H, W]  RGB 骨干特征
            phase_feats:   [B, C, H, W]  运动相位特征 (1个或2个)
        Returns:
            [B, C, H, W]  相位感知融合特征
        """
        B, C, H, W = spatial_feat.shape

        # 空间 tokens → query
        spatial_tokens = spatial_feat.flatten(2).transpose(1, 2)  # [B, N, C]

        # 对每个相位做交叉注意力
        phase_weights = torch.softmax(self.phase_weights, dim=0)
        phase_outputs = []
        for i, phase_feat in enumerate(phase_feats):
            phase_tokens = phase_feat.flatten(2).transpose(1, 2)  # [B, N, C]
            out, _ = self.cross_attn(spatial_tokens, phase_tokens, phase_tokens)
            phase_outputs.append(out * phase_weights[i])

        # 加权合并各相位贡献
        cross = sum(phase_outputs)  # [B, N, C]
        cross = cross.transpose(1, 2).reshape(B, C, H, W)

        # 门控融合: 自适应控制运动信息的融入比例
        all_feats = torch.cat([spatial_feat] + list(phase_feats), dim=1)  # [B, (P+1)*C, H, W]
        gate_val = self.gate(all_feats)  # [B, 1]
        gate_val = gate_val.view(B, 1, 1, 1)  # [B, 1, 1, 1] 确保4D广播

        return spatial_feat + gate_val * cross


# ===================================================================
#  BPTNet v0.4
# ===================================================================

class BPTNet(nn.Module):
    """
    BPTNet: Phase-Aware Temporal Attention Network

    ResNet18 (预训练) → CBAM → [PhaseCrossAttention] → Transformer → Head

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

        # ---- 解析帧类型 ----
        ft = data_config.get('frame_type', 'apex')
        ft_config = data_config.get('frame_type_config', {}).get(ft, {})
        self.is_multimodal = ft_config.get('is_multimodal', False)
        self.is_dual_flow = ft_config.get('is_dual_flow', False)
        self.is_triplet = (ft == 'rgb_triplet')
        self.is_flow_only = (ft == 'flow')

        # ---- 架构参数 ----
        backbone_type = model_config.get('backbone', 'resnet18')
        pretrained = model_config.get('pretrained', True)
        dropout = model_config.get('dropout', 0.15)
        transformer_dim = model_config.get('transformer_dim', 512)
        transformer_layers = model_config.get('transformer_layers', 2)
        transformer_heads = model_config.get('transformer_heads', 8)
        drop_path_rate = model_config.get('drop_path_rate', 0.1)

        # 新模块开关
        use_cbam = model_config.get('use_cbam', True)
        phase_fusion = model_config.get('phase_fusion', 'cross_attention')

        # ==================== Backbone ====================
        if backbone_type == 'resnet18':
            backbone, out_channels = self._build_resnet18(pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        self.backbone = backbone
        self.backbone_out_channels = out_channels

        # 投影到 transformer 维度
        if out_channels != transformer_dim:
            self.proj = nn.Conv2d(out_channels, transformer_dim, 1)
        else:
            self.proj = nn.Identity()

        # ==================== CBAM 注意力 ====================
        fmap_size = image_size // 32
        cbam_ks = 7 if fmap_size >= 7 else 3

        self.cbam = None
        self.cbam_flow = None
        self.cbam_flow_ao = None
        if use_cbam:
            self.cbam = CBAM(transformer_dim, kernel_size=cbam_ks)
            if self.is_multimodal:
                self.cbam_flow = CBAM(transformer_dim, kernel_size=cbam_ks)
                if self.is_dual_flow:
                    self.cbam_flow_ao = CBAM(transformer_dim, kernel_size=cbam_ks)

        # ==================== 相位感知融合 ====================
        self.phase_fusion = None
        if self.is_multimodal and phase_fusion == 'cross_attention':
            num_phases = 2 if self.is_dual_flow else 1
            self.phase_fusion = PhaseCrossAttention(
                dim=transformer_dim,
                num_phases=num_phases,
                num_heads=min(transformer_heads, 4),
                dropout=dropout,
            )

        # ==================== Position Embedding ====================
        self.pos_embed = nn.Parameter(
            torch.randn(1, transformer_dim, fmap_size, fmap_size) * 0.02
        )

        # ==================== CLS Token ====================
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)

        # ==================== Transformer (DropPath) ====================
        self.drop_paths = nn.ModuleList()
        drop_rates = [drop_path_rate * i / max(transformer_layers - 1, 1)
                      for i in range(transformer_layers)]

        self.transformer_layers = nn.ModuleList()
        for i in range(transformer_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dropout=dropout,
                dim_feedforward=transformer_dim * 4,
                batch_first=True,
                activation='gelu',
                norm_first=True,  # Pre-norm，配合 DropPath
            )
            self.transformer_layers.append(layer)
            self.drop_paths.append(DropPath(drop_rates[i]))

        # ==================== 分类头 ====================
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Linear(transformer_dim, num_classes)

        # 初始化非预训练权重
        self._init_weights()

    # ------------------------------------------------------------------
    #  构建方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_resnet18(pretrained: bool):
        import torchvision.models as models
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        # 移除 avgpool + fc，保留到 layer4
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        return backbone, 512

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'backbone' in name:
                continue  # 预训练权重不动
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # CLS token + pos embed 特殊处理
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, 'pos_embed'):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # ------------------------------------------------------------------
    #  输入转换 & 特征提取
    # ------------------------------------------------------------------

    def _to_rgb_input(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c == 3:
            return x
        elif c == 2:
            return torch.cat([x, x[:, :1]], dim=1)  # [B, 2, H, W] → [B, 3, H, W]
        elif c == 9:
            B, _, H, W = x.shape
            return x.reshape(B * 3, 3, H, W)
        return x

    def _extract_features(self, x: torch.Tensor,
                          is_triplet: bool = False,
                          apply_cbam: nn.Module = None) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.proj(feat)

        if is_triplet:
            B3, C, H_f, W_f = feat.shape
            B = B3 // 3
            feat = feat.reshape(B, 3, C, H_f, W_f).mean(dim=1)

        if apply_cbam is not None:
            feat = apply_cbam(feat)

        return feat

    # ------------------------------------------------------------------
    #  多模态融合
    # ------------------------------------------------------------------

    def _fuse_multimodal(self, f_rgb: torch.Tensor,
                         f_flow_oa: torch.Tensor,
                         f_flow_ao: torch.Tensor = None) -> torch.Tensor:
        """
        多模态特征融合。
        - 有 PhaseCrossAttention → 相位感知融合
        - 否则 → 简单求和 (v0.3 行为)
        """
        if self.phase_fusion is not None and self.is_multimodal:
            # PhaseCrossAttention 融合
            if f_flow_ao is not None:
                return self.phase_fusion(f_rgb, f_flow_oa, f_flow_ao)
            else:
                return self.phase_fusion(f_rgb, f_flow_oa)
        else:
            # 回退: 简单求和
            fused = f_rgb + f_flow_oa
            if f_flow_ao is not None:
                fused = fused + f_flow_ao
            return fused

    # ------------------------------------------------------------------
    #  Transformer 前向
    # ------------------------------------------------------------------

    def _forward_transformer(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape

        # 自适应位置编码插值
        if H != self.pos_embed.shape[2] or W != self.pos_embed.shape[3]:
            pe = nn.functional.interpolate(
                self.pos_embed, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            pe = self.pos_embed

        # 转为 token 序列
        feat = feat.flatten(2).transpose(1, 2)    # [B, N, C]
        pe = pe.flatten(2).transpose(1, 2)         # [B, N, C]

        feat = feat + pe

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls_tokens, feat], dim=1)  # [B, 1+N, C]

        # Transformer layers (Pre-Norm + DropPath)
        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            feat = dp(layer(feat))

        # CLS → LayerNorm → Head
        feat = self.norm(feat[:, 0])
        return self.head(feat)

    # ------------------------------------------------------------------
    #  Main Forward
    # ------------------------------------------------------------------

    def forward(self, img):
        """
        前向传播，自动适配所有 frame_type。

        Returns:
            [B, num_classes] logits
        """
        if isinstance(img, (list, tuple)):
            # ======== 多模态 ========
            cbam_rgb = self.cbam
            cbam_flow = self.cbam_flow
            cbam_flow_ao = self.cbam_flow_ao

            if self.is_dual_flow:
                rgb, flow_oa, flow_ao = img
                f_rgb = self._extract_features(self._to_rgb_input(rgb), apply_cbam=cbam_rgb)
                f_oa = self._extract_features(self._to_rgb_input(flow_oa), apply_cbam=cbam_flow)
                f_ao = self._extract_features(self._to_rgb_input(flow_ao), apply_cbam=cbam_flow_ao)
                feat = self._fuse_multimodal(f_rgb, f_oa, f_ao)
            else:
                # rgb_flow
                rgb, flow = img
                f_rgb = self._extract_features(self._to_rgb_input(rgb), apply_cbam=cbam_rgb)
                f_flow = self._extract_features(self._to_rgb_input(flow), apply_cbam=cbam_flow)
                feat = self._fuse_multimodal(f_rgb, f_flow)
        else:
            # ======== 单模态 ========
            if self.is_triplet:
                B, _, H, W = img.shape
                img = img.reshape(B * 3, 3, H, W).contiguous()
                feat = self._extract_features(img, is_triplet=True, apply_cbam=self.cbam)
            else:
                img_rgb = self._to_rgb_input(img)
                feat = self._extract_features(img_rgb, apply_cbam=self.cbam)

        return self._forward_transformer(feat)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
