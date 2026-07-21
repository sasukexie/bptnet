"""
BPTNet 核心模型实现
Phase-Aware Hierarchical Transformer Network for Micro-Expression Recognition

基于 HTNet 改进：
1. 多层次特征提取（局部细粒度 + 全局粗粒度）
2. Phase Gating: 自适应融合 onset→apex 和 apex→offset 双向光流
3. RGB Guidance: RGB 外观特征引导运动注意力
4. 自注意力机制捕捉肌肉运动关系

支持 frame_type:
  - apex:        单帧 RGB → 原始 HTNet 模式
  - flow:        onset→apex 纯光流
  - rgb_triplet: onset+apex+offset 三帧通道堆叠
  - rgb_flow:    RGB + onset→apex 光流 (通道拼接)
  - rgb_dual_flow: RGB + onset→apex光流 + apex→offset光流 (Phase Gating 融合)
"""

import torch
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn, einsum


def cast_tuple(val, depth):
    """
    将值转换为元组
    
    Args:
        val: 输入值
        depth: 元组深度
        
    Returns:
        tuple: 转换后的元组
    """
    return val if isinstance(val, tuple) else ((val,) * depth)


class LayerNorm(nn.Module):
    """
    自定义层归一化
    
    Args:
        dim (int): 特征维度
        eps (float): 数值稳定性参数
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    """
    预归一化包装器
    
    Args:
        dim (int): 特征维度
        fn (nn.Module): 要包装的函数/层
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    前馈神经网络模块
    
    Args:
        dim (int): 输入/输出维度
        mlp_mult (int): MLP扩展倍数
        dropout (float): Dropout概率
    """

    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    多头自注意力机制
    
    Args:
        dim (int): 输入维度
        heads (int): 注意力头数
        dropout (float): Dropout概率
    """

    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        # 生成Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), qkv)

        # 计算注意力分数
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 应用注意力
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 加权求和
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


def Aggregate(dim, dim_out):
    """
    特征聚合模块（卷积 + 归一化 + 池化）
    
    Args:
        dim (int): 输入维度
        dim_out (int): 输出维度
        
    Returns:
        nn.Sequential: 聚合模块
    """
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )


class Transformer(nn.Module):
    """
    Transformer编码器块
    
    Args:
        dim (int): 特征维度
        seq_len (int): 序列长度
        depth (int): Transformer层数
        heads (int): 注意力头数
        mlp_mult (int): MLP扩展倍数
        dropout (float): Dropout概率
    """

    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        *_, h, w = x.shape

        # 添加位置编码
        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h=h, w=w)
        x = x + pos_emb

        # 逐层处理
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PhaseGate(nn.Module):
    """
    自适应相位门控模块

    学习 onset→apex（运动增强期）和 apex→offset（运动衰减期）
    两个相位光流在各空间位置的融合权重。

    Args:
        dim (int): 特征维度
    """

    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, f_oa, f_ao):
        """
        Args:
            f_oa: onset→apex 光流特征 [B, D, h, w]
            f_ao: apex→offset 光流特征 [B, D, h, w]
        Returns:
            相位门控融合特征 [B, D, h, w]
        """
        gate = torch.cat([f_oa, f_ao], dim=1)   # [B, 2D, h, w]
        gate = self.gate_conv(gate)              # [B, 1, h, w]
        return gate * f_oa + (1 - gate) * f_ao   # 逐位置加权融合


class RGBGuide(nn.Module):
    """
    RGB 外观引导注意力模块

    用 RGB 外观特征生成空间注意力图，引导/增强光流运动特征。
    直觉：面部外观明确的区域（如眼角、嘴角）运动信息更有判别力。

    Args:
        dim (int): 特征维度
    """

    def __init__(self, dim):
        super().__init__()
        self.guide = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, flow_feat):
        """
        Args:
            rgb_feat:  RGB 外观特征 [B, D, h, w]
            flow_feat: 光流运动特征 [B, D, h, w]
        Returns:
            引导增强后的运动特征 [B, D, h, w]
        """
        return flow_feat * self.guide(rgb_feat)


class BPTNet(nn.Module):
    """
    BPTNet主模型：相位感知层次化Transformer网络

    基于 HTNet 改进，支持双向光流相位感知融合。
    非 dual_flow 模式下完全兼容原 HTNet 行为。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        data_config = config['data']
        image_size = data_config.get('image_size', 28)
        patch_size = model_config.get('patch_size', 7)
        num_classes = data_config.get('num_classes', 3)
        dim = model_config.get('dim', 256)
        heads = model_config.get('heads', 3)
        num_hierarchies = model_config.get('num_hierarchies', 3)
        block_repeats = tuple(model_config.get('block_repeats', [2, 2, 10]))
        mlp_mult = model_config.get('mlp_mult', 4)
        
        # 使用配置文件中的 frame_type_config
        ft_config = data_config.get('frame_type_config').get(data_config.get('frame_type'))
        channels = ft_config['channels']
        self.is_multimodal = ft_config.get('is_multimodal', False)
        self.is_dual_flow = ft_config.get('is_dual_flow', False)
        
        # dim_head = model_config.get('dim_head', 64)
        dropout = model_config.get('dropout', 0.)

        # 参数验证
        assert (image_size % patch_size) == 0, \
            'Image dimensions must be divisible by the patch size.'

        # 计算基础参数
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)
        seq_len = (fmap_size // blocks) ** 2

        # 计算各层级的维度
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]
        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]
        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        # ---- Patch嵌入层 ----
        if self.is_dual_flow:
            # 双流光流模式：RGB + onset→apex 光流 + apex→offset 光流各自独立嵌入
            self.rgb_embed = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w',
                          p1=patch_size, p2=patch_size),
                nn.Conv2d(3 * patch_size ** 2, layer_dims[0], 1),
            )
            self.flow_oa_embed = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w',
                          p1=patch_size, p2=patch_size),
                nn.Conv2d(2 * patch_size ** 2, layer_dims[0], 1),
            )
            self.flow_ao_embed = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w',
                          p1=patch_size, p2=patch_size),
                nn.Conv2d(2 * patch_size ** 2, layer_dims[0], 1),
            )
            # 相位门控 + RGB 引导
            self.phase_gate = PhaseGate(layer_dims[0])
            self.rgb_guide = RGBGuide(layer_dims[0])
        else:
            # 原始模式：通道拼接嵌入（兼容 apex / flow / rgb_triplet / rgb_flow）
            patch_dim = channels * patch_size ** 2
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w',
                          p1=patch_size, p2=patch_size),
                nn.Conv2d(patch_dim, layer_dims[0], 1),
            )

        # 构建多层级Transformer
        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])

        for level, heads, (dim_in, dim_out), block_repeat in \
                zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        # 分类头
        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        """
        前向传播
        
        Args:
            img: 输入数据
               - 单模态RGB: [B, 3, H, W]                                → apex 模式
               - 纯光流: [B, 2, H, W]                                   → flow 模式
               - 通道堆叠: [B, 9, H, W]                                 → rgb_triplet 模式
               - 2-tuple: ([B, 3, H, W], [B, 2, H, W])                  → rgb_flow 模式
               - 3-tuple: ([B, 3, H, W], [B, 2, H, W], [B, 2, H, W])  → rgb_dual_flow 模式
            
        Returns:
            Tensor: 分类 logits [B, num_classes]
        """
        if isinstance(img, (list, tuple)) and self.is_dual_flow:
            # ---- 相位感知双流模式 ----
            rgb, flow_oa, flow_ao = img

            # 各自独立 Patch Embedding
            r = self.rgb_embed(rgb)          # [B, D, h, w]
            f_oa = self.flow_oa_embed(flow_oa)
            f_ao = self.flow_ao_embed(flow_ao)

            # Phase Gating: 自适应融合两个相位的运动信息
            f_fused = self.phase_gate(f_oa, f_ao)

            # RGB 引导: 外观特征增强运动特征
            f_fused = self.rgb_guide(r, f_fused)

            # 残差融合: RGB + 运动
            x = r + f_fused

        elif isinstance(img, (list, tuple)) and self.is_multimodal:
            # ---- 原始多模态模式 (rgb_flow) ----
            rgb, flow = img
            # 拼接通道: [B, 5, H, W]
            img = torch.cat([rgb, flow], dim=1)
            x = self.to_patch_embedding(img)

        else:
            # ---- 单模态模式 (apex / flow / rgb_triplet) ----
            x = self.to_patch_embedding(img)

        num_hierarchies = len(self.layers)

        # 逐层处理 (与原 HTNet 一致)
        for level, (transformer, aggregate) in \
                zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w',
                          b1=block_size, b2=block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)',
                          b1=block_size, b2=block_size)
            x = aggregate(x)

        return self.mlp_head(x)

    def get_num_parameters(self):
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===================================================================
# BPTNetv2: CNN Backbone + Lightweight Transformer Hybrid
# ===================================================================
# 
# 核心改进（相对于 BPTNet）：
# 1. ResNet18 预训练骨干（ImageNet）替代从零训练的 Patch Embedding
# 2. 2层轻量 Transformer 替代 14 层深层 Transformer
# 3. 强正则化（Dropout=0.15, 数据增强, LR scheduling）
# 4. 多模态统一处理——所有输入通过预训练骨干提取特征
#
# frame_type 支持：
#   - apex:        RGB → ResNet18 → Transformer → Head
#   - flow:        光流 → 转3ch → ResNet18 → Transformer → Head
#   - rgb_triplet: 三帧 → 分别ResNet18 → 平均 → Transformer → Head
#   - rgb_flow:    RGB + 光流 → 双路ResNet18 → 相加融合 → Transformer → Head
#   - rgb_dual_flow: RGB + 双向光流 → 三路ResNet18 → 相加融合 → Transformer → Head
# ===================================================================

class BPTNetv2(nn.Module):
    """
    BPTNet-v2: CNN Backbone + Lightweight Transformer for MER

    设计原则：
    ┌─────────────────────────────────────────────────────┐
    │  输入 → ResNet18 (预训练) → 特征图 [B,512,H/32,W/32]  │
    │       → PosEmbed + CLS Token                          │
    │       → 2层 TransformerEncoder                        │
    │       → CLS → LayerNorm → Linear(num_classes)         │
    └─────────────────────────────────────────────────────┘

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

        # ---- Project backbone output to transformer dim (if needed) ----
        if out_channels != transformer_dim:
            self.proj = nn.Conv2d(out_channels, transformer_dim, 1)
        else:
            self.proj = nn.Identity()

        # ---- Position Embedding (2D, bilinear-interpolatable) ----
        # 初始化在 7x7 网格上（224/32=7），推理时自适应插值
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
            # 未知通道数，尝试直接通过（可能报错）
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

    def forward(self, img):
        """
        前向传播。支持所有 frame_type，自动适配输入格式。

        Args:
            img: 
              - apex:  [B, 3, H, W]
              - flow:  [B, 2, H, W]
              - rgb_triplet: [B, 9, H, W]
              - rgb_flow: ((B,3,H,W), (B,2,H,W))
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
                f_oa  = self._extract_features(
                    self._to_rgb_input(flow_oa), is_triplet=False)
                f_ao  = self._extract_features(
                    self._to_rgb_input(flow_ao), is_triplet=False)
                feat = f_rgb + f_oa + f_ao
            else:
                # rgb_flow: 双路
                rgb, flow = img
                f_rgb  = self._extract_features(
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
