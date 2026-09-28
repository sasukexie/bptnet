"""
BPTNet v0.6 — Phase-Aware Temporal Attention Network for MER
=============================================================

  核心创新：
  1. GatedSumFusion (v0.8 推荐) — 各模态先 per-location LayerNorm 对齐尺度、
     再加权求和：flow 空间内容完整保留（纯 flow 超集），apex RGB 互补整合；
     **根因修复**：RGB 自然图像 与 光流 特征尺度/分布差异巨大，直接求和/加权
     （含旧 FiLM 以 RGB 为 carrier）会触发类别坍缩（rgb_dual_flow 混淆矩阵几乎
     只预测多数类），归一化+门控求和后 dual_flow ≥ flow 且融合 apex 优势。
    2. CBAM — 轻量空间+通道注意力，flow 模态关闭通道注意力保护运动模式
  3. DropPath — 随机深度正则化，正确包裹每个残差分支（非整个 block 输出）
  4. Mixup — batch-level 数据增强，缓解极小数集过拟合
  5. Phase-Token Temporal Attention — 把 onset/apex/offset（或 apex/oa/ao）各相位
     聚合为全局 phase token 并注入可学习相位位置编码，让 CLS 经 Transformer
     自注意力**显式跨相位时序建模**（使"Temporal Attention"名副其实；统一 1/2/3 相位输入）

架构概览：
  输入 → ResNet18 (ImageNet预训练) → CBAM(按需) → [FiLMFusion/PhaseCrossAttention/Sum] 融合
       → PosEmbed + CLS Token → 2层 Post-Norm TransformerBlock (DropPath) → CLS → Head

frame_type 支持：
  - apex:         RGB → ResNet18 → [无CBAM] → Transformer → Head
  - flow:         光流 → 转3ch → ResNet18 → CBAM(仅空间) → Transformer → Head
  - rgb_triplet:  三帧 → 分别ResNet18 → 平均 → CBAM → Transformer → Head
  - rgb_flow:     RGB + 光流 → 双路ResNet18+CBAM → FiLMFusion 融合 → Transformer → Head
  - rgb_dual_flow: RGB + 双向光流 → 三路ResNet18+CBAM → FiLMFusion 融合 → Transformer → Head

"""

import torch
from torch import nn

from src.utils.logger import logger


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
#  TransformerBlock — Post-Norm Transformer 块，DropPath 正确包裹残差分支
# ------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Post-Norm Transformer 块，DropPath 仅作用于各自残差分支。

    与 nn.TransformerEncoderLayer 的区别：
      - 用 Post-Norm（Norm 在残差相加之后），对极小数集提供更强的隐式正则化
      - DropPath 正确应用于每个残差分支，而非整个 block 输出

    Post-Norm 公式:
      x = Norm(x + drop_path(Sublayer(x)))   ← Norm 在最外层
    (Pre-Norm 则是 x = x + drop_path(Sublayer(Norm(x)))，训练更稳但隐式正则化弱)
    """

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.15,
                 drop_path_rate: float = 0.0, ff_mult: int = 4):
        super().__init__()
        # ---- Self-Attention 分支 ----
        self.attn = nn.MultiheadAttention(dim, heads,
                                           dropout=dropout,
                                           batch_first=True)
        self.drop_path1 = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(dim)

        # ---- FFN 分支 ----
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- Self-Attention 分支 (Post-Norm) ----
        # x = Norm1(x + drop_path1(Attn(x)))
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop_path1(attn_out))

        # ---- FFN 分支 (Post-Norm) ----
        # x = Norm2(x + drop_path2(MLP(x)))
        x = self.norm2(x + self.drop_path2(self.mlp(x)))
        return x


# ------------------------------------------------------------------
#  CBAM — Convolutional Block Attention Module
#  轻量级空间+通道注意力，弥补纯单帧空间感知短板
# ------------------------------------------------------------------

class CBAM(nn.Module):
    """
    CBAM: 先通道注意力，再空间注意力。
    对 224→7×7 特征图用 kernel_size=7，对小特征图自适应降为 3。

    Args:
        use_channel_attn: 是否启用通道注意力。Flow 模态建议关闭，
                          因为光流通道非 RGB 语义，通道重标定可能破坏运动模式。
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7,
                 use_channel_attn: bool = True):
        super().__init__()
        # ---- 通道注意力 ----
        if use_channel_attn:
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(channels // reduction, channels, 1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.channel_attn = nn.Identity()
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
                 num_heads: int = 4, dropout: float = 0.1,
                 cross_attn_dropout: float = None):
        super().__init__()
        self.dim = dim
        self.num_phases = num_phases

        # 可学习的相位重要性权重 (如收缩相 vs 放松相)
        self.phase_weights = nn.Parameter(torch.ones(num_phases) / num_phases)

        # Cross-Attention: 空间特征 (query) 关注运动特征 (key/value)
        # 使用更高的 dropout 防止极小数集下交叉注意力过拟合
        if cross_attn_dropout is None:
            cross_attn_dropout = max(dropout * 2, 0.3)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=cross_attn_dropout,
            batch_first=True,
        )

        # 门控: 自适应决定融入多少相位信息
        # final Linear 设置正偏置，防止 gate 退化到 0 导致相位信息被完全忽略
        gate_dim = dim * (num_phases + 1)  # RGB + 各相位特征
        gate_final = nn.Linear(dim // 4, 1)
        # 正偏置 → 初始 gate≈0.73，保证相位融合路径在训练初期有足够梯度
        # （正偏置使初始 gate≈0.73，保证相位融合路径在训练初期有足够梯度）
        nn.init.constant_(gate_final.bias, 1.0)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(gate_dim, dim // 4),
            nn.GELU(),
            gate_final,
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


# ------------------------------------------------------------------
#  FiLMFusion — 特征线性调制融合 (替代 Cross-Attention)
# ------------------------------------------------------------------

class FiLMFusion(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) 多模态融合。

    设计动机：
      Cross-Attention 要求 RGB 和光流特征的 token 空间兼容，但两个独立
      ResNet18 骨干提取的特征处于不同 embedding 空间，直接做 cross-attention
      会产生无意义的注意力分布。

      FiLM 通过通道级调制避免了这个对齐问题:
        FiLM(rgb | flow) = gamma(flow) ⊙ rgb + beta(flow)

      运动相位信息通过轻量 MLP 生成 gamma/beta，直接调制 RGB 特征的通道，
      无需 token 间的高维注意力运算。

    相比 Cross-Attention:
      - 参数量少 ~40% (MLP vs MultiheadAttention)
      - 无需 query/key/value 空间对齐，天然适合异构特征融合
      - 对小数据集更稳定

    Args:
        dim: 特征通道数
        num_phases: 运动相位数 (1 for rgb_flow, 2 for rgb_dual_flow)
        reduction: 调制 MLP 瓶颈缩减比
    """

    def __init__(self, dim: int, num_phases: int = 2, reduction: int = 4):
        super().__init__()
        self.dim = dim
        self.num_phases = num_phases

        # 调制网络: concat(phase_feats) → GAP → MLP → [γ, β]
        modulator_in = dim * num_phases
        self.modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(modulator_in, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim * 2),
        )
        # 初始化确保 FiLM 从恒等变换开始: γ≈1, β≈0
        nn.init.zeros_(self.modulator[-1].weight)
        with torch.no_grad():
            self.modulator[-1].bias[:dim] = 1.0
            self.modulator[-1].bias[dim:] = 0.0

        # Gate: 自适应控制调制强度
        gate_dim = dim * (num_phases + 1)
        gate_final = nn.Linear(dim // 4, 1)
        nn.init.constant_(gate_final.bias, 1.0)  # sigmoid(1)≈0.73，初始偏开启，保证FiLM路径有足够梯度
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(gate_dim, dim // 4),
            nn.GELU(),
            gate_final,
            nn.Sigmoid(),
        )

    def forward(self, spatial_feat: torch.Tensor,
                *phase_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_feat: [B, C, H, W]  RGB 骨干特征
            phase_feats:   [B, C, H, W]  运动相位特征 (1个或2个)
        Returns:
            [B, C, H, W]  FiLM 调制融合特征
        """
        B, C, H, W = spatial_feat.shape

        # 拼接所有相位特征 → 生成 γ, β
        phase_concat = torch.cat(phase_feats, dim=1)  # [B, P*C, H, W]
        params = self.modulator(phase_concat)           # [B, 2C]
        gamma, beta = params.chunk(2, dim=1)            # [B, C] each
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        # FiLM 调制: out = γ ⊙ rgb + β
        modulated = gamma * spatial_feat + beta

        # Gate 控制调制强度的接入比例
        all_feats = torch.cat([spatial_feat] + list(phase_feats), dim=1)
        gate_val = self.gate(all_feats).view(B, 1, 1, 1)

        return spatial_feat + gate_val * modulated


# ------------------------------------------------------------------
#  GatedSumFusion — 保留空间内容的门控求和融合 (推荐替代 FiLM)
# ------------------------------------------------------------------

class GatedSumFusion(nn.Module):
    """
    门控求和融合：所有模态空间特征等权保留，经 per-sample 可学习门控加权求和。

    为何替代 FiLM（根因）:
      FiLM 以 RGB 为 carrier、flow 仅作通道调制:
        out = f_rgb + gate·(γ(flow)⊙f_rgb + β(flow))
      flow 的 *空间* 运动结构被 modulator 的 GAP 压成 per-channel 标量，融合特征
      实质是「被光流通道重标定的 RGB」——flow 这一最强信号被丢弃，导致
      rgb_dual_flow 比纯 flow 暴跌（≈退化为弱 RGB 表征），且 FiLM 不是 flow 的
      超集，dual_flow 永远追不上 flow。

    本模块直接对所有模态空间特征加权求和:
        fused = Σ_i  w_i · f_i ,  w = softmax(GateNet(cat(f_i)))
      - flow 的空间内容永不丢失 → 是 *纯 flow 的超集*（网络把 flow 权重学到 1、
        其余到 0 即可无损恢复 flow 性能），故 dual_flow ≥ flow；
      - apex RGB（f_rgb 即 apex 帧）作为互补空间信号加入，提供外观/形变线索；
      - per-sample 门控使不同数据集自适应（CASME II 偏 flow，SMIC 偏 apex RGB）；
      - softmax + 零初始化 → 初期各模态等权平均，起点即包含 flow，训练稳定。

    Args:
        dim: 特征通道数
        num_feats: 输入模态数 (rgb_flow=2: rgb+flow; rgb_dual_flow=3: rgb+oa+ao)
        reduction: 门控 MLP 瓶颈缩减比
    """

    def __init__(self, dim: int, num_feats: int, reduction: int = 4,
                 init_flow_bias: float = 0.0):
        super().__init__()
        self.num_feats = num_feats
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim * num_feats, dim * num_feats // reduction),
            nn.GELU(),
            nn.Linear(dim * num_feats // reduction, num_feats),
        )
        # 零初始化 → logits=0 → softmax 均匀，初期即各模态等权平均（已含 flow）
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)
        # v0.8.1 诊断：feats 顺序为 (rgb, flow_oa[, flow_ao])，index>=1 为光流。
        # init_flow_bias>0 → 初期 softmax 权重偏向 flow，使 dual_flow 起点≈纯 flow
        # （真正的「flow 超集」起点），再逐步学 RGB apex 互补，避免初期三模态均匀平均
        # 被 RGB 稀释导致的多数类捷径。默认 0.0 = 均匀（保持 v0.8 行为）。
        if init_flow_bias != 0.0 and num_feats >= 2:
            with torch.no_grad():
                for i in range(1, num_feats):
                    self.gate_net[-1].bias[i] = init_flow_bias

    def forward(self, *feats: torch.Tensor) -> torch.Tensor:
        # feats: (f_rgb, f_flow_oa[, f_flow_ao])，顺序一致即可
        gate_in = torch.cat(feats, dim=1)                  # [B, num_feats*C, H, W]
        w = torch.softmax(self.gate_net(gate_in), dim=1)   # [B, num_feats]
        w = w.view(-1, self.num_feats, 1, 1, 1)
        fused = sum(w[:, i] * feats[i] for i in range(self.num_feats))
        return fused


# ===================================================================
#  BPTNet v0.6
# ===================================================================

class BPTNet(nn.Module):
    """
    BPTNet: Phase-Aware Temporal Attention Network

    ResNet18 (预训练) → CBAM(按需) → [PhaseCrossAttention] → Post-Norm TransformerBlock → Head

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
        # ---- SSL 权重注入（仅 flow 单模态）----
        # 该 checkpoint 为可选的领域自监督（SSL）权重；不提供时使用 ImageNet 初始化（论文报告的数字均为 ImageNet 初始化）
        # （2972 个光流样本；输入 [u,v,u] + 与训练完全相同的归一化）。
        # 只对 flow 单模态生效，避免误注入到 RGB 分支（两者分布完全不同）。
        flow_ssl_ckpt = model_config.get('flow_ssl_ckpt', None) if ft == 'flow' else None

        if backbone_type == 'resnet18':
            backbone, out_channels = self._build_resnet18(pretrained, flow_ssl_ckpt)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        self.backbone = backbone
        self.backbone_out_channels = out_channels

        # ---- 模态独立骨干（手稿 §2.5: "RGB appearance and optical flow come from
        #      two independent backbones and reside in heterogeneous embedding spaces"）
        # 实测根因：RGB（自然图像，ImageNet 统计）与光流（帧差图，分布迥异）若共享
        # 同一个 ResNet18，两者的 BatchNorm running statistics 会互相污染——即使 RGB
        # 支路的输出被丢弃，其前向仍会更新 BN 统计，使光流特征被破坏，多模态分支
        # 类别坍缩（UAR 恒等于 1/num_classes，不同 fold 逐位相同）。
        # 分离骨干后两者统计彻底隔离，同时与手稿"两个独立骨干"的描述一致。
        self.separate_backbone = model_config.get('separate_backbone', False)
        # 光流 → 3 通道的构造方式（见 _to_rgb_input）: "dup"(历史) / "uvmag"
        self.flow_3ch = model_config.get('flow_3ch', 'dup')
        self.backbone_rgb = None
        self.proj_rgb = None
        if self.separate_backbone and self.is_multimodal:
            self.backbone_rgb, _ = self._build_resnet18(pretrained)

        # 投影到 transformer 维度
        if out_channels != transformer_dim:
            self.proj = nn.Conv2d(out_channels, transformer_dim, 1)
            if self.backbone_rgb is not None:
                self.proj_rgb = nn.Conv2d(out_channels, transformer_dim, 1)
        else:
            self.proj = nn.Identity()
            if self.backbone_rgb is not None:
                self.proj_rgb = nn.Identity()

        # ==================== CBAM 注意力 ====================
        fmap_size = image_size // 32
        cbam_ks = 7 if fmap_size >= 7 else 3

        self.cbam = None
        self.cbam_flow = None
        self.cbam_flow_ao = None
        if use_cbam:
            self.cbam = CBAM(transformer_dim, kernel_size=cbam_ks)
            if self.is_multimodal:
                # flow 模态: 默认关闭通道注意力，避免破坏光流运动模式。
                # v0.8.4 确认实验开关 flow_use_channel_attn=True 时开启通道注意力，
                # 以把 motion_only 嫁接回 flow 基线的『完整 CBAM』强管线（见 run_tune tier mo_strong）。
                flow_use_channel = model_config.get('flow_use_channel_attn', False)
                self.cbam_flow = CBAM(transformer_dim, kernel_size=cbam_ks,
                                      use_channel_attn=flow_use_channel)
                if self.is_dual_flow:
                    self.cbam_flow_ao = CBAM(transformer_dim, kernel_size=cbam_ks,
                                             use_channel_attn=flow_use_channel)

        # ==================== 多模态融合 ====================
        self.phase_fusion = None
        self.modality_norm = None
        # v0.8.1 诊断开关：融合前 per-location LayerNorm 是否启用（默认 True，保持 v0.8 行为）。
        # 关闭用于验证假设「modality_norm 把光流运动幅度在每个位置归一化掉 →
        # flow 判别信号被削弱 → train/test 双双类别坍缩」。
        self.use_modality_norm = model_config.get('use_modality_norm', True)
        if self.is_multimodal:
            # 模态归一化（融合前置，修复 RGB/光流 特征尺度不兼容导致的类别坍缩）：
            # RGB 自然图像 与 光流(2ch→3ch) 经同一 ResNet18 后激活幅度/分布差异巨大，
            # 直接求和/加权（或 FiLM 以 RGB 为 carrier）会让主导模态淹没其余、引发
            # 类别坍缩（见 3c/5c dual_flow 混淆矩阵：几乎只预测多数类）。融合前对每个
            # 模态特征图按 (H,W) 位置做 per-location LayerNorm（同 token 归一化），
            # 使各模态同尺度可比；flow 空间内容完整保留 → dual_flow 是 flow 的超集。
            # 单模态路径（apex/flow）不归一化，保持原有性能。
            if self.use_modality_norm:
                self.modality_norm = nn.LayerNorm(transformer_dim)
            num_phases = 2 if self.is_dual_flow else 1
            if phase_fusion == 'cross_attention':
                self.phase_fusion = PhaseCrossAttention(
                    dim=transformer_dim,
                    num_phases=num_phases,
                    num_heads=min(transformer_heads, 4),
                    dropout=dropout,
                )
            elif phase_fusion == 'film':
                self.phase_fusion = FiLMFusion(
                    dim=transformer_dim,
                    num_phases=num_phases,
                )
            elif phase_fusion == 'gated_sum':
                # 推荐融合：保留 flow 空间内容（纯 flow 超集），apex RGB 互补加入
                self.phase_fusion = GatedSumFusion(
                    dim=transformer_dim,
                    num_feats=num_phases + 1,
                    init_flow_bias=model_config.get('gated_sum_flow_bias', 0.0),
                )
                # v0.8.3 决胜隔离：仅当 phase_feats_source='motion_only' 启用。
                # 纯运动双流融合（零 RGB）：对双向光流 f_oa/f_ao 做门控求和，回归 dual 的
                # "双向相位"思想但彻底排除弱/噪声 RGB 模态。这是 dual_flow 设计最后一块
                # 未被干净检验的资产（"双向运动互补性"）——若 ≥ flow 基线→dual 思想可 salvage
                # 为 motion-only；若仍 < flow→dual 架构无益，应彻底回归单流 flow。
                if model_config.get('phase_feats_source') == 'motion_only':
                    self.motion_fusion = GatedSumFusion(
                        dim=transformer_dim, num_feats=2, init_flow_bias=0.0)
            # phase_fusion == 'sum' → self.phase_fusion = None, 回退简单求和

        # ==================== Position Embedding ====================
        self.pos_embed = nn.Parameter(
            torch.randn(1, transformer_dim, fmap_size, fmap_size) * 0.02
        )

        # ==================== CLS Token ====================
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim) * 0.02)

        # ==================== Phase-Token 位置编码 ====================
        # 每个相位（最多 3：onset/apex/offset 或 apex/oa/ao）一个可学习位置向量，
        # 注入相位时序顺序，使 CLS 经自注意力显式跨相位建模。
        self.phase_pos_embed = nn.Parameter(
            torch.randn(1, 3, transformer_dim) * 0.02
        )
        self.use_phase_tokens = model_config.get('use_phase_tokens', True)
        # 相位 token 来源（v0.8.2 诊断）：默认 'all'，dual_flow 向 Transformer 注入
        # [f_rgb, f_oa, f_ao] 三个相位 token；纯 flow 仅 [f_flow] 一个。diag 已证伪
        # "尺度不兼容 / 融合起点被 RGB 稀释"两假设（gs_flowbias flow 权重≈0.98 仍塌），
        # 怀疑 RGB 作为弱/噪声模态的独立可学习 token 给网络提供了"压多数类"的退化捷径。
        #   'no_rgb' : dual/rgb_flow 丢弃 RGB 相位 token（dual=[f_oa,f_ao]），保留运动相位结构
        #   'fused'  : 仅用融合特征作单一相位 token（dual=[fused]），Transformer 输入对齐纯 flow
        # 若 no_rgb / fused 回升到 flow 水平 → 坐实 RGB token 为元凶（纯架构问题）。
        self.phase_feats_source = model_config.get('phase_feats_source', 'all')

        # ==================== Transformer (Post-Norm + 正确 DropPath) ====================
        # 手稿 §3.6.2: "layer-independent probability p_drop = 0.1" → uniform
        # 历史实现为按层线性递增 0→p（第 1 层无 dropout），保留为对照开关
        drop_path_mode = model_config.get('drop_path_mode', 'uniform')
        if drop_path_mode == 'uniform':
            drop_rates = [drop_path_rate] * transformer_layers
        else:
            drop_rates = [drop_path_rate * i / max(transformer_layers - 1, 1)
                          for i in range(transformer_layers)]

        self.transformer_blocks = nn.ModuleList()
        for i in range(transformer_layers):
            block = TransformerBlock(
                dim=transformer_dim,
                heads=transformer_heads,
                dropout=dropout,
                drop_path_rate=drop_rates[i],
                ff_mult=4,
            )
            self.transformer_blocks.append(block)

        # ==================== 分类头 ====================
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Linear(transformer_dim, num_classes)

        # 初始化非预训练权重
        self._init_weights()

        # ==================== 骨干冻结（小样本抗过拟合）====================
        # 背景：CASME II 3c 严格 LOSO 下训练样本仅 ~120-230 个，而 ResNet18 +
        #   Transformer 的可训练参数达 30-42M。同分布（同被试内随机划分样本）诊断
        #   显示 val UF1 在第 20 轮见顶后持续下滑、train_acc 仍升至 0.77 —— 典型
        #   过拟合；把 epoch 从 40 加到 150 反而更差。
        # 冻结 ImageNet 预训练骨干后，可训练自由度主要落在 Transformer 与分类头，
        #   是 MER 小样本的常见做法。
        # 关键细节：冻结时骨干必须保持 eval()。否则 BatchNorm 的 running stats 仍会
        #   被更新，等于"权重冻结了、特征分布却仍在漂移"，冻结效果大打折扣。
        #   故下方重写 train()。
        self.freeze_backbone = model_config.get('freeze_backbone', False)
        self._frozen_tensors = 0
        if self.freeze_backbone:
            for module in (self.backbone, self.backbone_rgb):
                if module is None:
                    continue
                for p in module.parameters():
                    p.requires_grad = False
                    self._frozen_tensors += 1
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_all = sum(p.numel() for p in self.parameters())
            logger.info(
                f"骨干已冻结: {self._frozen_tensors} 个参数张量 / 可训练参数 "
                f"{n_train / 1e6:.2f}M (总计 {n_all / 1e6:.2f}M)，BN 保持 eval()"
            )

    def train(self, mode: bool = True):
        """重写 train()：骨干冻结时强制其保持 eval()，避免 BN 统计漂移。"""
        super().train(mode)
        if getattr(self, 'freeze_backbone', False):
            for module in (self.backbone, self.backbone_rgb):
                if module is not None:
                    module.eval()
        return self

    # ------------------------------------------------------------------
    #  构建方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_resnet18(pretrained: bool, ckpt_path: str = None):
        """构建 ResNet-18 骨干（到 layer4）。

        ckpt_path（可选）: SSL 预训练权重，用于覆盖 ImageNet 初始化。
          来源 = 可选的领域自监督（SSL）checkpoint；不提供时使用 ImageNet 初始化
          state_dict；输入表示与训练严格一致（[u,v,u] 3 通道 + 同一归一化），
          因此可原样注入、无需改结构。
          键名前缀兼容 'encoder_q.' / 'encoder_k.' / 'module.' / 'backbone.'；
          fc 层不匹配会被忽略（strict=False）。
        """
        import torchvision.models as models
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        if ckpt_path:
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict):
                # 兼容各种包装键：MoCo 预训练产物(state_dict) / 训练检查点(model_state_dict) 等
                for wrap in ("state_dict", "model_state_dict", "model", "net", "weights"):
                    if wrap in sd and isinstance(sd[wrap], dict):
                        sd = sd[wrap]
                        break
            fixed = {}
            for k, v in sd.items():
                kk = k
                for pre in ("module.", "encoder_q.", "encoder_k.", "backbone."):
                    if kk.startswith(pre):
                        kk = kk[len(pre):]
                fixed[kk] = v
            missing, unexpected = resnet.load_state_dict(fixed, strict=False)
            n_loaded = len(fixed) - len(unexpected)
            logger.info(f"[SSL] 注入光流分支 SSL 权重: {ckpt_path} "
                        f"(匹配 {n_loaded} 个张量, 未使用 {len(unexpected)}, 缺失 {len(missing)})")
            if n_loaded < 50:
                logger.warning("[SSL] 权重匹配数过少，可能没生效，请检查键名/文件是否正确！")

        # 移除 avgpool + fc，保留到 layer4
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        return backbone, 512

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'backbone' in name:
                continue  # 预训练权重不动
            if 'phase_fusion' in name:
                continue  # FiLM/CrossAttn 自行管理初始化
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # CLS token + pos embed 特殊处理
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, 'pos_embed'):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, 'phase_pos_embed'):
            nn.init.trunc_normal_(self.phase_pos_embed, std=0.02)

    # ------------------------------------------------------------------
    #  输入转换 & 特征提取
    # ------------------------------------------------------------------

    def _to_rgb_input(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c == 3:
            return x
        elif c == 2:
            # 2 通道光流 → 3 通道，以复用 ImageNet 预训练骨干
            #   "dup"  (历史实现): [u, v, u] —— 第 3 通道是 u 的复制，几乎不增信息
            #   "uvmag"          : [u, v, |flow|] —— 显式给出运动强度；幅值本身对
            #                      微表情判别很关键（第一层卷积无法自行算出模长）
            if getattr(self, "flow_3ch", "dup") == "uvmag":
                mag = torch.sqrt(x[:, :1] ** 2 + x[:, 1:2] ** 2 + 1e-8)
                return torch.cat([x, mag], dim=1)
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

    def _forward_transformer(self, spatial_feat: torch.Tensor,
                              phase_feats: list) -> torch.Tensor:
        B, C, H, W = spatial_feat.shape

        # 自适应位置编码插值
        if H != self.pos_embed.shape[2] or W != self.pos_embed.shape[3]:
            pe = nn.functional.interpolate(
                self.pos_embed, size=(H, W), mode='bilinear', align_corners=False
            )
        else:
            pe = self.pos_embed

        # 空间 patch tokens
        patches = spatial_feat.flatten(2).transpose(1, 2)   # [B, N, C]
        pe = pe.flatten(2).transpose(1, 2)                  # [B, N, C]
        patches = patches + pe

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # ---- Phase-Token Temporal Attention ----
        # 每个相位（onset/apex/offset 或 apex/oa/ao）聚合为全局 token，
        # 加可学习相位位置编码，CLS 经 Transformer 自注意力显式跨相位时序建模。
        if self.use_phase_tokens and phase_feats:
            P = len(phase_feats)
            phase_tok = torch.stack(
                [p.mean(dim=(-1, -2)) for p in phase_feats], dim=1
            )                                               # [B, P, C]
            phase_tok = phase_tok + self.phase_pos_embed[:, :P, :]
            tokens = torch.cat([cls_tokens, phase_tok, patches], dim=1)
        else:
            # 回退：纯空间注意力（v0.6 行为，供消融对照）
            tokens = torch.cat([cls_tokens, patches], dim=1)

        # Transformer blocks (Post-Norm + DropPath on each residual branch)
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # CLS → LayerNorm → Head
        out = self.norm(tokens[:, 0])
        return self.head(out)

    # ------------------------------------------------------------------
    #  Main Forward
    # ------------------------------------------------------------------

    def forward(self, img):
        """
        前向传播，自动适配所有 frame_type。

        返回：
            [B, num_classes] logits

        相位 token 来源（用于 Phase-Token Temporal Attention）：
          - apex:           [f_apex]
          - flow:           [f_flow]
          - rgb_triplet:    [f_onset, f_apex, f_offset]   ← 三帧各自特征（不再平均丢弃）
          - rgb_flow:       [f_apex, f_oa]
          - rgb_dual_flow:  [f_apex, f_oa, f_ao]
        """
        if isinstance(img, (list, tuple)):
            # ======== 多模态 ========
            # 提取顺序：backbone+proj → per-location LayerNorm（对齐 RGB/光流尺度）
            #   → CBAM（在归一化之后，效果不被后续归一化抵消）→ 融合。
            if self.is_dual_flow:
                rgb, flow_oa, flow_ao = img
                f_oa  = self._backbone_proj(flow_oa)
                f_ao  = self._backbone_proj(flow_ao)
                f_oa  = self._norm_feature(f_oa)
                f_ao  = self._norm_feature(f_ao)
                if self.cbam_flow is not None:
                    f_oa = self.cbam_flow(f_oa)
                if self.cbam_flow_ao is not None:
                    f_ao = self.cbam_flow_ao(f_ao)
                if self.phase_feats_source == 'motion_only':
                    # 纯运动双流（双向光流，零 RGB）：保留 dual 的"双向相位"思想，但完全
                    # 排除弱/噪声 RGB 模态。这是 dual_flow 设计最后一块干净资产 ——
                    # 若 ≥ flow 基线 → dual 可 salvage 为 motion-only；若仍 < flow → dual 无益。
                    spatial_feat = self.motion_fusion(f_oa, f_ao)
                    phase_feats = [f_oa, f_ao]
                else:
                    f_rgb = self._backbone_proj(rgb, stream='rgb')
                    f_rgb = self._norm_feature(f_rgb)
                    if self.cbam is not None:
                        f_rgb = self.cbam(f_rgb)
                    spatial_feat = self._fuse_multimodal(f_rgb, f_oa, f_ao)
                    if self.phase_feats_source == 'flow_only':
                        # 决胜隔离：dual 架构但空间特征=纯 flow（完全不参与任何 RGB 融合），
                        # 相位 token 也只用 flow。若此配置回升到 flow 基线 → 坐实『RGB 进入融合后的
                        # 空间特征（可学习 gate 漂移到压多数类捷径）』是唯一元凶，修复=flow 锚定融合。
                        spatial_feat = f_oa
                        phase_feats = [f_oa]
                    elif self.phase_feats_source == 'no_rgb':
                        phase_feats = [f_oa, f_ao]      # 丢弃 RGB 相位 token，保留双向流相位结构
                    elif self.phase_feats_source == 'fused':
                        phase_feats = [spatial_feat]    # 单一融合 token，Transformer 输入对齐纯 flow
                    else:
                        phase_feats = [f_rgb, f_oa, f_ao]   # 默认：含 RGB 相位 token
            else:
                # rgb_flow
                rgb, flow = img
                f_rgb  = self._backbone_proj(rgb, stream='rgb')
                f_flow = self._backbone_proj(flow)
                f_rgb  = self._norm_feature(f_rgb)
                f_flow = self._norm_feature(f_flow)
                if self.cbam is not None:
                    f_rgb = self.cbam(f_rgb)
                if self.cbam_flow is not None:
                    f_flow = self.cbam_flow(f_flow)
                spatial_feat = self._fuse_multimodal(f_rgb, f_flow)
                if self.phase_feats_source == 'flow_only':
                    spatial_feat = f_flow               # 决胜隔离：空间特征=纯 flow，排除 RGB 融合
                    phase_feats = [f_flow]
                elif self.phase_feats_source == 'no_rgb':
                    phase_feats = [f_flow]              # 丢弃 RGB 相位 token
                elif self.phase_feats_source == 'fused':
                    phase_feats = [spatial_feat]        # 单一融合 token，对齐纯 flow
                else:
                    phase_feats = [f_rgb, f_flow]       # 默认：含 RGB 相位 token
        else:
            # ======== 单模态 ========
            if self.is_triplet:
                B, _, H, W = img.shape
                feats = self._extract_single(
                    img.reshape(B * 3, 3, H, W).contiguous(), self.cbam
                )                                               # [B*3, C, Hf, Wf]
                C = feats.shape[1]
                feats = feats.reshape(B, 3, C, feats.shape[2], feats.shape[3])
                spatial_feat = feats.mean(dim=1)                # 局部空间细节：三帧平均
                phase_feats = [feats[:, i] for i in range(3)]   # 三相位 token 源
            elif self.is_flow_only:
                spatial_feat = self._extract_single(img, self.cbam)
                phase_feats = [spatial_feat]
            else:
                # apex 单帧 RGB: 关闭 CBAM，保留预训练特征纯度
                spatial_feat = self._extract_single(img, None)
                phase_feats = [spatial_feat]

        return self._forward_transformer(spatial_feat, phase_feats)

    # ------------------------------------------------------------------
    #  单图特征提取（供 forward 复用，区分 triplet 多帧）
    # ------------------------------------------------------------------

    def _backbone_proj(self, x: torch.Tensor, stream: str = 'flow') -> torch.Tensor:
        """仅 backbone + 投影（不含 CBAM / 归一化），供多模态分支分步调用。

        Args:
            stream: 'flow' → 使用 self.backbone；
                    'rgb'  → 当 separate_backbone=True 时使用独立骨干
                             self.backbone_rgb（避免与光流共享 BN 统计），
                             否则回落到共享骨干。
        """
        x = self._to_rgb_input(x)
        if stream == 'rgb' and self.backbone_rgb is not None:
            feat = self.backbone_rgb(x)
            feat = self.proj_rgb(feat)
            return feat
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat

    def _extract_single(self, x: torch.Tensor,
                        apply_cbam: nn.Module = None) -> torch.Tensor:
        feat = self._backbone_proj(x)
        if apply_cbam is not None:
            feat = apply_cbam(feat)
        return feat

    def extract_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """
        复用 flow 单模态强骨干，返回 Transformer 编码后的 token 序列
        （Head 之前），供外部相位引导模块（PSGM）叠加。

        返回：[B, N+1+P, D]，其中 N=空间 patch 数, 1=CLS, P=相位 token 数
              (flow 单模态 P=1，即 f_flow)。
        注意：与 forward 的 flow 分支完全一致，只是不接 Head。
        """
        if img.shape[1] == 2:
            img = self._to_rgb_input(img)
        if self.frozen:
            with torch.no_grad():
                spatial_feat = self._extract_single(img, self.cbam)
        else:
            spatial_feat = self._extract_single(img, self.cbam)
        phase_feats = [spatial_feat]
        tokens = self.backbone(spatial_feat)
        tokens = self.proj(tokens)
        # 复用 _forward_transformer 到 Head 之前：手工展开其 token 构建逻辑
        B, C, H, W = tokens.shape
        if H != self.pos_embed.shape[2] or W != self.pos_embed.shape[3]:
            pe = nn.functional.interpolate(
                self.pos_embed, size=(H, W), mode='bilinear',
                align_corners=False)
        else:
            pe = self.pos_embed
        patches = tokens.flatten(2).transpose(1, 2)
        pe = pe.flatten(2).transpose(1, 2)
        patches = patches + pe
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.use_phase_tokens and phase_feats:
            P = len(phase_feats)
            phase_tok = torch.stack(
                [p.mean(dim=(-1, -2)) for p in phase_feats], dim=1)
            phase_tok = phase_tok + self.phase_pos_embed[:, :P, :]
            seq = torch.cat([cls_tokens, phase_tok, patches], dim=1)
        else:
            seq = torch.cat([cls_tokens, patches], dim=1)
        # 不经 Head，直接返回 token 序列供 PSGM 使用
        return seq

    def _norm_feature(self, x: torch.Tensor) -> torch.Tensor:
        """per-location LayerNorm over channel C：
        [B,C,H,W] → [B,H,W,C] → LayerNorm(C) → [B,C,H,W]。
        使 RGB / 光流 各模态特征在同一尺度可比，避免融合时主导模态淹没其余。
        modality_norm 为 None（诊断关闭）时直接透传，不改变特征。"""
        if self.modality_norm is None:
            return x
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.modality_norm(x)
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
