"""
MPFNet (参考复现 · 简化监督版)
================================

⚠️ 重要说明（决定本文件的诚实边界）
-------------------------------
MPFNet (Chuang Ma et al., TAFFC 2025, arXiv 2506.09735) 原文为基于度量的
**元学习 (metric-based meta-learning)** 框架：support/query 集、N-way K-shot
episode、类质心余弦相似度 + 交叉熵分类；且依赖**未公开**的预处理
(VFI 帧插值 / FlowNet2.0 光流 / EVM 运动放大 / 阿里云人脸对齐)。
论文**未提供官方代码链接**（已核实 arXiv/IEEE 页均无仓库 URL）。

→ 本文依论文描述做**简化监督近似 (simplified supervised approximation)**，
  而非逐点忠实复现：

    · 仅实现 CA-I3D 主干 (3D 卷积 + Coordinate Attention)，作为**单编码器**
      监督分类，直接接入本框架现有 trainer.py (标准 CrossEntropy / Focal)。
    · 原文输入为 ℱ∈R^{128×128×5×10}（2 光流 + 3 帧差，由 VFI 生成的 11 帧
      序列构造的 10 个帧对），该预处理未释放。
    · 本近似改用本框架 `rgb_triplet`（onset+apex+offset 三帧 RGB，9 通道）
      重塑为 [B, C=3, T=3, H, W] 作为 3D 主干输入。**忠实性有限，须披露。**

论文报数（引用对照，须标协议）:
    SDE Acc 0.811 / 0.924 / 0.857 (SMIC / CASME II / SAMM, MPFNet-C)
    CDE  UF1 0.840 (MEGC2019-CD)
  └─ 与本近似版**非同协议**，不可直接比较；原 SOTA 声明**尚待独立验证**。

架构 (CA-I3D 主干):
  输入 [B, 9, H, W] (rgb_triplet)
    → reshape [B, 3, 3, H, W]  (C, T)
    → Conv3d(3, 64, 3, p=1) → BN → ReLU
    → MaxPool3d((1,3,3), stride (1,2,2))
    → Inception3D + CoordinateAttention3D  ×2
    → GlobalAvgPool(T,H,W) → [B, D]
    → Dropout → Linear(D, num_classes)

版本: v0.1 (简化监督近似, 直接接入 trainer.py)
"""

import torch
import torch.nn as nn


# ===================================================================
#  Coordinate Attention 3D (3D 坐标注意力)
# ===================================================================

class CoordinateAttention3D(nn.Module):
    """
    3D 坐标注意力，将 2D Coordinate Attention (Hou et al., CVPR 2021) 推广到
    时空体 (T, H, W) 三轴。

    核心思想（与 2D 版一致）：
      沿每个时空轴，对"其余两轴"做平均池化得到 [B, C, L, 1, 1]，
      再用 1D 卷积（kernel=3）编码该轴的位置上下文，sigmoid 得到注意力掩码，
      与原特征逐元素相乘。三轴掩码顺序相乘。

    相比 2D 版：新增 T 轴分支；2D 版只编码 H、W 两轴。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        inner = max(8, channels // reduction)
        # 三个轴 (T, H, W) 各用一对 1D 卷积 (降维 → 升维)
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, inner, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(inner),
                nn.ReLU(inplace=True),
                nn.Conv1d(inner, channels, kernel_size=3, padding=1, bias=False),
            )
            for _ in range(3)
        ])
        self.sigmoid = nn.Sigmoid()

    def _axis(self, x: torch.Tensor, axis: int, conv: nn.Module) -> torch.Tensor:
        # other = 除 axis 外的两个时空轴
        other = tuple(d for d in (2, 3, 4) if d != axis)
        # 对 other 两轴平均池化 -> [B, C, L, 1, 1] (L 位于 axis 位)
        pooled = x.mean(dim=other, keepdim=True)
        # 把目标轴 L 移到最后一位，以便 Conv1d 处理
        perm = [0, 1] + [d for d in (2, 3, 4) if d != axis] + [axis]  # -> [B, C, 1, 1, L]
        t = pooled.permute(perm).squeeze(2).squeeze(2)               # -> [B, C, L]
        # 1D 卷积编码位置上下文 -> sigmoid
        t = self.sigmoid(conv(t))                                    # -> [B, C, L]
        # 还原回 [B, C, 1, 1, L]
        t = t.unsqueeze(2).unsqueeze(2)
        # 逆置换回原轴序 [B, C, L, 1, 1]
        inv = [0, 0, 0, 0, 0]
        for i, d in enumerate(perm):
            inv[d] = i
        t = t.permute(inv)
        # 广播回 [B, C, T, H, W]
        mask = t.expand_as(x)
        return x * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, axis in enumerate((2, 3, 4)):  # T, H, W
            x = self._axis(x, axis, self.fcs[i])
        return x


# ===================================================================
#  Inception3D (3D Inception v1 风格块)
# ===================================================================

class Inception3D(nn.Module):
    """
    3D Inception v1 风格块：四条分支在通道维拼接。
      branch1: 1×1×1
      branch2: 1×1×1 → 3×3×3
      branch3: 1×1×1 → 5×5×5
      branch4: 3×3×3 maxpool(stride 1) → 1×1×1
    所有卷积分支使用 padding 以保持时空尺寸不变。
    """

    def __init__(self, in_ch: int,
                 ch1x1: int,
                 ch3x3_red: int, ch3x3: int,
                 ch5x5_red: int, ch5x5: int,
                 pool_ch: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_ch, ch1x1, kernel_size=1, bias=False),
            nn.BatchNorm3d(ch1x1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_ch, ch3x3_red, kernel_size=1, bias=False),
            nn.BatchNorm3d(ch3x3_red),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch3x3_red, ch3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ch3x3),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_ch, ch5x5_red, kernel_size=1, bias=False),
            nn.BatchNorm3d(ch5x5_red),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch5x5_red, ch5x5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm3d(ch5x5),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_ch, pool_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(pool_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


# ===================================================================
#  MPFNet (简化监督版)
# ===================================================================

class MPFNet(nn.Module):
    """
    MPFNet 简化监督近似：CA-I3D 3D 主干 + 3D 坐标注意力，单编码器监督分类。

    输入约定：依赖 data.frame_type == 'rgb_triplet'，即单张张量
      [B, 9, H, W]（onset+apex+offset 三帧 RGB 通道堆叠）。
      在 forward 中重塑为 [B, 3, 3, H, W]（C=3, T=3）作为 3D 主干输入，
      以替代论文未释放的 5 通道 × 10 帧对张量 ℱ。

    Args:
        config: 配置字典 (与 trainer.py 一致)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        data_config = config.get('data', {})

        self.image_size = data_config.get('image_size', 224)
        self.num_classes = data_config.get('num_classes', 3)

        # ---- 架构超参 (可由 config['model.mpfnet'] 覆盖) ----
        mpf_cfg = model_config.get('mpfnet', {})
        self.dropout = mpf_cfg.get('dropout', 0.5)
        ca_reduction = mpf_cfg.get('ca_reduction', 16)
        stem_ch = mpf_cfg.get('stem_ch', 64)

        # ---- 1) Stem: 3D 卷积 ----
        self.stem = nn.Sequential(
            nn.Conv3d(3, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(stem_ch),
            nn.ReLU(inplace=True),
        )
        # 时空最大池化（仅下采样空间，保持 T 不变）
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        # ---- 2) 两个 Inception3D 块 + CoordinateAttention3D ----
        # 通道数（轻量，避免小数据集过拟合）
        inc1_out = 112   # 64 -> 32+48+16+16
        inc2_out = 160   # 112 -> 48+64+16+32
        self.incep1 = Inception3D(
            stem_ch,
            ch1x1=32, ch3x3_red=32, ch3x3=48,
            ch5x5_red=8, ch5x5=16, pool_ch=16,
        )
        self.ca1 = CoordinateAttention3D(inc1_out, reduction=ca_reduction)

        self.incep2 = Inception3D(
            inc1_out,
            ch1x1=48, ch3x3_red=48, ch3x3=64,
            ch5x5_red=16, ch5x5=16, pool_ch=32,
        )
        self.ca2 = CoordinateAttention3D(inc2_out, reduction=ca_reduction)

        self.feat_dim = inc2_out

        # ---- 3) 全局时空平均池化 + 分类头 ----
        self.avgpool = nn.AdaptiveAvgPool3d(1)   # [B, feat_dim, 1, 1, 1]
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(self.dropout)
        self.head = nn.Linear(self.feat_dim, self.num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    #  权重初始化 (3D 主干无预训练，xavier 初始化)
    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 9, H, W] (rgb_triplet) 单张量
        Returns:
            [B, num_classes] logits
        """
        B, C, H, W = x.shape
        # 9 通道 = 3 帧 × 3 RGB → 重塑为 (C=3, T=3)
        x = x.reshape(B, 3, 3, H, W)

        x = self.stem(x)
        x = self.pool(x)
        x = self.ca1(self.incep1(x))
        x = self.ca2(self.incep2(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.drop(x)
        return self.head(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
