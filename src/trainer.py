"""
训练器：负责模型的训练、测试和交叉验证
支持标准 CrossEntropyLoss 和 Focal Loss (含类别权重)
v2: 新增 LR Scheduler + Warmup + Gradient Clipping + Early Stopping
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import Evaluator
from src.utils.logger import logger,set_color
from tqdm import tqdm
import src.utils.tool as tool



class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    专为类别不平衡设计，聚焦于难分类样本。

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    参考: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Args:
        alpha: 类别权重 [num_classes] 或 float
        gamma: 聚焦参数，越大越关注难样本（默认2.0）
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, config, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha.to(config.get('train.device', 'cuda'))

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) 原始 logits
            targets: (N,) 类别索引 (long)
        Returns:
            scalar loss
        """
        # [!] p_t 必须由**未加权**的 CE 求得。原实现把逆频权重 alpha 一并塞进 ce_loss，
        #     于是同一组权重同时进入 (1-p_t)^γ 与 ce_loss 两处，对稀有类形成**双重且指数
        #     放大**的强调（实测相对权重被放大 ~10×；而少数类每折只有 16~20 个样本）→
        #     损失被个位数样本主导，正是"折间 UF1 大幅摆动、单 seed 正信号被多种子推翻"的成因。
        #     规范实现（Lin 2017 / kornia）：p_t 用未加权 CE，alpha_t 单独相乘。
        #     归一化仍用 .mean()：Σα_t 的样本均值恒为 1，量纲无偏。
        ce_raw = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_raw)           # 真实 p_t（未加权）
        if self.alpha is not None:
            alpha_t = self.alpha.to(ce_raw.device).gather(0, targets)
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_raw
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_raw

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def _compute_class_weights(dataset, num_classes: int) -> Optional[torch.Tensor]:
    """
    从数据集中统计类别分布，计算逆频率权重。

    weight[c] = total_samples / (num_classes * count[c])

    用于 CrossEntropyLoss 或 FocalLoss 的 alpha 参数。
    对于 count=0 的类别赋权重为 1.0.
    """
    from collections import Counter
    try:
        labels = [dataset[i][1] for i in range(len(dataset))]
    except Exception:
        # 某些 Dataset 不支持直接索引，尝试收集
        labels = []
        for _, target in dataset:
            labels.append(target.item() if hasattr(target, 'item') else int(target))

    counter = Counter(labels)
    total = len(labels)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        if counter.get(c, 0) > 0:
            weights[c] = total / (num_classes * counter[c])
    return weights


class Trainer:
    """
    训练器
    
    封装完整的训练、测试和评估流程
    
    Args:
        config (dict): 配置字典
    """

    def __init__(self, config: Dict, model):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config

        # 设置设备（支持字符串和字典格式）
        self.device = torch.device(config.get('train.device', 'cuda'))
        
        # 设置随机种子
        seed = config.get('base.seed', 1024)
        self._set_seed(seed)
        
        # 创建模型
        self.model = model.to(self.device)
        
        # 创建数据加载器（train / val / test 三级隔离）
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)

        # 折内验证集：
        #   - epoch 选择 / 早停 / LR 调度只允许在 val 上发生
        #   - 测试被试只在训练结束后评估一次
        #   - val_mode='none' 时 val_loader 即 test_loader（历史行为，正式结果禁用）
        self.val_mode = config.get('data.val_mode', 'inner_subject')
        # 只要 val_loader 与 test_loader 不是同一对象（即存在独立验证集），
        # 就用验证集做模型选择。用对象身份判断而非枚举 val_mode 名称，
        # 避免新增 val 模式（如 stratified）时漏配而导致退回"在测试集上选择"。
        self.use_val_for_selection = (self.val_loader is not self.test_loader)
        if self.use_val_for_selection:
            logger.info(
                f"折内验证集: {len(self.val_loader.dataset)} 样本"
                f"（与测试集 {len(self.test_loader.dataset)} 样本完全隔离）"
            )
        else:
            logger.warning(
                "val_mode != 'inner_subject'：best 将在测试被试上选取"
                "（乐观偏置），禁止用于正式结果"
            )
        
        # ---- 损失函数: CrossEntropyLoss 或 FocalLoss (含类别权重) ----
        num_classes = config.get('data.num_classes', 3)
        use_focal = config.get('data.use_focal_loss', False)
        use_class_weight = config.get('data.use_class_weight', use_focal)  # focal 默认启用

        if use_class_weight:
            # 从训练集统计类别分布，计算逆频率权重
            class_weights = _compute_class_weights(
                self.train_loader.dataset, num_classes
            )
            logger.info(f"类别权重 (inverse freq): {class_weights.tolist()}")
        else:
            class_weights = None

        if use_focal:
            focal_gamma = config.get('data.focal_gamma', 2.0)
            self.criterion = FocalLoss(
                config, alpha=class_weights, gamma=focal_gamma, reduction='mean'
            ).to(self.device)
            logger.info(f"使用 Focal Loss (gamma={focal_gamma}, class_weight={'Yes' if use_class_weight else 'No'})")
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None
            )
            logger.info(f"使用 CrossEntropyLoss (class_weight={'Yes' if use_class_weight else 'No'})")

        # 将 Trainer 的 criterion 注入模型内部（如 VITSRMCL 的 self.id_loss）
        # 确保 class_weight 和 FocalLoss 对模型内部的分类损失也生效
        if hasattr(self.model, 'id_loss'):
            self.model.id_loss = self.criterion
            logger.info("已将 Trainer criterion 注入到模型内部 id_loss")
        
        # 使用扁平化配置（如果存在）或直接访问
        learning_rate = config.get('train.learning_rate', 5e-5)
        weight_decay = config.get('train.weight_decay', 1e-4)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # ---- LR Scheduler: Warmup + Cosine Annealing ----
        self.scheduler = self._create_scheduler(
            config.get('train.warmup_epochs', 10),
            config.get('train.epochs', 200)
        )
        
        # 评估器
        self.evaluator = Evaluator(num_classes=num_classes)
        
        # 最佳模型追踪 — 改用 UF1 (MER 领域更合理)
        self.best_accuracy = 0.0
        self.best_uf1 = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        
        # Early stopping
        es_cfg = config.get('train.early_stopping', {})
        self.es_enabled = es_cfg.get('enabled', True)
        self.es_patience = es_cfg.get('patience', 30)
        self.es_counter = 0
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
        logger.info(f"模型参数量: {self.model.get_num_parameters():,}")
    
    def _create_scheduler(self, warmup_epochs: int, total_epochs: int):
        """
        创建学习率调度器。

        train.scheduler:
          - "ReduceLROnPlateau"（手稿 §3.6.4 声明）: 按验证指标平台期衰减，
            监控 val UF1（mode='max'）；
          - 其他/缺省: Warmup(线性) + CosineAnnealing（历史配置）。

        plateau 模式不使用 warmup（由 factor/patience 控制节奏）。
        """
        sched_cfg = str(self.config.get('train.scheduler', 'CosineAnnealing'))
        if 'plateau' in sched_cfg.lower():
            factor = self.config.get('train.plateau.factor', 0.5)
            patience = self.config.get('train.plateau.patience', 5)
            self.scheduler_on_plateau = True
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=factor,
                patience=patience, min_lr=1e-7,
            )
            logger.info(
                f"LR Scheduler: ReduceLROnPlateau(mode=max, factor={factor}, "
                f"patience={patience}, monitor=val UF1)"
            )
            return scheduler

        self.scheduler_on_plateau = False

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup: 0 → 1.0
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing: 1.0 → ~0
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"LR Scheduler: Warmup({warmup_epochs}ep) + CosineAnnealing({total_epochs}ep)")
        return scheduler

    def _scheduler_step(self, metrics: Dict[str, float]):
        """按调度器类型推进 LR（plateau 模式需要传入被监控指标）"""
        if getattr(self, 'scheduler_on_plateau', False):
            self.scheduler.step(metrics.get('UF1', 0.0))
        else:
            self.scheduler.step()
    
    def _set_seed(self, seed: int):
        """
        设置随机种子以确保可复现性
        
        Args:
            seed: 随机种子
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"随机种子设置为: {seed}")

    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch (支持 Mixup batch-level 增强)
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            dict: 训练指标
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        mixup_alpha = self.config.get('data.mixup.alpha', 0.0)

        # 使用 tqdm 包装 DataLoader
        iter_data = tqdm(self.train_loader, desc=f"Epoch {epoch} Train")
        for batch_idx, (data, target) in enumerate(iter_data):
            # 移动到设备（支持 tuple 输入，如 rgb_flow）
            if isinstance(data, (list, tuple)):
                data = [d.to(self.device) for d in data]
            else:
                data = data.to(self.device)
            target = target.to(self.device)

            # ---- Mixup 增强 (batch-level, 仅非 use_target 模型) ----
            do_mixup = (mixup_alpha > 0
                        and not self.config.get('model.use_target', False))
            if do_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(target.size(0), device=self.device)
                if isinstance(data, (list, tuple)):
                    data_mixed = [lam * d + (1 - lam) * d[idx] for d in data]
                else:
                    data_mixed = lam * data + (1 - lam) * data[idx]
                target_a, target_b = target, target[idx]
                data = data_mixed

            # 前向传播
            self.optimizer.zero_grad()
            if self.config.get('model.use_target'):
                output = self.model(data, target)
                # 支持两种返回格式：(logits, loss) 或 logits
                if isinstance(output, tuple):
                    logits, model_loss = output
                    loss = model_loss  # 使用模型返回的loss
                else:
                    logits = output
                    loss = self.criterion(logits, target)
            else:
                output = self.model(data)
                logits = output
                if do_mixup:
                    loss = (lam * self.criterion(logits, target_a) + (1 - lam) * self.criterion(logits, target_b))
                else:
                    loss = self.criterion(logits, target)
            
            # 反向传播
            loss.backward()
            
            # Gradient clipping (防止 Transformer 梯度爆炸)
            grad_clip = self.config.get('train.gradient_clip', 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            
            self.optimizer.step()
            
            # 统计 (mixup batch 以主标签 target_a 作参考统计近似准确率，
            # 保证训练曲线可观测；原实现对 mixup batch 完全跳过统计，
            # 导致 alpha>0 时 Train Acc 恒为 0)
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=1)
            ref = target_a if do_mixup else target
            correct += pred.eq(ref).sum().item()
            total += batch_size

            # 更新进度条显示
            usage = tool.get_gpu_usage(self.device)
            use_device = "Device: CPU" if usage == "CPU" else "GPU RAM: " + usage
            iter_data.set_postfix_str(set_color(use_device, "yellow"))
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = correct / max(total, 1) if total > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    @property
    def _tta_mode(self):
        """归纳式 TTA 模式。仅 'flip' 在此启用多视图概率平均；
        'bn_adapt'/'tent' 属转导式，走 adapt_on_test_subject 那条独立路径。"""
        return str(self.config.get('eval.tta.mode', 'none')).lower()

    @property
    def _adj_tau(self):
        """长尾 logit 调整强度 τ（0=关闭）。建议在 val 上从 {0, 0.25, 0.5, 0.75, 1.0} 选。"""
        return float(self.config.get('eval.logit_adjust.tau', 0.0))

    def _class_prior(self):
        """训练集类先验（仅用训练集标签，杜绝测试信息泄漏）。结果缓存。"""
        if getattr(self, "_prior_cache", "unset") != "unset":
            return self._prior_cache
        labels = []
        ds = getattr(self.train_loader, "dataset", None)
        K = int(self.config.get('data.num_classes', 3))
        # [!] 必须用 dataset 里**合并后**的标签（`ds.labels`）。
        #     这里必须用**合并后**的标签；若取 samples 里的原始标签（如 casme2_7c 为 0..5）
        #     → K=max+1=6，
        #     但模型输出只有 num_classes=3 维 →
        #     `RuntimeError: The size of tensor a (3) must match the size of tensor b (6)`，
        #     24 折全部被跳过、该特性自实现以来**从未真正跑通**。
        raw = getattr(ds, "labels", None)
        if raw is not None:
            labels = [int(x) for x in raw]
        else:
            for s in (getattr(ds, "samples", None) or []):
                try:
                    labels.append(int(s["label"]))
                except Exception:
                    pass
        n_all = len(labels)
        labels = [l for l in labels if 0 <= l < K]
        if not labels:
            logger.warning("[LogitAdj] 取不到可用的训练集标签，logit 调整已跳过")
            self._prior_cache = None
            return None
        if n_all != len(labels):
            logger.warning(f"[LogitAdj] 有 {n_all - len(labels)} 个标签落在 [0,{K}) 之外，"
                           f"已忽略（若数量很多说明读到的仍是合并前标签，请检查 ds.labels）")
        cnt = [0.0] * K
        for l in labels:
            cnt[l] += 1.0
        tot = sum(cnt)
        # 与论文一致：先验直接取频率；样本数为 0 的类用 1/tot 兜底避免 0^τ 出现 inf
        prior = torch.tensor([max(c, 1.0) / tot for c in cnt], dtype=torch.float32)
        logger.info(f"[LogitAdj] 训练集先验({K} 类): {[round(float(x), 4) for x in prior]} "
                    f"τ={self._adj_tau}")
        self._prior_cache = prior
        return prior

    def _tta_views(self):
        """返回 TTA 的视图变换列表（identity 必须在其中）。

        'flip'        : {原图, 水平翻转}                          → 2 视图
        'flip_scale'  : {原图, 水平翻转} × 尺度表（默认 3 尺度）   → 6 视图
        'flipv_scale' : {原图, 水平翻, 竖直翻, 双向翻} × 尺度表    → 12 视图
        'scale5'      : {原图, 水平翻转} × 5 尺度                 → 10 视图

        尺度表可用 `eval.tta.scales` 直接覆盖（例 [0.75, 0.875, 1.0, 1.125, 1.25]）。

        [!] 视图选择有物理约束，不是"加得越多越好"：
            水平翻转是训练时见过的对称（horizontal_flip），最安全；
            竖直翻转训练时**没见过**，属外推视图，有效性必须单独实测；
            缩放用于吸收库间人脸尺度/光流幅值差异。
        """
        def _id(t):
            return t

        def _scale_fn(s):
            def f(t):
                if not torch.is_tensor(t) or t.dim() != 4:
                    return t
                h, w = t.shape[-2:]
                nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
                y = torch.nn.functional.interpolate(
                    t, size=(nh, nw), mode='bilinear', align_corners=False)
                return torch.nn.functional.interpolate(
                    y, size=(h, w), mode='bilinear', align_corners=False)
            return f

        mode = self._tta_mode
        H = self._flip_flow_aware
        V = lambda t: self._flip_flow_aware(t, vertical=True)     # noqa: E731
        HV = lambda t: H(V(t))                                    # noqa: E731

        if mode == 'flip':
            return [_id, H]

        if mode == 'flipv_scale':
            flips = [_id, H, V, HV]
            default_scales = (1.0, 0.875, 1.125)
        else:                                    # 'flip_scale' / 'scale5'
            flips = [_id, H]
            default_scales = ((0.75, 0.875, 1.0, 1.125, 1.25)
                              if mode == 'scale5' else (1.0, 0.875, 1.125))
        scales = tuple(self.config.get('eval.tta.scales', None) or default_scales)

        out = []
        for s in scales:
            # s=1.0 用恒等而非重采样：与历史 'flip_scale' 行为逐位一致，便于对比
            f = _id if s == 1.0 else _scale_fn(s)
            for g in flips:
                out.append(lambda t, f=f, g=g: g(f(t)))
        return out

    @staticmethod
    def _flip_flow_aware(t, vertical: bool = False):
        """翻转输入；**2 通道光流必须对相应方向的分量取负**。

        原因：光流是位移场，图像镜像后相应方向的位移随之反向：
          水平翻转（左右镜像，dims=-1）→ 水平位移反向 → **u 取负**；
          竖直翻转（上下镜像，dims=-2）→ 竖直位移反向 → **v 取负**。
        若只翻数组不取负，模型看到的"增强视图"在物理上不成立 → TTA 会掉分。
        （3 通道 RGB、9 通道三帧堆叠：直接翻转即可，无需取负。）
        """
        if not torch.is_tensor(t) or t.dim() != 4:
            return t
        y = torch.flip(t, dims=[2 if vertical else 3])
        if t.shape[1] == 2:
            y = y.clone()
            c = 1 if vertical else 0
            y[:, c] = -y[:, c]
        return y

    @torch.no_grad()
    def evaluate_loader(self, loader, desc: str = "Eval") -> Dict[str, float]:
        """
        在指定 DataLoader 上评估（val / test 共用同一实现，保证口径一致）

        Args:
            loader: DataLoader（val_loader 或 test_loader）
            desc:   进度条描述

        Returns:
            dict: 评估指标 (loss / accuracy / UF1 / UAR / WF1)
        """
        self.evaluator.reset()
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        iter_data = tqdm(loader, desc=desc)
        for batch_idx, (data, target) in enumerate(iter_data):
            # 移动到设备（支持 tuple 输入，如 rgb_flow）
            if isinstance(data, (list, tuple)):
                data = [d.to(self.device) for d in data]
            else:
                data = data.to(self.device)
            target = target.to(self.device)
            logits_is_prob = False   # TTA 分支输出的是概率，而非原始 logits

            if self.config.get('model.use_target'):
                output = self.model(data, target)
                # 支持两种返回格式
                if isinstance(output, tuple):
                    logits, model_loss = output
                    loss = model_loss
                else:
                    logits = output
                    loss = self.criterion(logits, target)
            elif self._tta_mode in ('flip', 'flip_scale', 'flipv_scale', 'scale5'):
                # ---- 归纳式 TTA：多视图概率平均 ----
                # 合规：不读测试标签、不更新任何参数或 BN 统计（inductive），
                # 与转导式 bn_adapt/tent 有本质区别（后者实测有害）。
                with torch.no_grad():
                    views = self._tta_views()
                    prob = None
                    for fn in views:
                        d = [fn(x) for x in data] if isinstance(data, (list, tuple)) else fn(data)
                        p = self.model(d).softmax(dim=1)
                        prob = p if prob is None else prob + p
                    prob = prob / len(views)
                logits = prob          # 指标只依赖 argmax，概率与 logits 等价
                logits_is_prob = True
                loss = self.criterion(logits, target)
            else:
                output = self.model(data)
                logits = output
                loss = self.criterion(logits, target)

            # ---- 长尾 logit 调整（Menon et al., ICLR2021 "Long-Tail Learning via Logit Adjustment"）----
            # 推理时 p /= prior^τ，抵消"训练中多数类被系统性高估"的偏置 → 直接抬升宏平均
            # （UF1/UAR），准确率基本不变。先验只由**训练集**标签统计得到，无测试信息泄漏；
            # τ=0（默认）时完全关闭，行为与改动前一致。与翻转型 TTA 可叠加。
            _tau = self._adj_tau
            if _tau > 0:
                _prior = self._class_prior()
                if _prior is not None:
                    p = logits if logits_is_prob else torch.softmax(logits, dim=1)
                    # 防御：先验维数必须与类别数一致，否则会以难懂的广播错误崩掉整个 fold
                    # （维数不匹配时曾发生）。宁可跳过并告警，也不要静默出错。
                    if _prior.numel() == p.size(1):
                        logits = p / (_prior.to(p.device) ** _tau)
                    else:
                        logger.warning(
                            f"[LogitAdj] 先验维数 {_prior.numel()} 与类别数 {p.size(1)} 不一致，"
                            f"本次跳过 logit 调整")

            batch_size = target.size(0)  # 从target获取batch size（支持多模态输入）
            total_loss += loss.item() * batch_size
            pred = logits.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

            # 更新进度条显示
            usage = tool.get_gpu_usage(self.device)
            use_device = "Device: CPU" if usage == "CPU" else "GPU RAM: " + usage
            iter_data.set_postfix_str(set_color(use_device, "yellow"))

        # 计算指标
        avg_loss = total_loss / max(len(loader.dataset), 1)
        accuracy = sum(1 for p, t in zip(all_preds, all_targets)
                       if p == t) / max(len(all_targets), 1)

        # 更新评估器
        self.evaluator.update(all_targets, all_preds)
        eval_metrics = self.evaluator.compute_metrics()

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'UF1': eval_metrics['UF1'],
            'UAR': eval_metrics['UAR'],
            'WF1': eval_metrics['WF1']
        }

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """在测试集上评估（保留原接口；等价于 evaluate_loader(self.test_loader)）"""
        return self.evaluate_loader(self.test_loader, desc="Test Eval")
    
    def adapt_on_test_subject(self, mode: str = "bn_adapt"):
        """转导式测试时适应（TTA）——**只使用测试被试的无标签输入**。

        合规声明（必须在论文/回复信中如实写明，并与 inductive 结果分列两行）：
          * 不读取测试标签，不参与任何模型选择（epoch / 超参仍只在 val 上定）；
          * 属于 transductive 设定，对比方法必须施加**相同**处理才可比。

        mode:
          'bn_adapt': 只把 BN 层切到 train() 前向，用测试集重估 running statistics
                      （AdaBN 风格；不更新任何可学习参数）
          'tent'    : 在 bn_adapt 之上，对归一化层的 weight/bias 做熵最小化
                      （TENT 风格；只更新归一化仿射参数）
        """
        steps = int(self.config.get('eval.tta.steps', 32))
        lr = float(self.config.get('eval.tta.lr', 1e-4))
        model = self.model
        logger.info(f"[TTA] 转导式适应开始: mode={mode}, steps={steps} batch, lr={lr} "
                    f"（仅用无标签测试输入，不读标签）")

        def _mv(x):
            if isinstance(x, (list, tuple)):
                return [_mv(i) for i in x]
            return x.to(self.device) if torch.is_tensor(x) else x

        def _bn_train(flag: bool):
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.train(flag)

        for p in model.parameters():
            p.requires_grad_(False)
        params = []
        if mode == 'tent':
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.LayerNorm, nn.GroupNorm)):
                    for name in ('weight', 'bias'):
                        pp = getattr(m, name, None)
                        if pp is not None:
                            pp.requires_grad_(True)
                            params.append(pp)
        opt = torch.optim.Adam(params, lr=lr) if params else None

        model.eval()       # dropout 等保持 eval...
        _bn_train(True)    # ...只让 BN 处于 train（重估 running stats）

        seen = 0
        while seen < steps:
            progressed = False
            for data, _target in self.test_loader:   # 标签仅随 batch 返回，不参与任何计算
                data = _mv(data)
                if opt is not None:
                    opt.zero_grad()
                out = model(data)
                logits = out[0] if isinstance(out, tuple) else out
                if opt is not None:
                    prob = torch.softmax(logits, dim=1)
                    ent = -(prob * torch.log(prob + 1e-8)).sum(dim=1).mean()
                    ent.backward()
                    opt.step()
                seen += 1
                progressed = True
                if seen >= steps:
                    break
            if not progressed:
                break

        for p in model.parameters():
            p.requires_grad_(True)
        _bn_train(False)
        model.eval()
        logger.info(f"[TTA] 完成（共前向 {seen} 个 batch）")

    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict):
        """
        保存检查点
        
        Args:
            filepath: 保存路径
            epoch: 当前epoch
            metrics: 评估指标
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        加载检查点
        
        Args:
            filepath: 检查点路径
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 验证配置是否匹配
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            current_frame_type, saved_frame_type = self.config.get('data.frame_type'), saved_config.get('data.frame_type')
            current_split_mode, saved_split_mode = self.config.get('data.split_mode'), saved_config.get('data.split_mode')
            
            if current_frame_type != saved_frame_type or current_split_mode != saved_split_mode:
                raise RuntimeError(
                    f"配置不匹配！\n"
                    f"  检查点中的 frame_type: {saved_frame_type}，当前配置的 frame_type: {current_frame_type}\n"
                    f"  检查点中的 split_mode: {saved_split_mode}，当前配置的 split_mode: {current_split_mode}\n"
                    f"请确保训练和测试使用相同的 frame_type 配置。"
                )
        
        # 加载模型权重（允许部分不匹配，给出警告）
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                raise RuntimeError(
                    f"模型架构不匹配！\n"
                    f"{str(e)}\n\n"
                    f"可能原因：\n"
                    f"  1. 训练和测试使用了不同的 frame_type 配置\n"
                    f"  2. 模型架构被修改过\n"
                    f"  3. 检查点来自不同的模型\n\n"
                    f"请检查：\n"
                    f"  - data.frame_type 配置是否一致\n"
                    f"  - 使用的模型类是否正确\n"
                ) from e
            else:
                raise
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"检查点已加载: {filepath}")
        logger.info(f"Epoch: {checkpoint['epoch']}")
        logger.info(f"Metrics: {checkpoint['metrics']}")
    
    def train(self) -> Dict[str, float]:
        """
        完整训练流程（模型选择协议）

        协议:
          1. 每个 epoch 在**折内验证集 (val)** 上评估，按 val UF1 选 best 并早停；
          2. 训练结束后加载 val-best 权重，在**测试被试上只评估一次**；
          3. 测试被试全程不参与模型选择 / 早停 / LR 调度 / 超参固定。

        val_mode='none' 为历史行为（在测试被试上选 best），仅用于复现旧结果。

        Returns:
            dict: 测试被试上的单次评估指标
        """
        logger.info("="*80)
        logger.info("开始训练")
        logger.info("="*80)

        epochs = self.config.get('train.epochs', 200)
        self.config['flag.eval'] = 'train'
        start_time = time.time()
        es_patience = self.es_patience

        # 模型选择所用的数据集：val（正确协议）；legacy 模式下退化为 test
        sel_loader = self.val_loader if self.use_val_for_selection else self.test_loader
        sel_name = "Val" if self.use_val_for_selection else "Test(legacy)"
        history = []
        stopped_early = False

        for epoch in range(1, epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)

            # 选择用评估（val；legacy 模式为 test）
            eval_metrics = self.evaluate_loader(sel_loader, desc=f"{sel_name} Eval")

            # LR Scheduler step
            self._scheduler_step(eval_metrics)
            current_lr = self.optimizer.param_groups[0]['lr']

            history.append({
                'epoch': epoch,
                'lr': current_lr,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'sel_loss': eval_metrics['loss'],
                'sel_accuracy': eval_metrics['accuracy'],
                'sel_UF1': eval_metrics['UF1'],
                'sel_UAR': eval_metrics['UAR'],
                'sel_WF1': eval_metrics['WF1'],
            })

            # 打印进度
            if epoch % 5 == 0 or epoch == 1:
                log_msg = (
                    f"Epoch [{epoch}/{epochs}] LR: {current_lr:.2e} | "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"{sel_name} Loss: {eval_metrics['loss']:.4f} "
                    f"{sel_name} Acc: {eval_metrics['accuracy']:.4f} "
                    f"{sel_name} UF1: {eval_metrics['UF1']:.4f} "
                    f"UAR: {eval_metrics['UAR']:.4f} "
                    f"WF1: {eval_metrics['WF1']:.4f}"
                )
                logger.info(log_msg)

            # ---- 保存最佳模型 (按 val UF1) ----
            is_better = False
            if eval_metrics['UF1'] > self.best_uf1 + 1e-5:
                is_better = True
                self.best_uf1 = eval_metrics['UF1']
                self.best_epoch = epoch
            if eval_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = eval_metrics['accuracy']

            if is_better:
                self.es_counter = 0
                self.best_model_state = {k: v.detach().clone()
                                         for k, v in self.model.state_dict().items()}

                checkpoint_dir = self.config.get('base.checkpoint.checkpoint_dir', 'checkpoints')
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
                self.save_checkpoint(str(checkpoint_path), epoch, eval_metrics)
                logger.info(f"  >>> 保存最佳模型 ({sel_name} UF1={self.best_uf1:.4f}) @ epoch {epoch}")

                # ---- [可选] 逐折另存一份，供**多种子集成（ensemble）**使用 ----
                # 背景（2026-09-26）：LOSO 一个 run 里逐折顺序训练，都写同一个
                # `checkpoints/best_model.pth` → **只有最后一折的权重留下来**
                # （实测 236 个 run 只有 236 个 .pth，即每 run 一个）。
                # 于是"把多个种子的同类折模型做概率平均"这种最可靠的提分手段
                # 在现有产物上根本做不了。开启本开关后，每折会另存
                # `fold_<折号>_<测试被试>.pth`，多 seed × 同折即可直接集成。
                # 默认关闭，不影响任何既有行为。
                if self.config.get('base.checkpoint.keep_folds', False):
                    _sub = str(self.config.get('data.test_subject', 'unknown'))
                    _fidx = self.config.get('data.fold_idx', None)
                    _fname = (f"fold_{_fidx}_{_sub}.pth" if _fidx is not None
                              else f"fold_{_sub}.pth")
                    self.save_checkpoint(str(Path(checkpoint_dir) / _fname),
                                         epoch, eval_metrics)
                    logger.info(f"  >>> 逐折检查点已另存: {_fname}")
            else:
                self.es_counter += 1

            # ---- Early Stopping ----
            if self.es_enabled and self.es_counter >= es_patience:
                logger.info(f"Early stopping @ epoch {epoch} (patience={es_patience}, "
                            f"best {sel_name} UF1={self.best_uf1:.4f})")
                stopped_early = True
                break

        # 训练完成
        elapsed_time = time.time() - start_time
        logger.info("="*80)
        logger.info(f"训练完成！耗时: {elapsed_time:.2f}秒"
                    + ("（early stopped）" if stopped_early else ""))
        logger.info(f"最佳 {sel_name} UF1: {self.best_uf1:.4f} @ epoch {self.best_epoch}"
                    f" | 最佳 {sel_name} Acc: {self.best_accuracy:.4f}")
        logger.info("="*80)

        # ---- 测试被试：只评估一次（不参与任何选择）----
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        logger.info(f"已加载 {sel_name}-best 权重，在测试被试上执行唯一一次评估")

        # ---- [可选] 转导式测试时适应（TTA）----
        # 默认 'none' → inductive 协议，行为与改动前逐位一致；
        # 开启后属 transductive 设定：只用无标签测试输入，必须在论文中单独一行如实报告，
        # 且对所有对比方法（HTNet/MPFNet…）施加相同处理才可比。
        _tta_mode = str(self.config.get('eval.tta.mode', 'none')).lower()
        if _tta_mode in ('bn_adapt', 'tent'):
            self.adapt_on_test_subject(_tta_mode)

        test_metrics = self.evaluate_loader(self.test_loader, desc="Test Eval (once)")

        logger.info("测试结果（测试被试，单次评估）:")
        logger.info(f"  准确率: {test_metrics['accuracy']:.4f}")
        logger.info(f"  UF1: {test_metrics['UF1']:.4f}")
        logger.info(f"  UAR: {test_metrics['UAR']:.4f}")
        logger.info(f"  WF1: {test_metrics['WF1']:.4f}")

        # 最终评估报告
        self.evaluator.print_report(self.config)

        # 保存最终结果（含选择协议与 val/test 分离证据）
        result_file = Path(self.config['base.results']) / 'final_results.txt'
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {elapsed_time:.2f}秒\n")
            f.write(f"模型选择协议: {sel_name} (best by UF1)\n")
            f.write(f"最佳 {sel_name} UF1: {self.best_uf1:.4f} @ epoch {self.best_epoch}\n")
            f.write(f"最佳 {sel_name} Acc: {self.best_accuracy:.4f}\n")
            f.write(f"\n[测试被试单次评估]\n")
            for key in ('UF1', 'UAR', 'accuracy', 'WF1', 'loss'):
                f.write(f"{key}: {test_metrics[key]}\n")
            f.write(f"\n详细指标:\n")
            metrics = self.evaluator.compute_metrics()
            for key, value in metrics.items():
                if key != 'Confusion_Matrix':
                    f.write(f"{key}: {value}\n")

        # 保存验证曲线（供手稿 Fig.2 使用）
        try:
            curve_file = Path(self.config['base.results']) / 'val_curve.json'
            with open(curve_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"验证曲线保存失败: {e}")

        return test_metrics
    
    @torch.no_grad()
    def test(self, checkpoint_path: str) -> Dict[str, float]:
        """
        测试模式
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            dict: 测试评估指标
        """
        logger.info("="*80)
        logger.info("开始测试")
        logger.info("="*80)
        self.config['flag.eval'] = 'test'
        
        # 加载模型
        self.load_checkpoint(checkpoint_path)
        
        # 评估
        eval_metrics = self.evaluate()
        
        logger.info(f"测试结果:")
        logger.info(f"  准确率: {eval_metrics['accuracy']:.4f}")
        logger.info(f"  UF1: {eval_metrics['UF1']:.4f}")
        logger.info(f"  UAR: {eval_metrics['UAR']:.4f}")
        logger.info(f"  WF1: {eval_metrics['WF1']:.4f}")
        
        # 打印详细报告
        self.evaluator.print_report(self.config)
        
        return eval_metrics
    
    def cross_validate(self, folds: int = 5):
        """
        交叉验证
        
        Args:
            folds: 折数
        """
        logger.warning("交叉验证功能尚未完全实现")
        # 注：LOSO 划分由数据层实现（见 src/data/dataset.py 的 split_mode='loso'），训练器本身无需改动
        pass
