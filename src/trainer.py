"""
训练器：负责模型的训练、测试和交叉验证
支持标准 CrossEntropyLoss 和 Focal Loss (含类别权重)
v2: 新增 LR Scheduler + Warmup + Gradient Clipping + Early Stopping
"""

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
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)          # p_t = exp(-CE)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

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
        
        # 创建数据加载器
        self.train_loader, self.test_loader = create_dataloaders(config)
        
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
        创建 Warmup + CosineAnnealing 学习率调度器
        
        warmup 阶段: 线性从 0 升至 learning_rate
        cosine 阶段: 余弦退火从 learning_rate 降至 ~0
        """
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
            
            # 统计 (mixup batch 不追踪准确率，因为标签是混合的)
            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            if not do_mixup:
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
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
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        Returns:
            dict: 评估指标
        """
        self.evaluator.reset()
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        iter_data = tqdm(self.test_loader, desc=f"Eval")
        for batch_idx, (data, target) in enumerate(iter_data):
            # 移动到设备（支持 tuple 输入，如 rgb_flow）
            if isinstance(data, (list, tuple)):
                data = [d.to(self.device) for d in data]
            else:
                data = data.to(self.device)
            target = target.to(self.device)
            
            if self.config.get('model.use_target'):
                output = self.model(data, target)
                # 支持两种返回格式
                if isinstance(output, tuple):
                    logits, model_loss = output
                    loss = model_loss
                else:
                    logits = output
                    loss = self.criterion(logits, target)
            else:
                output = self.model(data)
                logits = output
                loss = self.criterion(logits, target)
            
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
        avg_loss = total_loss / len(self.test_loader.dataset)
        accuracy = sum(1 for p, t in zip(all_preds, all_targets) 
                      if p == t) / len(all_targets)
        
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
    
    def train(self):
        """
        完整训练流程 (v2: + Scheduler + UF1 追踪 + Early Stopping)
        """
        logger.info("="*80)
        logger.info("开始训练")
        logger.info("="*80)
        
        epochs = self.config.get('train.epochs', 200)
        self.config['flag.eval'] = 'train'
        start_time = time.time()
        es_patience = self.es_patience
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 评估
            eval_metrics = self.evaluate()
            
            # LR Scheduler step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 打印进度
            if epoch % 10 == 0 or epoch == 1:
                log_msg = (
                    f"Epoch [{epoch}/{epochs}] LR: {current_lr:.2e} | "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Test Loss: {eval_metrics['loss']:.4f} "
                    f"Test Acc: {eval_metrics['accuracy']:.4f} "
                    f"UF1: {eval_metrics['UF1']:.4f} "
                    f"UAR: {eval_metrics['UAR']:.4f} "
                    f"WF1: {eval_metrics['WF1']:.4f}"
                )
                logger.info(log_msg)
            
            # ---- 保存最佳模型 (优先按 UF1, 备选 Acc) ----
            is_better = False
            if eval_metrics['UF1'] > self.best_uf1 + 1e-5:
                is_better = True
                self.best_uf1 = eval_metrics['UF1']
            if eval_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = eval_metrics['accuracy']
            
            if is_better:
                self.es_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                checkpoint_dir = self.config.get('base.checkpoint.checkpoint_dir', 'checkpoints')
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
                self.save_checkpoint(str(checkpoint_path), epoch, eval_metrics)
                logger.info(f"  >>> 保存最佳模型 (UF1={self.best_uf1:.4f}) @ epoch {epoch}")
            else:
                self.es_counter += 1
            
            # ---- Early Stopping ----
            if self.es_enabled and self.es_counter >= es_patience:
                logger.info(f"Early stopping @ epoch {epoch} (patience={es_patience}, best UF1={self.best_uf1:.4f})")
                break
        
        # 训练完成
        elapsed_time = time.time() - start_time
        logger.info("="*80)
        logger.info(f"训练完成！耗时: {elapsed_time:.2f}秒 (early stopped)" if self.es_counter >= es_patience else f"训练完成！耗时: {elapsed_time:.2f}秒")
        logger.info(f"最佳 UF1: {self.best_uf1:.4f} | 最佳 Acc: {self.best_accuracy:.4f}")
        logger.info("="*80)
        
        # 最终评估报告
        self.evaluator.print_report(self.config)
        
        # 保存最终结果
        result_file = Path(self.config['base.results'])/ 'final_results.txt'
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {elapsed_time:.2f}秒\n")
            f.write(f"最佳 UF1: {self.best_uf1:.4f}\n")
            f.write(f"最佳准确率: {self.best_accuracy:.4f}\n")
            f.write(f"\n详细指标:\n")
            metrics = self.evaluator.compute_metrics()
            for key, value in metrics.items():
                if key != 'Confusion_Matrix':
                    f.write(f"{key}: {value}\n")
    
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
        # TODO: 实现留一法交叉验证（LOSO）
        pass
