"""
评估指标计算模块
包含UF1、UAR、WF1、混淆矩阵等微表情识别常用指标
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # 无头服务器不弹窗，静默生成图片
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime
from src.utils.logger import logger

# 设置中文字体支持，避免警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _cfg_get(config, dotted, default=None):
    """安全读取嵌套配置（支持点号路径），兼容 ConfigDict。"""
    cur = config
    for part in dotted.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def append_result_record(record: Dict, config):
    """将单条实验结果追加写入 JSONL（供论文表格/绘图脚本直接消费）。

    输出路径优先取环境变量 MER_RESULTS_DIR（每台服务器指向各自的
    results_serverN 目录），否则回退到日志目录。
    """
    d = os.environ.get('MER_RESULTS_DIR')
    if not d:
        d = _cfg_get(config, 'log.log_dir', 'log')
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, 'metrics.jsonl')
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:
        logger.warning(f"写入 metrics.jsonl 失败: {e}")


def calculate_f1_recall(gt: List[int], pred: List[int]) -> Tuple[float, float]:
    """
    计算二分类的F1分数和召回率
    
    Args:
        gt: 真实标签列表
        pred: 预测标签列表
        
    Returns:
        tuple: (F1分数, 平均召回率)
    """
    try:
        # 计算混淆矩阵，指定labels确保返回2x2矩阵
        cm = confusion_matrix(gt, pred, labels=[0, 1])
        
        # 确保是2x2矩阵
        if cm.shape != (2, 2):
            logger.warning(f"混淆矩阵形状异常: {cm.shape}，预期(2, 2)")
            return 0.0, 0.0
        
        tn, fp, fn, tp = cm.ravel()
        
        # 计算F1分数
        f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        
        # 计算召回率
        num_samples = sum(1 for x in gt if x == 1)
        average_recall = tp / num_samples if num_samples > 0 else 0.0
        
        return f1_score, average_recall
        
    except Exception as e:
        logger.warning(f"计算F1/Recall时出错: {e}")
        return 0.0, 0.0


def calculate_uf1_uar(gt: List[int], pred: List[int], 
                      num_classes: int = 3) -> Tuple[float, float, float]:
    """
    计算UF1（未加权F1）、UAR（未加权平均召回率）和WF1（加权F1）
    
    这是微表情识别的标准评估指标，对每个类别分别计算
    F1和召回率，然后取平均。
    
    Args:
        gt: 真实标签列表
        pred: 预测标签列表
        num_classes: 类别数量
        
    Returns:
        tuple: (UF1, UAR, WF1)
        
    Example:
        >>> gt = [0, 1, 2, 0, 1, 2]
        >>> pred = [0, 1, 1, 0, 2, 2]
        >>> uf1, uar, wf1 = calculate_uf1_uar(gt, pred)
        >>> print(f"UF1: {uf1:.4f}, UAR: {uar:.4f}, WF1: {wf1:.4f}")
    """
    from collections import Counter
    
    # 统计每个类别的样本数（用于WF1）
    label_counts = Counter(gt)
    total_samples = len(gt)
    
    f1_list = []
    recall_list = []
    weighted_f1_sum = 0.0
    
    # 对每个类别计算One-vs-All的F1和Recall
    for class_idx in range(num_classes):
        # 转换为二分类问题
        gt_binary = [1 if x == class_idx else 0 for x in gt]
        pred_binary = [1 if x == class_idx else 0 for x in pred]
        
        try:
            f1, recall = calculate_f1_recall(gt_binary, pred_binary)
            f1_list.append(f1)
            recall_list.append(recall)
            
            # 累加加权F1（权重为该类别的样本比例）
            class_weight = label_counts.get(class_idx, 0) / total_samples if total_samples > 0 else 0
            weighted_f1_sum += f1 * class_weight
            
        except Exception as e:
            logger.warning(f"计算类别 {class_idx} 的指标时出错: {e}")
            continue
    
    # 计算平均值
    if len(f1_list) == 0 or len(recall_list) == 0:
        logger.warning("没有有效的类别指标")
        return 0.0, 0.0, 0.0
    
    uf1 = np.mean(f1_list)
    uar = np.mean(recall_list)
    wf1 = weighted_f1_sum  # 加权F1
    
    return uf1, uar, wf1


def plot_and_save_confusion_matrix(
    gt: List[int], 
    pred: List[int], 
    class_names: List[str] = None,
    save_path: str = None,
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> Optional[str]:
    """
    绘制并保存混淆矩阵可视化图
    
    Args:
        gt: 真实标签列表
        pred: 预测标签列表
        class_names: 类别名称列表
        save_path: 保存路径（可选），如果不提供则返回图像对象
        title: 图表标题
        normalize: 是否归一化（显示百分比）
    
    Returns:
        str or None: 如果提供了save_path，返回保存的文件路径；否则返回None
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(set(gt)))]
    
    # 计算混淆矩阵
    cm = confusion_matrix(gt, pred)
    
    if normalize:
        # 防止除以0：对于没有样本的行，设置为0
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        # 避免除以0，将0替换为1（这样0/1=0，不会产生NaN）
        row_sums_safe = np.where(row_sums == 0, 1, row_sums)
        cm_normalized = cm.astype('float') / row_sums_safe
        # 对于原本就是0的行，确保结果为0而不是NaN
        cm_normalized = np.where(row_sums == 0, 0, cm_normalized)
        display_cm = cm_normalized
        fmt = '.2f'
    else:
        display_cm = cm
        fmt = 'd'
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(
        display_cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    # 设置标签和标题
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存到: {save_path}")
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None


def compute_confusion_matrix(gt: List[int], pred: List[int], 
                            class_names: List[str] = None) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        gt: 真实标签列表
        pred: 预测标签列表
        class_names: 类别名称列表
        
    Returns:
        np.ndarray: 混淆矩阵
    """
    if class_names is None:
        class_names = ['Negative', 'Positive', 'Surprise']
    
    cm = confusion_matrix(gt, pred)
    return cm


def print_cv_average_report(all_fold_metrics, config, cv_name):
    """
    打印交叉验证平均报告
    
    Args:
        all_fold_metrics: 各 fold 的指标列表，每个元素为 {'accuracy': ..., 'UF1': ..., 'UAR': ..., 'WF1': ...}
        config: 配置字典
        cv_name: 交叉验证名称，如 "5Fold" 或 "loso"
    """
    if not all_fold_metrics:
        logger.warning(f"{cv_name}: 没有有效的 fold 结果，无法计算平均值")
        return

    num_folds = len(all_fold_metrics)
    avg_acc = np.mean([m['accuracy'] for m in all_fold_metrics])
    avg_uf1 = np.mean([m['UF1'] for m in all_fold_metrics])
    avg_uar = np.mean([m['UAR'] for m in all_fold_metrics])
    avg_wf1 = np.mean([m['WF1'] for m in all_fold_metrics])

    std_acc = np.std([m['accuracy'] for m in all_fold_metrics])
    std_uf1 = np.std([m['UF1'] for m in all_fold_metrics])
    std_uar = np.std([m['UAR'] for m in all_fold_metrics])
    std_wf1 = np.std([m['WF1'] for m in all_fold_metrics])

    report = f"""
{'='*60}
微表情识别评估 {cv_name} 交叉验证平均报告
{'='*60}
模型: {config['model.name']}, 数据集: {config['data.dataset']}
总样本数: N/A (CV平均)
有效 fold 数: {num_folds}
其他参数: split_mode:{config['data.split_mode']},frame_type:{config['data.frame_type']},augmentation:{config['data.augmentation.enabled']}
ACC (整体准确率): {avg_acc:.4f} ± {std_acc:.4f}
UF1 (未加权F1): {avg_uf1:.4f} ± {std_uf1:.4f}
UAR (未加权平均召回率): {avg_uar:.4f} ± {std_uar:.4f}
WF1 (加权F1): {avg_wf1:.4f} ± {std_wf1:.4f}
ACC|UF1|UAR|WF1: {avg_acc:.4f}  {avg_uf1:.4f}   {avg_uar:.4f}  {avg_wf1:.4f}
混淆矩阵:
N/A (各 fold 混淆矩阵见上方)
{'各 Fold 详细结果:'}
"""
    for i, m in enumerate(all_fold_metrics):
        report += f"  fold_{i}: ACC={m['accuracy']:.4f}, UF1={m['UF1']:.4f}, UAR={m['UAR']:.4f}, WF1={m['WF1']:.4f}\n"

    report += f"{'='*60}\n"

    logger.info(report)
    # 记录评估报告
    with open(f'{config["log.log_dir"]}/result_{config["flag.id"]}.log', 'a') as f:
        f.write(f'{report} \n')

    # ---- 追加 JSON 记录（论文表格 / 绘图脚本直接消费） ----
    try:
        record = {
            'kind': 'cv_average',
            'model': config.get('model.name'),
            'dataset': config.get('data.dataset'),
            'split_mode': config.get('data.split_mode'),
            'frame_type': config.get('data.frame_type'),
            'exp_tag': _cfg_get(config, 'flag.exp_tag', ''),
            'phase_fusion': _cfg_get(config, 'model.phase_fusion'),
            'use_phase_tokens': _cfg_get(config, 'model.use_phase_tokens'),
            'transformer_layers': _cfg_get(config, 'model.transformer_layers'),
            'drop_path_rate': _cfg_get(config, 'model.drop_path_rate'),
            'mixup_alpha': _cfg_get(config, 'data.mixup.alpha'),
            'seed': config.get('base.seed'),
            'learning_rate': config.get('train.learning_rate'),
            'cv_name': cv_name,
            'num_folds': num_folds,
            'metrics': {
                'ACC': [round(float(m['accuracy']), 4) for m in all_fold_metrics],
                'UF1': [round(float(m['UF1']), 4) for m in all_fold_metrics],
                'UAR': [round(float(m['UAR']), 4) for m in all_fold_metrics],
                'WF1': [round(float(m['WF1']), 4) for m in all_fold_metrics],
            },
            'mean': {
                'ACC': round(float(avg_acc), 4),
                'UF1': round(float(avg_uf1), 4),
                'UAR': round(float(avg_uar), 4),
                'WF1': round(float(avg_wf1), 4),
            },
            'std': {
                'ACC': round(float(std_acc), 4),
                'UF1': round(float(std_uf1), 4),
                'UAR': round(float(std_uar), 4),
                'WF1': round(float(std_wf1), 4),
            },
            'timestamp': datetime.now().isoformat(),
        }
        append_result_record(record, config)
    except Exception as e:
        logger.warning(f"构建 CV 结果记录失败: {e}")


class Evaluator:
    """
    评估器类
    封装所有评估功能
    """
    
    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        """
        初始化评估器
        
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表（可选），如果不提供则使用默认名称
        """
        self.num_classes = num_classes
        if class_names is not None:
            self.class_names = class_names[:num_classes]
        else:
            # 默认类别名称（支持3类和7类）
            default_names_3 = ['Negative', 'Positive', 'Surprise']
            default_names_7 = ['Negative', 'Positive', 'Surprise', 'Repression', 'Others', 'Happy', 'Anger']
            if num_classes <= 3:
                self.class_names = default_names_3[:num_classes]
            elif num_classes <= 7:
                self.class_names = default_names_7[:num_classes]
            else:
                self.class_names = [f'Class {i}' for i in range(num_classes)]
        
        # 存储历史结果
        self.all_gt = []
        self.all_pred = []
        
    def update(self, gt: List[int], pred: List[int]):
        """
        更新评估结果
        
        Args:
            gt: 真实标签
            pred: 预测标签
        """
        self.all_gt.extend(gt)
        self.all_pred.extend(pred)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            dict: 包含所有指标的字典
        """
        if len(self.all_gt) == 0:
            raise ValueError("没有数据可以评估")
        
        uf1, uar, wf1 = calculate_uf1_uar(self.all_gt, self.all_pred, self.num_classes)
        cm = compute_confusion_matrix(self.all_gt, self.all_pred, self.class_names)
        
        # 计算整体准确率
        accuracy = sum(1 for g, p in zip(self.all_gt, self.all_pred) 
                      if g == p) / len(self.all_gt)
        
        metrics = {
            'UF1': uf1,
            'UAR': uar,
            'WF1': wf1,
            'Accuracy': accuracy,
            'Confusion_Matrix': cm,
            'Total_Samples': len(self.all_gt)
        }
        
        return metrics
    
    def reset(self):
        """重置评估器状态"""
        self.all_gt = []
        self.all_pred = []
    
    def print_report(self, config):
        """打印评估报告"""
        metrics = self.compute_metrics()
        flag_eval = config.get('flag.eval', 'train')
        report = f"""
{'='*60}
微表情识别评估 {flag_eval} 报告
{'='*60}
模型: {config['model.name']}, 数据集: {config['data.dataset']}
总样本数: {metrics['Total_Samples']}
其他参数: split_mode:{config['data.split_mode']},frame_type:{config['data.frame_type']},augmentation:{config['data.augmentation.enabled']},test_subject:{config['data.test_subject']},fold_idx:{config['data.fold_idx']}
ACC (整体准确率): {metrics['Accuracy']:.4f}
UF1 (未加权F1): {metrics['UF1']:.4f}
UAR (未加权平均召回率): {metrics['UAR']:.4f}
WF1 (加权F1): {metrics['WF1']:.4f}
ACC|UF1|UAR|WF1: {metrics['Accuracy']:.4f}  {metrics['UF1']:.4f}   {metrics['UAR']:.4f}  {metrics['WF1']:.4f}
混淆矩阵:
{metrics['Confusion_Matrix']}
{'='*60}
"""
        logger.info(report)
        # 记录评估报告
        with open(f'{config["log.log_dir"]}/result_{config["flag.id"]}.log', 'a') as f:
            f.write(f'{report} \n\n')

        # ---- 追加 JSON 记录（论文表格 / 绘图脚本直接消费） ----
        try:
            cm = metrics.get('Confusion_Matrix')
            cm_list = cm.tolist() if hasattr(cm, 'tolist') else cm
            record = {
                'kind': 'single',
                'model': config.get('model.name'),
                'dataset': config.get('data.dataset'),
                'split_mode': config.get('data.split_mode'),
                'frame_type': config.get('data.frame_type'),
                'exp_tag': _cfg_get(config, 'flag.exp_tag', ''),
                'phase_fusion': _cfg_get(config, 'model.phase_fusion'),
                'use_phase_tokens': _cfg_get(config, 'model.use_phase_tokens'),
                'transformer_layers': _cfg_get(config, 'model.transformer_layers'),
                'drop_path_rate': _cfg_get(config, 'model.drop_path_rate'),
                'mixup_alpha': _cfg_get(config, 'data.mixup.alpha'),
                'num_folds': 1,
                'mean': {
                    'ACC': round(float(metrics.get('Accuracy', 0)), 4),
                    'UF1': round(float(metrics.get('UF1', 0)), 4),
                    'UAR': round(float(metrics.get('UAR', 0)), 4),
                    'WF1': round(float(metrics.get('WF1', 0)), 4),
                },
                'metrics': None,
                'confusion_matrix': cm_list,
                'total_samples': metrics.get('Total_Samples'),
                'timestamp': datetime.now().isoformat(),
            }
            append_result_record(record, config)
        except Exception as e:
            logger.warning(f"构建 single 结果记录失败: {e}")
        
        # 保存混淆矩阵可视化
        try:
            output_dir = Path(config.get('base.output_dir')) / 'confusion_matrices'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            model_name = config['model.name']
            dataset_name = config['data.dataset']
            split_mode = config['data.split_mode']
            test_subject = config.get('data.test_subject', 'N/A')
            fold_idx = config.get('data.fold_idx', 'N/A')
            
            filename = f"{model_name}_{dataset_name}_{split_mode}"
            if test_subject != 'N/A':
                filename += f"_sub{test_subject}"
            elif fold_idx != 'N/A':
                filename += f"_fold{fold_idx}"
            filename += ".png"
            
            save_path = output_dir / filename
            
            # 绘制并保存（归一化和非归一化两个版本）
            plot_and_save_confusion_matrix(
                self.all_gt, 
                self.all_pred,
                class_names=self.class_names,
                save_path=str(save_path),
                title=f"{model_name} - {dataset_name}\n({flag_eval})",
                normalize=False
            )
            
            # 归一化版本
            norm_save_path = str(save_path).replace('.png', '_normalized.png')
            plot_and_save_confusion_matrix(
                self.all_gt, 
                self.all_pred,
                class_names=self.class_names,
                save_path=norm_save_path,
                title=f"{model_name} - {dataset_name}\n({flag_eval}, Normalized)",
                normalize=True
            )
            
        except Exception as e:
            logger.warning(f"保存混淆矩阵可视化失败: {e}")
        
        return report
