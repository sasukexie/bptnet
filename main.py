"""
微表情识别系统 - 主入口脚本

使用方法:
    # 直接运行（使用默认配置 htnet.yml）
    python run_main.py
    
    # 指定其他模型
    python run_main.py --model swin_transformer
    
    # 训练后自动测试
    python run_main.py --auto-test
    
    # 启用交叉验证
    python run_main.py --cross-validate
    
    # 指定自定义配置文件
    python run_main.py --config src/config/my_custom.yml
    
    # 覆盖配置中的某些参数
    python run_main.py --overrides training.epochs=300 data.batch_size=64
"""

import importlib
import json
import os
from pathlib import Path

import src.utils.tool as tool
from src.data.dataset import get_all_subjects
from src.evaluation.metrics import print_cv_average_report
from src.trainer import Trainer
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger, create_timestamped_log


def train(config, logger):
    # 创建模型
    model_module = importlib.import_module(f"src.model.{config['model.name'].lower()}")
    model = getattr(model_module, config['model.name'])(config)

    tool.start_common_set(config)
    # 创建训练器
    trainer = Trainer(config, model)

    # 训练
    if config.get('train.enabled', False):
        logger.info("训练模式...")
        trainer.train()

    # 验证/测试
    eval_metrics = None
    if config.get('eval.enabled', False):
        logger.info("测试模式...")

        best_checkpoint = Path(config.get('base.checkpoint.checkpoint_dir', 'checkpoints')) / 'best_model.pth'

        if best_checkpoint.exists():
            eval_metrics = trainer.test(checkpoint_path=str(best_checkpoint))
        else:
            logger.warning(f"最佳模型检查点不存在: {best_checkpoint}")

    logger.info("程序执行完成！")
    logger.info(f"结果保存在: {config['base.output_dir']}")

    return eval_metrics


def main(init_config):
    """主函数"""
    # 创建配置管理器
    config_manager = ConfigManager(init_config)
    config = config_manager.get_config()

    model = config['model.name']
    dataset = config['data.dataset']
    split_mode = config['data.split_mode']
    frame_type = config['data.frame_type']

    # 如果存在 train_flag.json 文件，则读取其中的训练标记, 如果不存在，则创建一个空的训练标记字典
    # 读取 saved/train_flag.json 文件
    train_flag_file = os.path.join(init_config['base'].get('base_dir', init_config['base']['output_dir']), f"train_flag_{init_config['flag']['train_flag']}.json")
    if os.path.exists(train_flag_file):
        with open(train_flag_file, 'r') as f:
            train_flag = json.load(f)
    else:
        train_flag = {}
    train_key = model + '_' + dataset + '_' + '_' + split_mode + '_' + frame_type
    if train_key in train_flag:
        print(f"模型 {model} 在数据集 {dataset} 下 split_mode={split_mode} frame_type={frame_type} 已训练过，跳过")
        return

    # 设置日志
    log_file = create_timestamped_log(
        log_dir=config['log.log_dir'],
        prefix=f'train_{config["model.name"]}_{config["data.dataset"]}'
    )
    logger = setup_logger(
        name=config['model.name'],
        log_level=config['log.level'],
        log_file=log_file
    )

    logger.info("=" * 80)
    logger.info("微表情识别系统启动")
    logger.info(f"模型名称: {config['model.name']}")
    logger.info(f"数据集名称: {config['data.dataset']}")
    logger.info(f"实验名称: {config['base.name']}")
    logger.info(f"配置文件: {config}")
    logger.info("=" * 80)

    # 打印配置摘要
    config_manager.print_config()

    try:
        if config['data.split_mode'] == 'loso':
            subjects = get_all_subjects(config)
            logger.info(f"\n{'=' * 80}")
            logger.info(f"开始 LOSO 交叉验证（共 {len(subjects)} 个受试者）")
            logger.info(f"{'=' * 80}\n")

            valid_folds = 0
            skipped_folds = 0
            all_metrics = []  # 收集每个 fold 的测试指标

            for sub_idx, sub in enumerate(subjects, 1):
                logger.info(f"\n{'-' * 80}")
                logger.info(f"Fold {sub_idx}/{len(subjects)}: 测试受试者 = {sub}")

                # 检查 fold 有效性
                train_subjects = [s for s in subjects if s != sub]
                is_valid, missing_classes = tool.check_fold_validity(config, train_subjects, sub)

                if not is_valid:
                    logger.warning(f"❌ 跳过此 fold！训练集缺失类别: {missing_classes}")
                    skipped_folds += 1
                    continue

                logger.info(f"✅ Fold 有效，开始训练")
                valid_folds += 1

                config['data.test_subject'] = sub
                eval_metrics = train(config, logger)
                if eval_metrics is not None:
                    all_metrics.append(eval_metrics)

            logger.info(f"\n{'=' * 80}")
            logger.info(f"LOSO 验证完成")
            logger.info(f"总 fold 数: {len(subjects)}")
            logger.info(f"有效 fold 数: {valid_folds}")
            logger.info(f"跳过 fold 数: {skipped_folds}")
            logger.info(f"有效率: {valid_folds / len(subjects) * 100:.1f}%")
            logger.info(f"{'=' * 80}\n")

            # 输出 LOSO 平均结果
            print_cv_average_report(all_metrics, config, config['data.split_mode'])

        elif config['data.split_mode'] == '5fold':
            all_metrics = []  # 收集每个 fold 的测试指标
            for idx in range(5):
                config['data.fold_idx'] = idx
                eval_metrics = train(config, logger)
                if eval_metrics is not None:
                    all_metrics.append(eval_metrics)

            # 输出 5-fold 平均结果
            print_cv_average_report(all_metrics, config, config['data.split_mode'])

        else:  # fixed
            train(config, logger)

        # 记录训练标记
        train_flag[train_key] = True
        with open(train_flag_file, 'w') as f:
            json.dump(train_flag, f)
    except Exception as e:
        logger.error(f"模型|参数: {train_key} 训练失败: {str(e)}, 报错信息: {str(e)}", exc_info=True)

