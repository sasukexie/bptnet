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

import numpy as np

import src.utils.tool as tool
from src.data.dataset import get_all_subjects
from src.evaluation.metrics import print_cv_average_report
from src.trainer import Trainer
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger, create_timestamped_log


def _summarize(records):
    """把逐折指标汇总为 mean ± std（跨 fold）"""
    keys = ['UF1', 'UAR', 'accuracy', 'WF1']
    out = {}
    for k in keys:
        vals = [r[k] for r in records if r.get(k) is not None]
        if not vals:
            out[k] = {'mean': None, 'std': None, 'n': 0}
            continue
        arr = np.asarray(vals, dtype=float)
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out[k] = {'mean': float(arr.mean()), 'std': std, 'n': int(arr.size)}
    return out


def _dump_cv_results(config, logger, split_mode, subjects, valid_folds,
                     skipped_folds, fold_records):
    """
    落盘交叉验证逐折结果与 mean±std。

    目的：主表每个数字都要能追溯到具体的折与被试，
    且 std 的口径（跨 fold / 跨 seed）必须明确可查。
    """
    results_dir = Path(config['base.results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': config['model.name'],
        'dataset': config['data.dataset'],
        'split_mode': split_mode,
        'frame_type': config['data.frame_type'],
        'seed': config.get('base.seed'),
        'val_mode': config.get('data.val_mode'),
        'num_classes': config.get('data.num_classes'),
        'n_subjects': len(subjects),
        'n_folds_valid': valid_folds,
        'n_folds_skipped': skipped_folds,
        'folds': fold_records,
        'summary': _summarize(fold_records),
    }
    out_file = results_dir / 'cv_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    s = payload['summary']
    logger.info(f"逐折结果已保存: {out_file}")
    if s['UF1']['n'] > 0:
        logger.info(
            f"汇总（跨 fold）: UF1 {s['UF1']['mean']:.4f}±{s['UF1']['std']:.4f} | "
            f"UAR {s['UAR']['mean']:.4f}±{s['UAR']['std']:.4f} | "
            f"Acc {s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}"
        )


def train(config, logger):
    # 创建模型
    model_module = importlib.import_module(f"src.model.{config['model.name'].lower()}")
    model = getattr(model_module, config['model.name'])(config)

    tool.start_common_set(config)
    # 创建训练器
    trainer = Trainer(config, model)

    eval_metrics = None
    # 训练：train() 内部完成「val 选 best + 测试被试单次评估」，并返回测试指标
    # （测试被试不参与模型选择）
    if config.get('train.enabled', False):
        logger.info("训练模式...")
        eval_metrics = trainer.train()

    # 兼容：训练被禁用但仍需评估时，从 checkpoint 评估
    if eval_metrics is None and config.get('eval.enabled', False):
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

            # ---- 剔除"当前标签口径下根本没有样本"的受试者 ----
            # 场景: casme2_3c_megc 剔除 others/fear/sadness 后，目录里 26 个受试者只剩
            # 24 个有样本。不过滤的话，那 2 个被试会各产生一折**空测试集**，
            # evaluator 抛 "没有数据可以评估"，整个 LOSO 在第 10 折中断
            # （2026-09-25 实测踩到，两次 megc 口径 run 均因此只跑完 10/24 折）。
            try:
                from src.data.dataset import MEDataset as _MD
                _present = set(_MD(config=config,
                                   data_dir=config.get("data.data_dir", "dataset"),
                                   split="train", transform=None).subjects)
                _before = len(subjects)
                subjects = [s for s in subjects if s in _present]
                if len(subjects) != _before:
                    logger.info(f"受试者过滤: {_before} → {len(subjects)}"
                                f"（当前标签口径下无样本者已剔除）")
            except Exception as e:
                logger.warning(f"受试者可用性预检失败，沿用目录列表: {e}")

            logger.info(f"\n{'=' * 80}")
            logger.info(f"开始 LOSO 交叉验证（共 {len(subjects)} 个受试者）")
            logger.info(f"{'=' * 80}\n")

            valid_folds = 0
            skipped_folds = 0
            all_metrics = []  # 收集每个 fold 的测试指标
            fold_records = []  # 逐折可追溯记录（test 被试 + 指标）

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
                # 折序号写入 config（arm0，2026-09-25）：metrics.jsonl 的 'fold' 字段读的是
                # config['flag.fold']，而此前全仓只有读、没有写 → 该列恒为 None，导致
                # "折集合是否 = 1..N 连续""区分缺折 vs 折错位"两条断言拿不到数据。
                config['flag.fold'] = sub_idx
                try:
                    eval_metrics = train(config, logger)
                except Exception as e:
                    # 单折异常不应中断整个 LOSO（如某折样本过少导致空验证集）。
                    # 记录并跳过，最终在 cv_results.json 里如实反映跳过折数。
                    logger.warning(f"❌ 该 fold 训练/评估异常，已跳过: {type(e).__name__}: {e}")
                    skipped_folds += 1
                    valid_folds -= 1
                    continue
                if eval_metrics is not None:
                    all_metrics.append(eval_metrics)
                    fold_records.append({
                        'fold': sub_idx,
                        'test_subject': sub,
                        'UF1': eval_metrics.get('UF1'),
                        'UAR': eval_metrics.get('UAR'),
                        'accuracy': eval_metrics.get('accuracy'),
                        'WF1': eval_metrics.get('WF1'),
                    })

            logger.info(f"\n{'=' * 80}")
            logger.info(f"LOSO 验证完成")
            logger.info(f"总 fold 数: {len(subjects)}")
            logger.info(f"有效 fold 数: {valid_folds}")
            logger.info(f"跳过 fold 数: {skipped_folds}")
            logger.info(f"有效率: {valid_folds / len(subjects) * 100:.1f}%")
            logger.info(f"{'=' * 80}\n")

            # 输出 LOSO 平均结果
            print_cv_average_report(all_metrics, config, config['data.split_mode'])

            # 逐折结果落盘（数字可追溯、std 口径明确）
            _dump_cv_results(config, logger, 'loso', subjects,
                             valid_folds, skipped_folds, fold_records)

        elif config['data.split_mode'] == '5fold':
            all_metrics = []  # 收集每个 fold 的测试指标
            fold_records = []
            for idx in range(5):
                config['data.fold_idx'] = idx
                eval_metrics = train(config, logger)
                if eval_metrics is not None:
                    all_metrics.append(eval_metrics)
                    fold_records.append({
                        'fold': idx + 1,
                        'fold_idx': idx,
                        'UF1': eval_metrics.get('UF1'),
                        'UAR': eval_metrics.get('UAR'),
                        'accuracy': eval_metrics.get('accuracy'),
                        'WF1': eval_metrics.get('WF1'),
                    })

            # 输出 5-fold 平均结果
            print_cv_average_report(all_metrics, config, config['data.split_mode'])
            _dump_cv_results(config, logger, '5fold', list(range(5)),
                             5, 0, fold_records)

        else:  # fixed
            train(config, logger)

        # 记录训练标记
        train_flag[train_key] = True
        with open(train_flag_file, 'w') as f:
            json.dump(train_flag, f)
    except Exception as e:
        logger.error(f"模型|参数: {train_key} 训练失败: {str(e)}, 报错信息: {str(e)}", exc_info=True)

