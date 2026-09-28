"""
微表情识别系统 - 主入口脚本（手稿协议跑批）

本入口按手稿《Bidirectional Phase-aware Transformer Network for Micro-Expression
Recognition》(The Visual Computer 回稿) 的实验协议组织跑批：

  - split_mode = loso            留一被试交叉验证（手稿 §3.6.4 / §4.1）
  - val_mode   = inner_subject   折内再留 1 个被试作验证集，只用于 epoch 选择 / 早停
                                 （测试被试不参与模型选择）
  - seeds      = 1024/2048/4096  主结果 3 个随机种子，报 mean ± std（手稿 §3.6.4）
  - frame_type = rgb_dual_flow   BPTNet 主实验输入表示（双向稠密光流 + apex RGB）
  - datasets   = CASME II 3c/5c、SAMM 5c、SMIC-HS 3c（SDE 主表）

输出目录（每个 seed 独立，避免互相覆盖）:
  saved/model/BPTNet/<dataset>/loso_rgb_dual_flow_seed<seed>/
    ├── results/cv_results.json      逐折指标 + mean±std（数字可追溯）
    ├── results/val_curve.json       验证曲线（供手稿 Fig.2）
    └── checkpoints/best_model.pth   val-best 权重

使用方法:
    python run_main.py

调整跑批范围：修改下方「跑批配置」区块的 SEEDS / DATASETS 即可
（例如先跑 pilot：SEEDS = [1024]，DATASETS = ['casme2_3c']）。
"""

import json
import os
import re
from datetime import datetime

from main import main
from src.data.dataset import DATASET_MERGE_CONFIG


def _env_list(name, default):
    """从环境变量读取列表（逗号分隔），未设置时返回默认值"""
    raw = os.environ.get(name)
    if not raw:
        return default
    return [x.strip() for x in raw.split(',') if x.strip()]


def _num_classes(dataset: str) -> int:
    """类别数。

    优先查合并配置——这样 casme2_3c_ee / casme2_5c_ee 这类带后缀的数据集名也能
    正确解析；原实现用 int(name.split('_')[-1].replace('c',''))，遇 'ee' 会直接
    抛 ValueError。
    """
    mc = DATASET_MERGE_CONFIG.get(dataset)
    if mc and mc.get('num_classes'):
        return int(mc['num_classes'])
    m = re.search(r'_(\d+)c', dataset)
    if not m:
        raise ValueError(f"无法从数据集名解析类别数: {dataset}")
    return int(m.group(1))


# ==================================================================
#  消融覆盖注入（环境变量 BPTNET_CFG，JSON）
#  例:
#    BPTNET_CFG='{"_tag":"detrend","data":{"flow_detrend":true}}'
#    BPTNET_CFG='{"_tag":"L2","model":{"transformer_layers":2,"transformer_dim":256}}'
#  "_tag" 会进入结果目录名，保证各消融互不覆盖。
# ==================================================================
_CFG_RAW = json.loads(os.environ.get('BPTNET_CFG', '{}') or '{}')
_TAG_RAW = str(_CFG_RAW.pop('_tag', '') or '')
_VTAG = f"_{_TAG_RAW}" if _TAG_RAW else ""
_OVERRIDES = _CFG_RAW


# ==================================================================
#  跑批配置（手稿协议）
#  可用环境变量临时缩小范围（便于 pilot 与消融）:
#    BPTNET_DATASETS=casme2_3c  BPTNET_SEEDS=1024  BPTNET_FRAME_TYPES=flow
# ==================================================================
# 可用 BPTNET_MODELS 覆盖以跑对比方法（论文 Table 2-4 的 baseline 需要**自己复现**：
# 手稿里那些基线数字是引文值，从未在本仓库验证过，且 SAMM/SMIC 两表呈均匀等差递增，
# 未经本仓库验证。同一协议（LOSO + val 选模型）下自己跑出来的数才可比）。
MODELS = _env_list('BPTNET_MODELS', ['BPTNet'])

# SDE 主表（手稿 Table 2-4）
#  - casme2_3c / casme2_5c : CASME II 的 3 类（MEGC2019 映射）与 5 类
#  - samm_5c               : SAMM 5 类
#  - smic_hs_3c            : SMIC-HS 原生 3 类
DATASETS = _env_list('BPTNET_DATASETS', ['casme2_3c', 'casme2_5c', 'samm_5c', 'smic_hs_3c'])

# 手稿 §3.6.4 / §4.1：主结果 3 个随机种子（消融实验用 [1024]）
SEEDS = [int(s) for s in _env_list('BPTNET_SEEDS', ['1024', '2048', '4096'])]

SPLIT_MODES = ['loso']                # 手稿协议（LOSO）
FRAME_TYPES = _env_list('BPTNET_FRAME_TYPES', ['rgb_dual_flow'])  # BPTNet 主实验输入表示

# 消融 / 对照实验用的 frame_type（需要时替换 FRAME_TYPES）
#   'flow'         : 单向光流（onset→apex）
#   'apex'         : 仅 apex RGB 单帧
#   'rgb_triplet'  : 三帧通道堆叠
#   'rgb_flow'     : RGB + 单向光流
ABLATION_FRAME_TYPES = ['apex', 'flow', 'rgb_triplet', 'rgb_flow', 'rgb_dual_flow']

MODEL_TO_FRAME_TYPES = {
    'BPTNet': ['apex', 'flow', 'rgb_triplet', 'rgb_flow', 'rgb_dual_flow'],
    'MPFNet': ['rgb_triplet'],                  # 简化监督版仅支持 rgb_triplet (3D 主干输入)
    'HTNet': ['apex', 'flow', 'rgb_triplet', 'rgb_flow'],
    'LongShortActionFuseNet': ['rgb_triplet'],
    'MMNet': ['rgb_triplet'],
    'VITSRMCL': ['apex'],
    'AlexNet': ['apex'],
    'GoogLeNet': ['apex'],
    'VGG16': ['apex'],
}


if __name__ == '__main__':
    # 初始配置，优先级：*>cmd>init>model.yml>common.yml，*结尾标记最优先
    init_config = {
        'base': {
            'output_dir': 'saved',  # 输出根目录
            'name': None,           # 循环内按 (模型/数据集/协议/种子) 生成，防止多种子互相覆盖
            'seed': None,           # 循环内注入（手稿 seeds: 1024/2048/4096）
        },
        'model': {
            'name': None
        },
        'data': {
            'data_dir': 'dataset',
            'dataset': None,
            'cache': {
                'enabled': True,    # 是否启用磁盘缓存
                'refresh': False    # true=强制重新扫描并覆盖缓存（数据集更新或逻辑变更时开启一次）
            },
            'num_classes': None,    # 分类数量
            'batch_size*': 16,      # 手稿 §3.6.4: batch size = 16
            # [!] 首轮手稿 §3.6.4 写 224；实测 ~200 样本下严重过拟合，默认改 64。
            # 可用 BPTNET_IMAGE_SIZE 覆盖以做分辨率消融（64 → feature map 仅 2×2）。
            'image_size': int(os.environ.get('BPTNET_IMAGE_SIZE', '64')),
            'split_mode': 'loso',   # 手稿协议
            'val_mode': 'stratified',      # 按类别分层抽 val（测试被试严格隔离）
            'val_ratio': 0.2,
            'frame_type': 'rgb_dual_flow',
            # frame_type: apex(单帧RGB) / flow(光流onset→apex) / rgb_triplet(三帧堆叠)
            #             / rgb_flow(RGB+光流) / rgb_dual_flow(RGB+双向光流)
        },
        'train': {
            'enabled': True,
            'epochs': 40,           # 手稿 §3.6.4
            'learning_rate': 1e-4,  # 手稿 §3.6.4（bptnet.yml 内带 * 者优先，同为 1e-4）
            'weight_decay': 1e-4,
            'device': 'cuda:0',
        },
        'eval': {
            'enabled': True,
        },
        'flag': {
            'id': datetime.now().strftime("%Y%m%d%H%M%S"),  # id，用于标识当前实验批次
            'train_flag': '1',      # 循环内按 seed 改写 → 每个种子独立跳过表，避免被互相跳过
        }
    }

    for seed in SEEDS:
        init_config['base']['seed'] = seed
        # 每个 seed 用独立的 train_flag 文件，避免第 2/3 个种子被"已训练过"跳过。
        # [!] 还必须并入变体标签 _VTAG：跳过表的键是
        #     model_dataset__split_frame（不含超参），否则同一 (数据集, 输入) 下的
        #     不同消融会被互相判定为"已训练过"而整体跳过
        #     （第二波消融就因此 0 折；第一波只是靠并发启动抢在写入标记前才侥幸跑成）。
        init_config['flag']['train_flag'] = f"{seed}{_VTAG}"

        for model in MODELS:
            for dataset in DATASETS:
                for split_mode in SPLIT_MODES:
                    for frame_type in FRAME_TYPES:
                        if model not in MODEL_TO_FRAME_TYPES:
                            print(f"模型 {model} 不支持此 frame_type={frame_type} 选择")
                            continue
                        elif frame_type not in MODEL_TO_FRAME_TYPES[model]:
                            print(f"模型 {model} 不适合此 frame_type={frame_type} 训练")
                            continue

                        init_config['model']['name'] = model
                        init_config['data']['dataset'] = dataset
                        init_config['data']['num_classes'] = _num_classes(dataset)
                        init_config['data']['split_mode'] = split_mode
                        init_config['data']['frame_type'] = frame_type
                        # 注入消融覆盖（BPTNET_CFG）
                        # [!] 必须包含 'eval'：TTA 开关走 eval.tta.*，
                        #     否则 BPTNET_CFG 里写的 eval 段会被静默丢弃（2026-09-25 踩到）。
                        for _sec in ('model', 'data', 'train', 'eval'):
                            if _OVERRIDES.get(_sec):
                                init_config[_sec].update(_OVERRIDES[_sec])
                        # [!] base 段**不能**走上面的整段 update：那会把 base.name /
                        #     base.seed 冲掉。但 base.checkpoint 下的开关（如 keep_folds）
                        #     确实需要逐变体覆盖，故这里做**子键级**合并，只覆盖显式给出的键，
                        #     不动 checkpoint_dir 等其余项。
                        #     （教训：2026-09-26 写 keep_folds 时因为 base 不在上面的白名单里，
                        #       该开关被**静默丢弃**，跑到折数过半才发现一个逐折检查点都没落盘。）
                        _bc = (_OVERRIDES.get('base', {}) or {}).get('checkpoint', {}) or {}
                        for _k, _v in _bc.items():
                            init_config['base'].setdefault('checkpoint', {})[_k] = _v
                        # 输出名包含分辨率与变体标签 → 各消融互不覆盖
                        init_config['base']['name'] = (
                            f"model/{model}/{dataset}/{split_mode}_{frame_type}"
                            f"_img{init_config['data']['image_size']}"
                            f"{_VTAG}_seed{seed}"
                        )

                        main(init_config)
