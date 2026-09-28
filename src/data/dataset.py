"""数据集加载器
负责加载和预处理微表情数据
支持 fixed, 5fold, loso 三种划分模式

仅支持 rich 格式:
  dataset/<name>/rgb/  + flow/ + flow_ao/
"""

import csv
import hashlib
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import KFold
from src.utils.logger import logger
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ========== 磁盘缓存方法 ==========
def _load_disk_cache(cache_path: Path):
    """从磁盘加载缓存，失败返回 None"""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def _save_disk_cache(cache_path: Path, data):
    """保存缓存到磁盘"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


# ========== 类别合并配置 ==========
# 支持将原始细粒度类别合并为粗粒度类别，用于不同分类粒度的实验
# 键名 = 目标数据集名称 (如 casme2_5c)，值包含源数据集和标签映射
#
# 合并依据:
#   - 3类: MEGC2019 协议 — Positive(Happiness), Negative(其余), Surprise
#   - 4类: 原作者 note.txt 推荐 — Positive, Negative, Surprise, Others
#   - 5类: 论文主流标准 (15/17篇) — Happiness, Surprise, Disgust, Repression, Others
#
# casme2_7c 目录标签 = 官方 Objective Class（AU 客观类）：
#   已用 CASME2-ObjectiveClasses.xlsx 逐条核验：254/254 完全一致，0 例外。
#   目录 id 约定：0=repression, 1=sadness, 2=others, 3=disgust, 4=surprise,
#                 5=happiness, 6=fear（下方 *_3c/4c/5c 的 label_map 即按此书写）。
# CASME II 官方另发布 Estimated Emotion（主观估计类），MEGC2019 与主流文献按后者
#   构建三分类（negative←disgust/repression/sadness/fear、positive←happiness、
#   surprise←surprise；others 仅单库口径并入 negative）。
# 两套口径**并存**：casme2_* 用目录标签(=Objective Class)；casme2_*_ee 用 Estimated
#   Emotion（标签取自 casme2_labels_official.csv，见 load_casme2_official_labels()）。
#   → 既不影响既有跑完/在跑的实验，又天然构成一个"标注口径鲁棒性"消融。
CASME2_EE_7C = {"repression": 0, "sadness": 1, "others": 2, "disgust": 3,
                "surprise": 4, "happiness": 5, "fear": 6}
CASME2_EE_5C = {"happiness": 0, "surprise": 1, "disgust": 2, "repression": 3,
                "others": 4, "sadness": 4, "fear": 4}
CASME2_EE_4C = {"repression": 1, "sadness": 0, "others": 1, "disgust": 0,
                "surprise": 3, "happiness": 2, "fear": 0}
CASME2_EE_3C = {"happiness": 1, "surprise": 2, "others": 0, "disgust": 0,
                "repression": 0, "sadness": 0, "fear": 0}

# casme_8c   原始标签: 0=tense, 1=disgust, 2=repression, 3=surprise, 4=happiness, 5=sadness, 6=fear, 7=contempt
# casme_sq_8c 原始标签: 0=happiness, 1=disgust, 2=anger, 3=surprise, 4=fear, 5=sadness, 6=helpless, 7=pain

DATASET_MERGE_CONFIG = {
    # ---- CASME II 变体 (源: casme2_7c) ----
    'casme2_5c': {
        'source': 'casme2_7c',
        'num_classes': 5,
        'class_names': ['happiness', 'surprise', 'disgust', 'repression', 'others'],
        'label_map': {0: 3, 1: 4, 2: 4, 3: 2, 4: 1, 5: 0, 6: 4},
        #   repression→3, sadness→4, others→4, disgust→2, surprise→1, happiness→0, fear→4
    },
    'casme2_4c': {
        'source': 'casme2_7c',
        'num_classes': 4,
        'class_names': ['negative', 'others', 'positive', 'surprise'],
        'label_map': {0: 1, 1: 0, 2: 1, 3: 0, 4: 3, 5: 2, 6: 0},
        #   repression→others, sadness→negative, others→others, disgust→negative,
        #   surprise→surprise, happiness→positive, fear→negative
    },
    'casme2_3c': {
        'source': 'casme2_7c',
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
        'label_map': {0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 1, 6: 0},
        #   MEGC2019: positive←happiness, negative←其余, surprise←surprise
    },

    # ---- CASME II 变体 (标签=官方 Estimated Emotion，MEGC 口径) ----
    # 与 casme2_* 同结构，仅标签来源不同；两者对比即"标注口径"消融。
    # 目录标签已被写乱（见 load_casme2_official_labels 上方的说明），
    # 正式实验请优先使用 *_ee 口径。
    'casme2_7c_ee': {
        'source': 'casme2_7c',
        'num_classes': 7,
        'class_names': ['repression', 'sadness', 'others', 'disgust',
                        'surprise', 'happiness', 'fear'],
        'label_map': {i: i for i in range(7)},   # 兜底：无官方记录时用目录标签
        'emotion_map': CASME2_EE_7C,
    },
    'casme2_5c_ee': {
        'source': 'casme2_7c',
        'num_classes': 5,
        'class_names': ['happiness', 'surprise', 'disgust', 'repression', 'others'],
        'label_map': {0: 3, 1: 4, 2: 4, 3: 2, 4: 1, 5: 0, 6: 4},
        'emotion_map': CASME2_EE_5C,
    },
    'casme2_4c_ee': {
        'source': 'casme2_7c',
        'num_classes': 4,
        'class_names': ['negative', 'others', 'positive', 'surprise'],
        'label_map': {0: 1, 1: 0, 2: 1, 3: 0, 4: 3, 5: 2, 6: 0},
        'emotion_map': CASME2_EE_4C,
    },
    'casme2_3c_ee': {
        'source': 'casme2_7c',
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
        'label_map': {0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 1, 6: 0},
        # 官方 145 条构成 = negative 88 / positive 32 / surprise 25，与 MEGC 一致
        'emotion_map': CASME2_EE_3C,
    },

    # ---- CASME II 3 类 · MEGC 严口径（剔除 others/fear/sadness）----
    # 动机（2026-09-25 口径审计）: casme2_3c_ee 把 others(99)+fear(2)+sadness(7) 全并入
    # negative，得到 254 样本 / negative 占 77.6%；而 HTNet 用 145 样本(neg 88/pos 32/sur 25)、
    # EDMDBN 用 150（明确剔除 Others）、FRL-DGT 原文写 "Others are omitted"。
    # 同一个数据集名，实际是两个难度不同的任务 → 分数不可直接比较。
    # 本变体用于"把口径对齐文献"的对照实验。
    # 排除机制: emotion_map 里不列 others/fear/sadness → 这些样本 label 保持 None →
    #           落到 label_map 分支，而 label_map 为空 {} → 被显式排除（见 L341-344）。
    'casme2_3c_megc': {
        'source': 'casme2_7c',
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
        'label_map': {},
        'emotion_map': {'disgust': 0, 'repression': 0, 'happiness': 1, 'surprise': 2},
        #   预期构成: negative = disgust 63 + repression 27 = 90, positive 32, surprise 25 → 147
    },

    # ---- MEGC2019 跨库（CD）合并集：CASME II + SAMM + SMIC-HS ----
    # 协议：三库各自的 3 类子集（negative/positive/surprise）合并为统一标签空间，
    #       再对**全体被试**做留一（LOSO）。与单库 SDE 是两个不同的协议，不可混称。
    # [!] 三个源的标签空间天然一致（已核实）：
    #       casme2_7c   → casme2_3c_megc 口径（官方 Estimated Emotion）
    #       samm_8c     → samm_3c 的 label_map（neg=0, pos=1, sur=2）
    #       smic_hs_3c  → 原生目录标签即 0=negative, 1=positive, 2=surprise
    # [!] 被试名天然不冲突：sub01..(CASME II) / 006..(SAMM) / s01..(SMIC)，
    #     故无需加前缀命名空间。若将来引入命名重叠的库，**必须**加前缀，
    #     否则 LOSO 会把不同库的同名被试当成同一个被试（致命错误）。
    'megc_cd_3c': {
        'sources': [
            {'source': 'casme2_7c',
             'label_map': {},
             'emotion_map': {'disgust': 0, 'repression': 0, 'happiness': 1,
                             'surprise': 2}},
            {'source': 'samm_8c',
             'label_map': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 2}},
            {'source': 'smic_hs_3c'},           # 原生 3 类，无需映射
        ],
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
    },

    # ---- CASME 变体 (源: casme_8c) ----
    'casme_4c': {
        'source': 'casme_8c',
        'num_classes': 4,
        'class_names': ['negative', 'others', 'positive', 'surprise'],
        'label_map': {0: 1, 1: 0, 2: 1, 3: 3, 4: 2, 5: 0, 6: 0, 7: 0},
        #   tense→others, disgust→negative, repression→others, surprise→surprise,
        #   happiness→positive, sadness→negative, fear→negative, contempt→negative
    },
    'casme_3c': {
        'source': 'casme_8c',
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
        'label_map': {0: 0, 1: 0, 2: 0, 3: 2, 4: 1, 5: 0, 6: 0, 7: 0},
        #   MEGC2019: positive←happiness, negative←其余, surprise←surprise
    },

    # ---- CAS(ME)^2 变体 (源: casme_sq_8c) ----
    'casme_sq_4c': {
        'source': 'casme_sq_8c',
        'num_classes': 4,
        'class_names': ['negative', 'others', 'positive', 'surprise'],
        'label_map': {0: 2, 1: 0, 2: 0, 3: 3, 4: 0, 5: 0, 6: 1, 7: 1},
        #   happiness→positive, disgust→negative, anger→negative, surprise→surprise,
        #   fear→negative, sadness→negative, helpless→others, pain→others
    },
    'casme_sq_3c': {
        'source': 'casme_sq_8c',
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
        'label_map': {0: 1, 1: 0, 2: 0, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0},
        #   MEGC2019: positive←happiness, negative←其余, surprise←surprise
    },

    # ---- MMEW 变体 (源: mmew_7c) ----
    # mmew_7c 原始: 0=surprise, 1=disgust, 2=others, 3=happiness, 4=fear, 5=sadness, 6=anger
    'mmew_5c': {
        'source': 'mmew_7c',
        'num_classes': 5,
        'class_names': ['happiness', 'surprise', 'disgust', 'negative', 'others'],
        'label_map': {0: 1, 1: 2, 2: 4, 3: 0, 4: 3, 5: 3, 6: 3},
        #   surprise→surprise, disgust→disgust, others→others, happiness→happiness,
        #   fear+sadness+anger→negative
    },
    'mmew_3c': {
        'source': 'mmew_7c',
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
        'label_map': {0: 2, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0},
        #   MEGC2019: positive←happiness, negative←其余(含others), surprise←surprise
    },

    # ---- SAMM 变体 (源: samm_8c) ----
    # samm_8c 原始: 0=anger, 1=contempt, 2=disgust, 3=fear, 4=happiness, 5=other, 6=sadness, 7=surprise
    'samm_5c': {
        'source': 'samm_8c',
        'num_classes': 5,
        'class_names': ['happiness', 'surprise', 'disgust', 'negative', 'other'],
        'label_map': {0: 3, 1: 3, 2: 2, 3: 3, 4: 0, 5: 4, 6: 3, 7: 1},
        #   anger+contempt+fear+sadness→negative, disgust→disgust, happiness→happiness,
        #   other→other, surprise→surprise
    },
    'samm_3c': {
        'source': 'samm_8c',
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
        'label_map': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 2},
        #   MEGC2019: positive←happiness, negative←其余(含other), surprise←surprise
    },

    # ---- DFME 变体 (源: dfme_7c) ----
    # dfme_7c 原始: 0=anger, 1=contempt, 2=disgust, 3=fear, 4=happiness, 5=sadness, 6=surprise
    'dfme_5c': {
        'source': 'dfme_7c',
        'num_classes': 5,
        'class_names': ['happiness', 'surprise', 'disgust', 'fear', 'negative'],
        'label_map': {0: 4, 1: 4, 2: 2, 3: 3, 4: 0, 5: 4, 6: 1},
        #   anger+contempt+sadness→negative, disgust→disgust, fear→fear,
        #   happiness→happiness, surprise→surprise
    },
    'dfme_3c': {
        'source': 'dfme_7c',
        'num_classes': 3,
        'class_names': ['negative', 'positive', 'surprise'],
        'label_map': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 2},
        #   MEGC2019: positive←happiness, negative←其余, surprise←surprise
    },
}


# ========== CASME II 官方标注表（权威标签）==========
# 背景（已逐条核实）：casme2_7c 的**类目录标签在早期预处理中被写乱**，
# 与官方标注表 CASME2-coding-20140508.xlsx 比对：
#   7 类口径一致率仅 37%：类2=disgust(96%)、类4=surprise(100%)、类5=happiness(92%)、
#   类6=fear(100%) 基本干净，而类0/1/3 是混杂桶（如类3 = others 64 + disgust 34）；
#   3 类口径错误率 8.3%，且**全部是 positive/surprise 被误标成 negative**
#   —— 系统性抬高多数类，直接压低 UAR，会污染所有 casme2 单库结论。
# 因此 casme2 的标签一律改以官方表为准（SAMM / SMIC 的目录标签已核实无误，不受影响）。
# 启用方式：给 merge config 加 'emotion_map'（官方 emotion → 目标类号），
#           即上方的 casme2_*_ee 变体；不带 emotion_map 的 casme2_* 仍走目录标签。
_CASME2_LABELS_PATH = Path(__file__).resolve().parent / "casme2_labels_official.csv"
_CASME2_LABELS_CACHE = None


def _mer_norm_key(s) -> str:
    """归一化 key：非字母数字丢弃 + 数字段去前导零（EP09f/EP09、ne_8/ne_08 等价）。"""
    toks = [t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if t]
    return "".join((t.lstrip("0") or "0") if t.isdigit() else t for t in toks)


def load_casme2_official_labels() -> Dict[str, str]:
    """读取 CASME II 官方标签：(subject, sample) 归一化键 -> 官方情绪（7 类）。

    同时登记「带数据集前缀」的变体键，便于直接用本地文件名 stem 命中。
    """
    global _CASME2_LABELS_CACHE
    if _CASME2_LABELS_CACHE is None:
        out: Dict[str, str] = {}
        if _CASME2_LABELS_PATH.exists():
            with open(_CASME2_LABELS_PATH, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    subj, samp, emo = r["subject"], r["sample"], r["emotion7"]
                    for k in (_mer_norm_key(f"{subj}_{samp}"),
                              _mer_norm_key(f"casme2_{subj}_{samp}")):
                        out[k] = emo
            logger.info(f"CASME II 官方标签表: {len(out)} 键 "
                        f"({_CASME2_LABELS_PATH.name})")
        else:
            logger.error(f"CASME II 官方标签表缺失: {_CASME2_LABELS_PATH}")
        _CASME2_LABELS_CACHE = out
    return _CASME2_LABELS_CACHE


class MEDataset(Dataset):
    """
    微表情数据集类 — 仅支持 rich 格式

    frame_type:
      - apex:           仅加载 rgb/ 下的 apex 帧
      - flow:           加载 flow/ 下的 .npy 光流 (onset→apex)
      - rgb_triplet:    加载 rgb/ 下的 onset+apex+offset 三帧
      - rgb_flow:       同时加载 rgb/ apex 帧 + flow/ 光流
      - rgb_dual_flow:  同时加载 rgb/ apex 帧 + flow/ 光流 + flow_ao/ 光流 (相位感知)
    """

    _global_cache = {}

    def __init__(self, config, data_dir: str, split: str = "train",
                 transform=None, subject_list: Optional[List[str]] = None,
                 use_cache: bool = True,
                 sample_indices: Optional[List[int]] = None):
        self.config = config
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.dataset_name = config.get("data.dataset", "unknown")
        self.frame_type = config.get("data.frame_type", "apex")
        self.image_size = config.get("data.image_size", 224)
        self.use_cache = use_cache
        # 光流预处理开关（详见 _load_flow_npy 内的实测说明）
        #   flow_norm   : "p99"(默认, 按本样本 p99 归一化) / "std" / "fixed20"(历史)
        #   flow_detrend: 是否移除整体刚性位移（头动）
        self.flow_norm = config.get("data.flow_norm", "p99")
        self.flow_detrend = bool(config.get("data.flow_detrend", True))
        # 光流所在子目录（默认即历史目录名，保证向后兼容）。
        # 背景（2026-09-25 审计）：rgb/ 是 224×224 对齐帧，而 flow/ 是 239~312 的可变尺寸，
        # 说明旧光流是在**未对齐**原帧上算的 → 混入头动/背景且与 RGB 空间不对应。
        # 新提取的"对齐版"光流放在 flow_align/ 与 flow_ao_align/，用这两个开关切换，
        # 不动旧目录（正在跑的 run 仍读旧数据）。
        self.flow_subdir = config.get("data.flow_subdir", "flow")
        self.flow_ao_subdir = config.get("data.flow_ao_subdir", "flow_ao")
        # 双相时序增强（对照 arXiv:2510.15466 "Phase-Aware Temporal Augmentation"）：
        #   微表情是"缓升—峰值—快降"的非线性过程，onset→apex 与 apex→offset 虽区间不同、
        #   但**标签相同**，可互为增强样本。开启后每个样本随机取其中一个相位，
        #   等价于把训练集扩一倍（论文报告 CASME-II UF1 +6.9%、SAMM +7.6%）。
        #   仅对 frame_type='flow' 生效；找不到配对的 _flow_ao.npy 时自动退回原相位。
        self.phase_aug = bool(config.get("data.phase_aug", False))

        # ---- 解析类别合并配置 ----
        self._merge_cfg = DATASET_MERGE_CONFIG.get(self.dataset_name)
        # 多源（跨库 CD）：merge cfg 用 'sources' 列表代替单个 'source'。
        #   每个源带自己的 label_map / emotion_map —— **必须逐源使用**，
        #   否则三库的原始标签编号会互相套用（例如 SAMM 的 4=positive 被
        #   按 CASME II 的 4=surprise 解释），这是致命且不会报错的错标。
        self._sources = (self._merge_cfg or {}).get('sources') or None
        if self._sources:
            self._source_name = '+'.join(s['source'] for s in self._sources)
        else:
            self._source_name = (self._merge_cfg['source'] if self._merge_cfg
                                 else self.dataset_name)
        self._src_cfg = {}
        if self._sources:
            for s in self._sources:
                self._src_cfg[s['source']] = (s.get('label_map'),
                                              s.get('emotion_map'))
        else:
            self._src_cfg[self._source_name] = (
                self._merge_cfg['label_map'] if self._merge_cfg else None,
                (self._merge_cfg or {}).get('emotion_map'),
            )
        # [!] 多源配置没有顶层 label_map（映射写在各 source 里），用 .get 避免 KeyError
        self._label_map = (self._merge_cfg or {}).get('label_map')
        # 官方 Estimated Emotion → 目标类号；存在时以官方标签为准（见文件头说明）
        self._emotion_map = (self._merge_cfg or {}).get('emotion_map')

        # ---- 检测目录格式 (仅 rich, 使用源数据集目录) ----
        _check_dirs = ([s['source'] for s in self._sources] if self._sources
                       else [self._source_name])
        for _sn in _check_dirs:
            main_path = self.data_dir / _sn
            if not (main_path / "rgb").exists() and not (main_path / "flow").exists():
                raise FileNotFoundError(
                    f"数据集必须是 rich 格式 (包含 rgb/ 或 flow/ 子目录): {main_path}"
                )

        # ---- 确定扫描路径和文件模式 ----
        self.all_samples = self._load_all_samples()

        # ---- 按 subject 过滤 + 类别合并 ----
        self.samples = []
        self.labels = []
        self.subjects = []

        target_subjects = set(subject_list) if subject_list else None
        _c2 = load_casme2_official_labels()
        n_c2, n_c2_miss, n_dropped = 0, 0, 0
        for s in self.all_samples:
            if target_subjects is not None and s["subject"] not in target_subjects:
                continue
            # ---- casme2：以官方标注表为准（类目录标签在早期预处理中被写乱，
            #      7 类一致率仅 37%；3 类错误率 8.3% 且全部是 positive/surprise
            #      被误标成 negative，会系统性抬高多数类、压低 UAR）----
            # [!] 多源（CD）时必须取**该样本所属源**的映射，不能用全局的：
            #     三个库的原始标签编号含义不同，套错就是静默错标。
            src = s.get('_src') or self._source_name
            lm, em = self._src_cfg.get(src, (None, None))
            label = None
            if em and src == 'casme2_7c':
                emo = _c2.get(_mer_norm_key(self._sample_stem(s)))
                if emo is not None and emo in em:
                    label = em[emo]
                    n_c2 += 1
                else:
                    n_c2_miss += 1
            if label is None:
                if lm is not None:
                    # label_map 中未列出的原始标签 = 显式排除该样本
                    if s["label"] not in lm:
                        n_dropped += 1
                        continue
                    label = lm[s["label"]]
                else:
                    label = s["label"]
            self.samples.append(s)
            self.labels.append(label)
            self.subjects.append(s["subject"])

        if self._emotion_map is not None:
            _msg = f"casme2 官方标签覆盖 {n_c2} 条"
            if n_c2_miss:
                _msg += f"；{n_c2_miss} 条无官方记录（沿用目录标签）"
            if n_dropped:
                _msg += f"；显式排除 {n_dropped} 条"
            logger.info(_msg)

        # ---- 样本级子集（可选）：val_mode='stratified' 时按类别分层切分 train/val ----
        if sample_indices is not None:
            self.samples = [self.samples[i] for i in sample_indices]
            self.labels = [self.labels[i] for i in sample_indices]
            self.subjects = [self.subjects[i] for i in sample_indices]

        if self._merge_cfg:
            logger.info(
                f"类别合并: {self.dataset_name} ← {self._source_name}, "
                f"合并后 {self._merge_cfg['num_classes']} 类 "
                f"{self._merge_cfg['class_names']}"
            )

        logger.info(
            f"加载 {self.split} 集: {len(self.samples)} 个样本, "
            f"受试者数: {len(set(self.subjects))}"
        )

    # ========== 目录格式检测 & 缓存键 ==========
    @staticmethod
    def _sample_stem(sample: dict) -> str:
        """取样本文件 stem（去掉 _flow/_flow_ao/.npy/.jpg 等后缀），用于查官方表。"""
        p = sample.get("path")
        if isinstance(p, dict):
            p = next(iter(p.values()), "")
        name = Path(p).name
        for suf in ("_flow_ao.npy", "_flow.npy", "_apex.jpg", "_onset.jpg",
                    "_offset.jpg", ".npy", ".jpg", ".png"):
            if name.endswith(suf):
                return name[: -len(suf)]
        return name

    def _phase_ao_path(self, sample: dict) -> Optional[Path]:
        """由 onset→apex 光流路径推出 apex→offset 光流路径（双相时序增强用）。

        真实目录布局（**不是同目录改名**，2026-09-25 踩过）:
            <src>/flow/subXX/<class>/<name>_flow.npy
            <src>/flow_ao/subXX/<class>/<name>_flow_ao.npy
        即 flow_ao 是与 flow **同级的另一个目录**，需要替换路径中的 subdir 段，
        同时把文件名后缀 _flow.npy 换成 _flow_ao.npy；两处都要改。
        找不到返回 None（调用方退回原相位，保证不因缺文件报错）。
        """
        p = sample.get("path")
        if isinstance(p, dict) or p is None:
            return None
        p = Path(p)
        name = p.name
        if not name.endswith("_flow.npy"):
            return None
        # 从右往左定位 flow_subdir 段并替换为 flow_ao_subdir
        parts = list(p.parts)
        try:
            k = len(parts) - 1 - parts[::-1].index(self.flow_subdir)
        except ValueError:
            return None
        parts[k] = self.flow_ao_subdir
        cand = Path(*parts).with_name(name[: -len("_flow.npy")] + "_flow_ao.npy")
        return cand if cand.exists() else None

    def _aug_hflip(self) -> bool:
        """本次是否施加水平翻转（**RGB 与所有光流共享同一决策**）。

        只在 train 且 `data.augmentation.enabled` 时按 `horizontal_flip` 的概率决策。
        val/test 恒为 False（与 _build_transform 的 is_train 门保持一致）。
        """
        if self.split != "train":
            return False
        aug = self.config.get("data.augmentation", {}) or {}
        if not aug.get("enabled", False) or not aug.get("horizontal_flip", False):
            return False
        return random.random() < 0.5

    def _hflip(self, x):
        """水平翻转张量/张量元组，**光流物理正确**（u 分量取负）。

        [!] 必须用 torch.flip（返回新张量），不可原地改写：
            _load_flow_npy 会把处理后的张量放进 _flow_cache 复用，原地写会永久污染缓存，
            使"随机增强"退化成"每文件一个固定伪翻转"，并在 num_workers>0 时跨 worker 不一致。
        """
        if isinstance(x, (list, tuple)):
            return type(x)(self._hflip(i) for i in x)
        if not torch.is_tensor(x) or x.dim() < 3:
            return x
        y = torch.flip(x, dims=[-1])
        if x.shape[0] == 2:              # (2,H,W) = 光流 (u,v) → 水平位移反向
            y = y.clone()
            y[0] = -y[0]
        return y



    def _load_all_samples(self) -> List[dict]:
        """
        加载 all_samples：内存缓存 → 磁盘缓存 → 扫描
        磁盘缓存通过 config.data.cache.enabled / refresh 控制
        """
        cache_key = self.config['data.cache_file']
        # 1) 内存缓存（进程内最快）
        if self.use_cache and cache_key in MEDataset._global_cache:
            logger.info(f"使用内存缓存: {cache_key}")
            return MEDataset._global_cache[cache_key]

        disk_enabled = self.config.get("data.cache.enabled", True)
        refresh = self.config.get("data.cache.refresh", False)
        cache_path = Path(cache_key)

        # 2) 磁盘缓存（跨进程/跨运行持久化）
        if self.use_cache and disk_enabled and not refresh:
            loaded = _load_disk_cache(cache_path)
            if loaded is not None:
                logger.info(
                    f"使用磁盘缓存 ({len(loaded)} 样本): {cache_path.name}"
                )
                MEDataset._global_cache[cache_key] = loaded
                return loaded

        # 3) 扫描磁盘
        samples = self._scan()

        # 4) 保存缓存
        if self.use_cache:
            if disk_enabled:
                _save_disk_cache(cache_path, samples)
                logger.info(
                    f"扫描完成，已保存磁盘缓存 ({len(samples)} 样本): {cache_path.name}"
                )
            MEDataset._global_cache[cache_key] = samples
        else:
            logger.info(f"扫描完成 ({len(samples)} 样本)，未启用缓存")

        return samples

    # ========== 扫描逻辑 ==========

    def _scan(self) -> List[dict]:
        """调度 rich 格式扫描。

        单源（历史行为）：扫 data_dir/<source>。
        多源（跨库 CD）：依次扫每个源目录，并给样本打上 `_src` 标记 ——
        后续标签解析靠它取各自源的 label_map / emotion_map。
        """
        if not self._sources:
            return self._scan_rich(self.data_dir / self._source_name)
        out = []
        for s in self._sources:
            for smp in self._scan_rich(self.data_dir / s['source']):
                smp = dict(smp)
                smp['_src'] = s['source']
                out.append(smp)
        logger.info(f"[CD] 多源扫描: " +
                    ", ".join(f"{s['source']}" for s in self._sources) +
                    f" → 合计 {len(out)} 个样本")
        return out

    def _scan_rich(self, main_path: Path) -> List[dict]:
        """
        扫描丰富格式数据集

        目录结构:
          rgb/subXX/{class_id}/{name}_apex.jpg
          flow/subXX/{class_id}/{name}_flow.npy
        """
        ft = self.frame_type

        if ft == "flow":
            return self._scan_flat(main_path / self.flow_subdir, "*.npy", ext=".npy")
        elif ft in ("rgb_triplet", "rgb_flow", "rgb_dual_flow"):
            # 以 rgb/ 目录的 apex 帧为锚点
            rgb_dir = main_path / "rgb"
            if not rgb_dir.exists():
                raise FileNotFoundError(f"RGB目录不存在: {rgb_dir}")
            return self._scan_rgb_anchor(rgb_dir, main_path, ft)
        else:
            # apex 模式: 仅扫描 apex 帧
            if (main_path / "rgb").exists():
                return self._scan_flat(main_path / "rgb", "*_apex.jpg")
            elif (main_path / "flow").exists():
                return self._scan_flat(main_path / "flow", "*.npy", ext=".npy")
            raise FileNotFoundError(f"丰富数据集缺少 rgb/ 或 flow/: {main_path}")

    def _build_class_mapping(self, base_path: Path) -> Dict[str, int]:
        """
        扫描 base_path 下所有 subject 中的 class 目录名，建立 name→int 映射。
        支持数字目录名 (0, 1, 2...) 和字符串目录名 (disgust, happiness...)。

        映射策略:
          - 全部为数字 → 直接使用目录名的 int 值
          - 包含字符串 → 按字母序排序，分配 0-based ID
        """
        class_names = set()
        for subject_dir in base_path.iterdir():
            if not subject_dir.is_dir():
                continue
            for class_dir in subject_dir.iterdir():
                if class_dir.is_dir():
                    class_names.add(class_dir.name)

        if not class_names:
            return {}

        # 尝试全部解析为 int
        try:
            return {name: int(name) for name in class_names}
        except ValueError:
            pass

        # 回退：按字母序排序建立映射
        return {name: idx for idx, name in enumerate(sorted(class_names))}

    def _scan_rgb_anchor(self, rgb_dir: Path, main_path: Path,
                         frame_type: str) -> List[dict]:
        """以 rgb/ 的 apex 帧为锚，构建 rgb_triplet 或 rgb_flow 或 rgb_dual_flow 样本"""
        samples = []
        flow_dir = main_path / self.flow_subdir
        flow_ao_dir = main_path / self.flow_ao_subdir
        # 静默剔除计数（#14，2026-09-25）：这三处 continue 原本没有任何日志，
        # 导致"同一数据集在不同 frame_type 下的样本集合不同"完全无声 ——
        # 例如 mmew_7c 只有 166/300 个 flow_ao，rgb_dual_flow 会静默丢掉 134 条，
        # 拿到的 N 与 flow 不同，使 flow↔dual 的对比不在同一批样本上。
        drops = {"no_onset_offset": 0, "no_flow": 0, "no_flow_ao": 0}

        # 建立 class_name → int label 映射 (支持字符串 & 数字目录名)
        class_map = self._build_class_mapping(rgb_dir)

        for subject_dir in sorted(rgb_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject = subject_dir.name

            for class_dir in sorted(subject_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                if class_name not in class_map:
                    continue
                label = class_map[class_name]

                for apex_file in sorted(class_dir.glob("*_apex.jpg")):
                    # 从 apex 文件名推导 basename
                    apex_stem = apex_file.stem  # e.g. casme2_sub01_EP02_01f_apex
                    base = apex_stem[:-len("_apex")]  # e.g. casme2_sub01_EP02_01f

                    sample = {
                        "label": label,
                        "subject": subject,
                    }

                    if frame_type == "rgb_triplet":
                        onset_file = class_dir / f"{base}_onset.jpg"
                        offset_file = class_dir / f"{base}_offset.jpg"
                        if not onset_file.exists() or not offset_file.exists():
                            drops["no_onset_offset"] += 1
                            continue
                        sample["path"] = {
                            "onset": onset_file,
                            "apex": apex_file,
                            "offset": offset_file,
                        }

                    elif frame_type == "rgb_flow":
                        # flow 路径: flow/subject/{class_name}/{base}_flow.npy
                        flow_file = (flow_dir / subject / class_name /
                                     f"{base}_flow.npy")
                        if not flow_file.exists():
                            drops["no_flow"] += 1
                            continue
                        sample["path"] = {
                            "rgb": apex_file,
                            "flow": flow_file,
                        }

                    elif frame_type == "rgb_dual_flow":
                        # 双流光流: flow/ + flow_ao/
                        flow_oa_file = (flow_dir / subject / class_name /
                                        f"{base}_flow.npy")
                        flow_ao_file = (flow_ao_dir / subject / class_name /
                                        f"{base}_flow_ao.npy")
                        if not flow_oa_file.exists() or not flow_ao_file.exists():
                            drops["no_flow_ao"] += 1
                            continue
                        sample["path"] = {
                            "rgb": apex_file,
                            "flow_oa": flow_oa_file,
                            "flow_ao": flow_ao_file,
                        }

                    samples.append(sample)

        # 让"静默剔除"出声（#14）：N 会随 frame_type 变化，跨 frame_type 对比前必须固定样本交集
        if any(drops.values()):
            logger.warning(
                f"[样本剔除] 数据集 {self.dataset_name} frame_type={frame_type} "
                f"可用样本 {len(samples)}；因缺模态文件静默跳过: "
                f"缺 onset/offset={drops['no_onset_offset']}、缺 flow={drops['no_flow']}、"
                f"缺 flow_ao={drops['no_flow_ao']}。"
                f"[!] 不同 frame_type 的样本集合可能不同（rgb_dual_flow 需要两条光流都齐全），"
                f"N 不可跨 frame_type 直接比较；比较前必须取样本交集。")

        return samples

    def _scan_flat(self, base_path: Path, glob_pattern: str,
                   ext: str = ".jpg") -> List[dict]:
        """通用扁平扫描"""
        samples = []

        if not base_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {base_path}")

        # 建立 class_name → int label 映射 (支持字符串 & 数字目录名)
        class_map = self._build_class_mapping(base_path)

        for subject_dir in sorted(base_path.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_name = subject_dir.name

            for class_dir in subject_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                if class_name not in class_map:
                    continue
                label = class_map[class_name]

                for f in class_dir.glob(glob_pattern):
                    samples.append({
                        "path": f,
                        "label": label,
                        "subject": subject_name,
                    })

        return samples

    # ========== 数据加载 ==========

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        label = self.labels[idx]  # 使用已合并的标签
        ft = self.frame_type
        # 水平翻转：在数据集层**统一决策一次**，保证 RGB 与全部光流拿到同一决策。
        # （若把 hflip 放在 transform 里，只会翻 RGB、光流不翻，多模态下会产生自相矛盾的样本。）
        do_flip = self._aug_hflip()

        if ft == "flow":
            # 双相时序增强：随机在 onset→apex 与 apex→offset 之间二选一（同标签）
            if self.phase_aug and random.random() < 0.5:
                ao = self._phase_ao_path(sample)
                if ao is not None:
                    s2 = dict(sample)
                    s2["path"] = ao
                    out = self._load_flow(s2)
                    return (self._hflip(out) if do_flip else out), label
            out = self._load_flow(sample)
        elif ft == "rgb_triplet":
            out = self._load_rgb_triplet(sample)
        elif ft == "rgb_flow":
            out = self._load_rgb_flow(sample)
        elif ft == "rgb_dual_flow":
            out = self._load_rgb_dual_flow(sample)
        else:
            # apex 等其余模式: 单张 RGB
            out = self._load_single_rgb(sample["path"])
        return (self._hflip(out) if do_flip else out), label

    def _cached_rgb(self, path) -> "Image.Image":
        """带缓存的 RGB 解码（返回 PIL Image，增强仍在 __getitem__ 中随机施加）。

        MER 数据集极小（CASME II 仅 254 样本），但训练要跑上百 epoch；
        每次 __getitem__ 都重新 Image.open+JPEG 解码是 dataloader 的主要瓶颈。
        缓存解码结果后每张图只需解码一次（增强的随机性不受影响）。
        """
        if not hasattr(self, "_rgb_cache"):
            self._rgb_cache = {}
        key = str(path)
        img = self._rgb_cache.get(key)
        if img is None:
            img = Image.open(key).convert("RGB")
            self._rgb_cache[key] = img
        return img

    def _load_single_rgb(self, path: Path) -> torch.Tensor:
        img = self._cached_rgb(path)
        if self.transform:
            img = self.transform(img)
        return img

    def _load_rgb_triplet(self, sample: dict) -> torch.Tensor:
        """加载 onset+apex+offset, 堆叠为 (9, H, W)"""
        paths = sample["path"]
        frames = []
        for key in ("onset", "apex", "offset"):
            img = self._cached_rgb(paths[key])
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        return torch.cat(frames, dim=0)  # (9, H, W)

    def _load_flow(self, sample: dict) -> torch.Tensor:
        """加载光流 .npy, 转换为 (2, H, W) tensor, 归一化到 [-1, 1]"""
        return self._load_flow_npy(sample["path"])

    def _load_rgb_flow(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载 RGB apex 帧 + 光流"""
        paths = sample["path"]
        # RGB
        rgb = self._cached_rgb(paths["rgb"])
        if self.transform:
            rgb = self.transform(rgb)

        # Flow
        flow = self._load_flow_npy(paths["flow"])
        return rgb, flow

    def _load_rgb_dual_flow(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """加载 RGB apex 帧 + onset→apex 光流 + apex→offset 光流 (相位感知三流)"""
        paths = sample["path"]
        # RGB
        rgb = self._cached_rgb(paths["rgb"])
        if self.transform:
            rgb = self.transform(rgb)

        # 双流光流
        flow_oa = self._load_flow_npy(paths["flow_oa"])
        flow_ao = self._load_flow_npy(paths["flow_ao"])
        return rgb, flow_oa, flow_ao

    def _load_flow_npy(self, path: Path) -> torch.Tensor:
        """加载光流 .npy，返回归一化后的 (2, H, W) tensor

        带进程内缓存：光流不含任何随机增强，同一文件在各 epoch 的结果完全相同，
        缓存后每文件只读一次（此前每个 epoch 重复 np.load + 归一化是 flow 模式
        dataloader 的主要瓶颈，也是该模式此前强制 num_workers=0 的根因）。
        """
        if not hasattr(self, "_flow_cache"):
            self._flow_cache = {}
        key = str(path)
        cached = self._flow_cache.get(key)
        if cached is not None:
            return cached

        flow = np.load(key)
        # 自适应通道布局：不同数据集的 npy 布局并不一致
        #   casme2: flow / flow_ao 均为 HWC = (H, W, 2)
        #   samm  : flow 为 HWC，但 flow_ao 已是 CHW = (2, H, W)
        # 原实现无条件 transpose((2,0,1))，遇到已是 CHW 的文件会错转成 (H, W, 2)，
        # 再被 interpolate 按 4D 解释成 (1, H, W, 2)→ 通道维变成 H，最终触发
        # "expected 3 channels, but got 224" 的报错（samm 任务即因此失败）。
        if flow.ndim != 3:
            raise ValueError(f"光流维度异常: {path} shape={flow.shape}")
        if flow.shape[-1] == 2:                    # HWC → CHW
            flow = np.transpose(flow, (2, 0, 1))
        elif flow.shape[0] == 2:                   # 已是 CHW，直接使用
            pass
        else:
            raise ValueError(
                f"无法判定的光流布局: {path} shape={flow.shape}"
                f"（末维/首维均不是 2）")
        flow = torch.from_numpy(np.ascontiguousarray(flow)).float()

        # ---- (1) 去趋势：移除整体刚性位移 ----
        # 实测 casme2_7c 的 flow / flow_ao，其「空间均值幅值 / 平均幅值」达
        # 0.56 / 0.61（单个样本最高 0.96）—— 即过半能量是头/整体平移。
        # 头动是**受试者特有、与情绪无关**的成分，会直接破坏跨被试泛化
        # （同被试内部仍能拟合，正是"val 尚可、LOSO 崩"的典型成因）。
        if self.flow_detrend:
            flow[0].sub_(flow[0].mean())
            flow[1].sub_(flow[1].mean())

        # ---- (2) 归一化 ----
        # 历史实现 clamp(-20, 20) / 20 的除数与数据量级不匹配：实测 |flow| 典型
        # 0.96、p99 3.93、最大 5.24 —— clamp 几乎从不触发，`/20` 的唯一效果是把
        # 有效动态范围压掉约 5 倍（p99 仅剩 0.20）。在冻结的 ImageNet BN 下，
        # 该支路输出会趋近常数，运动信息基本被抹平。
        if self.flow_norm == "fixed20":
            flow = torch.clamp(flow, -20.0, 20.0) / 20.0
        elif self.flow_norm == "std":
            scale = float(flow.std().item()) or 1.0
            flow = torch.clamp(flow / max(scale, 1e-6), -1.0, 1.0)
        else:  # "p99"（默认）：按本样本 |flow| 的 99 分位归一化，用满动态范围
            scale = float(torch.quantile(flow.abs().flatten(), 0.99).item())
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = float(flow.abs().max().item()) or 1.0
            flow = torch.clamp(flow / scale, -1.0, 1.0)

        if flow.shape[1] != self.image_size or flow.shape[2] != self.image_size:
            flow = torch.nn.functional.interpolate(
                flow.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        self._flow_cache[key] = flow
        return flow


# ========== 辅助函数 ==========

def get_all_subjects(config: Dict) -> List[str]:
    """获取所有受试者 ID (仅 rich 格式，支持合并数据集)"""
    data_dir = Path(config.get("data.data_dir", "dataset"))
    dataset_name = config.get("data.dataset", "")
    frame_type = config.get("data.frame_type", "apex")

    # 解析源数据集 (合并数据集指向源目录)
    merge_cfg = DATASET_MERGE_CONFIG.get(dataset_name)
    # 多源（跨库 CD）：被试集合是各源被试的并集（已核实命名不冲突，见 merge cfg 注释）
    sources = (merge_cfg or {}).get('sources') or None
    if sources:
        source_names = [s['source'] for s in sources]
    else:
        source_names = [merge_cfg['source'] if merge_cfg else dataset_name]

    subjects = []
    for source_name in source_names:
        main_path = data_dir / source_name
        if not main_path.exists():
            logger.warning(f"数据集目录不存在: {main_path}")
            continue

        scan_path = main_path / ("flow" if frame_type == "flow" else "rgb")
        if not scan_path.exists():
            logger.error(f"扫描路径不存在: {scan_path}")
            continue

        subs = sorted([d.name for d in scan_path.iterdir() if d.is_dir()])
        logger.info(f"数据集 {dataset_name} (源: {source_name}): 找到 {len(subs)} 个受试者")
        subjects.extend(subs)

    subjects = sorted(set(subjects))
    if sources:
        logger.info(f"[CD] 跨库合并 {dataset_name}: 合计 {len(subjects)} 个受试者")
    return subjects


def _build_transform(config: Dict, is_train: bool):
    """构建数据增强 pipeline"""
    from torchvision import transforms

    aug_cfg = config.get("data.augmentation", {})
    image_size = config.get("data.image_size", 224)

    ops = [transforms.Resize((image_size, image_size))]

    if is_train and aug_cfg["enabled"]:
        # [!] 水平翻转**不在这里做**（此前的实现只在 RGB 路径做，光流不翻 → 模态不一致）：
        #     self.transform 只作用于 RGB 路径（_load_single_rgb/_load_rgb_flow/
        #     _load_rgb_dual_flow 里对 rgb 调 transform），而光流走 .npy 分支、
        #     完全不经过 transform。于是开启 hflip 后会有 50% 的样本是
        #     "RGB 已镜像、光流未镜像" —— 同一事件的两种模态自相矛盾，等于主动注入
        #     错误监督；现已统一在 __getitem__ 中对 RGB 与两条光流同时翻转（u 取负）。
        #     现在统一在 __getitem__ 里决策一次（见 _aug_hflip / _hflip），
        #     再用 torch.flip 同时作用于 RGB 与全部光流（光流另需 u 分量取负）。
        if aug_cfg.get("rotation_degrees", 0) > 0 or aug_cfg.get("color_jitter", 0) > 0:
            logger.warning(
                "[增强] rotation / color_jitter 仅作用于 RGB 支路、不作用于光流；"
                "在多模态（rgb_flow / rgb_dual_flow）下会造成**模态不对称**。"
                "生产配置这两项为 0，请勿直接开启；如需开启请一并实现光流侧几何变换。")
        if aug_cfg.get("rotation_degrees", 0) > 0:
            ops.append(transforms.RandomRotation(
                degrees=aug_cfg["rotation_degrees"]))
        if aug_cfg.get("color_jitter", 0) > 0:
            cj = aug_cfg["color_jitter"]
            ops.append(transforms.ColorJitter(
                brightness=cj, contrast=cj))

    ops.append(transforms.ToTensor())

    # Normalize: 光流不参与 RGB normalize
    ft = config.get("data.frame_type", "apex")
    if ft != "flow":
        use_imagenet_norm = config.get("data.use_imagenet_norm", False)
        if use_imagenet_norm:
            # ImageNet 标准归一化 (预训练模型对齐)
            ops.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            # 传统归一化 [-1, 1]
            ops.append(transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    if is_train and aug_cfg["enabled"]:
        if aug_cfg.get("random_erasing", 0) > 0:
            ops.append(transforms.RandomErasing(p=aug_cfg["random_erasing"]))

    return transforms.Compose(ops)


def _stable_hash(text: str) -> str:
    """与 PYTHONHASHSEED 无关的稳定哈希（跨进程/跨机器可复现）"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _pick_val_subjects(all_subjects: List[str], train_subjects: List[str],
                       fold_anchor, num_val: int = 1) -> Tuple[List[str], List[str]]:
    """
    折内留出验证被试。

    目的：epoch 选择 / 早停 / 超参固定必须发生在与测试被试
    完全隔离的验证集上；测试被试只在训练结束后评估一次。

    - 确定性：以 fold_anchor（LOSO=测试被试名 / 5fold=fold_idx）在 all_subjects
      中的位置为起点，在 train_subjects 上轮换取 val，不依赖 PYTHONHASHSEED，
      同一 fold 恒定，跨机器可复现。
    - 轮换均匀：N 折 LOSO 下每个被试恰好轮一次 val（val 覆盖全部被试）。
    - 硬保证 val ∩ test = ∅（val 只从 train_subjects 中取）。
    """
    if num_val <= 0 or len(train_subjects) <= 1:
        return train_subjects, []
    n = len(train_subjects)
    num_val = min(num_val, n - 1)
    try:
        offset = all_subjects.index(fold_anchor)
    except (ValueError, TypeError):
        offset = 0
    # 等间隔抽样：使 val 被试在训练池中分散（类别/条件覆盖更均匀），
    # 且当 step*num_val ≈ n 时接近不重复地覆盖整个列表。
    # 单被试 val 样本过少（如 CASME II 仅 ~13 个）且类别高度不均，
    # 会让 val UF1 被多数类主导（实测 Acc=0.89 而 UF1=0.31），不可用作模型选择。
    step = max(1, n // num_val)
    idxs = []
    for j in range(num_val):
        i = (offset + j * step) % n
        if i not in idxs:
            idxs.append(i)
    val_subjects = [train_subjects[i] for i in idxs]
    val_set = set(val_subjects)
    train_subjects = [s for s in train_subjects if s not in val_set]
    return train_subjects, val_subjects


def _stratified_split_indices(labels: List[int], val_ratio: float,
                              seed: int) -> Tuple[List[int], List[int]]:
    """
    按类别分层把训练池切分为 train / val 索引（确定性、可复现）。

    动机：
      MER 每个被试只有 ~10 个样本（CASME II 约 254/26），且类别高度不均
      （3 类下 negative 约占七成）。若用单个被试作验证集，val 的类别分布会
      被多数类主导，UF1/UAR 完全失真（实测出现 Val Acc=1.0 而 val UF1=0.333）；
      若改用多个被试作 val，又会把训练数据砍掉 30%，对小数据集代价过大。

      分层抽样保证 val 的类别比例与全集一致 → 指标可靠、可作 epoch 选择依据；
      测试被试始终不参与（只在训练结束后评估一次）。
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for y in sorted(by_class):
        idxs = sorted(by_class[y])
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_ratio))
        # 每类至少给 train 留 1 个样本
        n_val = max(0, min(n_val, len(idxs) - 1)) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return sorted(train_idx), sorted(val_idx)


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 train / val / test 三个 DataLoader。

    三级划分:
      - train: 参与梯度更新
      - val:   仅用于 epoch 选择 / 早停 / 超参固定，与测试被试严格隔离
      - test:  训练结束后只评估一次

    data.val_mode:
      - "inner_subject" (默认): 从训练被试中再留 data.num_val_subjects 个作验证集
      - "none": 不设独立验证集，val_loader 退化为 test_loader
                （仅用于复现历史实验，正式结果禁用）
    """
    data_dir = config.get("data.data_dir", "dataset")
    batch_size = config.get("data.batch_size", 64)
    num_workers = config.get("data.num_workers", 4)
    pin_memory = config.get("data.pin_memory", True)

    all_subjects = get_all_subjects(config)
    split_mode = config.get("data.split_mode", "fixed")
    val_mode = config.get("data.val_mode", "inner_subject")

    # ---- 划分受试者 ----
    fold_anchor = 0
    if split_mode == "loso":
        test_subject = config.get("data.test_subject")
        if not test_subject:
            raise ValueError("LOSO 模式必须提供 test_subject")
        test_subjects = [test_subject]
        train_subjects = [s for s in all_subjects if s != test_subject]
        fold_anchor = test_subject
        logger.info(f"LOSO 模式: 测试受试者 -> {test_subject}")

    elif split_mode == "5fold":
        fold_idx = config.get("data.fold_idx")
        if fold_idx is None:
            raise ValueError("5-Fold 模式必须提供 fold_idx (0~4)")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(kf.split(all_subjects))
        train_idx, test_idx = folds[fold_idx]
        train_subjects = [all_subjects[i] for i in train_idx]
        test_subjects = [all_subjects[i] for i in test_idx]
        fold_anchor = fold_idx
        logger.info(f"5-Fold 模式: Fold {fold_idx + 1}/5")

    else:  # fixed
        # 固定划分: 按受试者稳定 hash 确定 train/test，确保可复现
        # （原实现用内置 hash()，受 PYTHONHASHSEED 影响、跨进程不可复现，已修正）
        seed = config.get('base.seed', 1024)
        ratio = config.get("data.fixed_split", [0.8, 0.2])[0]
        hashed = sorted(all_subjects, key=lambda s: _stable_hash(f"{seed}_{s}"))
        split_idx = int(len(hashed) * ratio)
        train_subjects = hashed[:split_idx]
        test_subjects = hashed[split_idx:]
        if not test_subjects:
            test_subjects = [hashed[-1]]
            train_subjects = hashed[:-1]

    # ---- 验证集构造（test 被试始终严格隔离）----
    strat_train_idx = strat_val_idx = None
    if val_mode == "stratified":
        # val 取自训练被试内部的按类别分层样本（test 被试完全不参与）
        val_subjects = train_subjects      # 样本级切分在下方执行
    elif val_mode == "inner_subject":
        train_subjects, val_subjects = _pick_val_subjects(
            all_subjects, train_subjects, fold_anchor,
            num_val=config.get("data.num_val_subjects", 1),
        )
    else:
        val_subjects = []
        logger.warning(
            "val_mode 未设独立验证集：epoch 选择将发生在测试被试上"
            "（乐观偏置），仅用于复现历史实验，正式结果禁用"
        )

    logger.info(f"训练集受试者 ({len(train_subjects)}个): {train_subjects[:5]}...")
    logger.info(f"验证集受试者 ({len(val_subjects)}个): {val_subjects}")
    logger.info(f"测试集受试者 ({len(test_subjects)}个): {test_subjects}")

    # ---- 构建 transform (训练集带增强, 验证/测试集无增强) ----
    train_transform = _build_transform(config, is_train=True)
    test_transform = _build_transform(config, is_train=False)

    # ---- 数据集构建 ----
    if val_mode == "stratified":
        # 先取训练池的类别标签（仅扫描，不解码图像），再做分层切分
        pool = MEDataset(
            config=config, data_dir=data_dir, split="train",
            transform=None, subject_list=train_subjects,
        )
        strat_train_idx, strat_val_idx = _stratified_split_indices(
            pool.labels,
            val_ratio=config.get("data.val_ratio", 0.2),
            seed=config.get("base.seed", 1024),
        )
        logger.info(
            f"分层验证集: 训练池 {len(pool.labels)} 样本 → "
            f"train {len(strat_train_idx)} / val {len(strat_val_idx)}"
            f"（val 类别比例与训练池一致；测试被试完全隔离）"
        )
        del pool
        train_dataset = MEDataset(
            config=config, data_dir=data_dir, split="train",
            transform=train_transform, subject_list=train_subjects,
            sample_indices=strat_train_idx,
        )
        val_dataset = MEDataset(
            config=config, data_dir=data_dir, split="test",
            transform=test_transform, subject_list=train_subjects,
            sample_indices=strat_val_idx,
        )
    else:
        train_dataset = MEDataset(
            config=config, data_dir=data_dir, split="train",
            transform=train_transform, subject_list=train_subjects,
        )
        val_dataset = None
        if val_subjects:
            # 复用 test 的 transform（无增强）与扫描路径，仅按 subject 过滤
            val_dataset = MEDataset(
                config=config, data_dir=data_dir, split="test",
                transform=test_transform, subject_list=val_subjects,
            )

    test_dataset = MEDataset(
        config=config, data_dir=data_dir, split="test",
        transform=test_transform, subject_list=test_subjects,
    )

    # 多 worker 设置
    # 原实现把 flow / rgb_dual_flow 强制设为 0 worker，理由记为"npy load 有 GIL 问题"。
    # 该理由不成立：DataLoader 的多 worker 是**多进程**（不受 GIL 限制）；此前的主要
    # 开销是每个 epoch 重复 np.load + JPEG 解码，现已通过进程内缓存消除（每文件只
    # 读一次）。因此放开多 worker 可显著提升吞吐（实测 flow 模式单进程时每折 >7 分钟）。
    # 仅在 CPU 或 CUDA 不可用时回退为单进程。
    ft = config.get("data.frame_type", "apex")
    device = config.get("train.device", "cuda")
    is_cpu = device == "cpu" or (isinstance(device, str) and not device.startswith("cuda"))
    if is_cpu or not torch.cuda.is_available():
        actual_workers = 0
    else:
        actual_workers = max(0, int(num_workers))

    # ---- 训练集采样器（可选类别平衡过采样）----
    # MER 类别极度不平衡（CASME II 3c 实测 train: negative 165 / positive 19 /
    # surprise 12，少数类合计仅 12%），仅靠 loss 类权重不足以让模型学到少数类
    # （UAR 长期贴近随机 1/num_classes）。开启后按类别逆频率过采样，使各类别
    # 每 epoch 的曝光量相当，并配合 mixup 降低重复样本的过拟合风险。
    # persistent_workers=True 是多 worker 下的关键：默认 False 时每个 epoch 都会
    # 重建 worker 进程，MEDataset 的进程内缓存（_rgb_cache / _flow_cache）随之丢失
    # → 每 epoch 重新解码全部图像 + np.load，反而比单进程更慢（实测某机器上
    # 从 ~6 分钟/折退化到 22 分钟/折）。
    # 多 worker 下的可复现性：显式给每个 worker 播种 random / numpy。
    # PyTorch 只保证 worker 内 torch 的种子由主进程推导，**不会**重置 Python
    # `random` 与 `numpy` —— 而数据增强（水平翻转、擦除等）走的正是 random。
    # 后果有两个，实测都已出现：
    #   (1) 同一 seed 两次运行结果不同（同一配置实测 UF1 0.4800 vs 0.4577）；
    #   (2) worker 由 fork 继承主进程同一 random 状态 → 各 worker 的"随机"增强
    #       完全相关，等于削弱了增强的多样性。
    def _seed_worker(worker_id):
        import random as _random
        s = torch.initial_seed() % (2 ** 32)
        _random.seed(s)
        np.random.seed(s)

    loader_kw = dict(
        num_workers=actual_workers,
        pin_memory=pin_memory,
        persistent_workers=actual_workers > 0,
        prefetch_factor=2 if actual_workers > 0 else None,
        worker_init_fn=_seed_worker if actual_workers > 0 else None,
        generator=torch.Generator().manual_seed(int(config.get("base.seed", 1024))),
    )

    if config.get("data.balanced_sampling", False):
        counts = Counter(train_dataset.labels)
        sample_w = [1.0 / counts[y] for y in train_dataset.labels]
        sampler = WeightedRandomSampler(
            sample_w, num_samples=len(train_dataset), replacement=True)
        logger.info(
            f"类别平衡采样已启用: {dict(sorted(counts.items()))} "
            f"(逆频率过采样，各类曝光量拉平)"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, **loader_kw)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **loader_kw)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kw)
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, **loader_kw)
    else:
        # 兼容历史行为：无独立 val 时退化为 test_loader（正式结果禁用）
        val_loader = test_loader

    return train_loader, val_loader, test_loader
