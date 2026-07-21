"""数据集加载器
负责加载和预处理微表情数据
支持 fixed, 5fold, loso 三种划分模式

仅支持 rich 格式:
  dataset/<name>/rgb/  + flow/ + flow_ao/
"""

import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import KFold
from src.utils.logger import logger
from torch.utils.data import Dataset, DataLoader


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
# casme2_7c 原始标签: 0=repression, 1=sadness, 2=others, 3=disgust, 4=surprise, 5=happiness, 6=fear
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
                 use_cache: bool = True):
        self.config = config
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.dataset_name = config.get("data.dataset", "unknown")
        self.frame_type = config.get("data.frame_type", "apex")
        self.image_size = config.get("data.image_size", 224)
        self.use_cache = use_cache

        # ---- 解析类别合并配置 ----
        self._merge_cfg = DATASET_MERGE_CONFIG.get(self.dataset_name)
        self._source_name = self._merge_cfg['source'] if self._merge_cfg else self.dataset_name
        self._label_map = self._merge_cfg['label_map'] if self._merge_cfg else None

        # ---- 检测目录格式 (仅 rich, 使用源数据集目录) ----
        main_path = self.data_dir / self._source_name
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
        for s in self.all_samples:
            if target_subjects is None or s["subject"] in target_subjects:
                self.samples.append(s)
                label = self._label_map[s["label"]] if self._label_map else s["label"]
                self.labels.append(label)
                self.subjects.append(s["subject"])

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
        """调度 rich 格式扫描 (使用源数据集目录)"""
        return self._scan_rich(self.data_dir / self._source_name)

    def _scan_rich(self, main_path: Path) -> List[dict]:
        """
        扫描丰富格式数据集

        目录结构:
          rgb/subXX/{class_id}/{name}_apex.jpg
          flow/subXX/{class_id}/{name}_flow.npy
        """
        ft = self.frame_type

        if ft == "flow":
            return self._scan_flat(main_path / "flow", "*.npy", ext=".npy")
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
        flow_dir = main_path / "flow"
        flow_ao_dir = main_path / "flow_ao"

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
                            continue
                        sample["path"] = {
                            "rgb": apex_file,
                            "flow_oa": flow_oa_file,
                            "flow_ao": flow_ao_file,
                        }

                    samples.append(sample)

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

        if ft == "flow":
            return self._load_flow(sample), label
        elif ft == "rgb_triplet":
            return self._load_rgb_triplet(sample), label
        elif ft == "rgb_flow":
            rgb, flow = self._load_rgb_flow(sample)
            return (rgb, flow), label
        elif ft == "rgb_dual_flow":
            rgb, flow_oa, flow_ao = self._load_rgb_dual_flow(sample)
            return (rgb, flow_oa, flow_ao), label
        else:
            # apex 等其余模式: 单张 RGB
            return self._load_single_rgb(sample["path"]), label

    def _load_single_rgb(self, path: Path) -> torch.Tensor:
        img = Image.open(str(path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _load_rgb_triplet(self, sample: dict) -> torch.Tensor:
        """加载 onset+apex+offset, 堆叠为 (9, H, W)"""
        paths = sample["path"]
        frames = []
        for key in ("onset", "apex", "offset"):
            img = Image.open(str(paths[key])).convert("RGB")
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
        rgb = Image.open(str(paths["rgb"])).convert("RGB")
        if self.transform:
            rgb = self.transform(rgb)

        # Flow
        flow = self._load_flow_npy(paths["flow"])
        return rgb, flow

    def _load_rgb_dual_flow(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """加载 RGB apex 帧 + onset→apex 光流 + apex→offset 光流 (相位感知三流)"""
        paths = sample["path"]
        # RGB
        rgb = Image.open(str(paths["rgb"])).convert("RGB")
        if self.transform:
            rgb = self.transform(rgb)

        # 双流光流
        flow_oa = self._load_flow_npy(paths["flow_oa"])
        flow_ao = self._load_flow_npy(paths["flow_ao"])
        return rgb, flow_oa, flow_ao

    def _load_flow_npy(self, path: Path) -> torch.Tensor:
        """加载光流 .npy，返回归一化后的 (2, H, W) tensor"""
        flow = np.load(str(path))  # (H, W, 2), float32
        flow = np.transpose(flow, (2, 0, 1))  # (2, H, W)
        flow = torch.from_numpy(flow).float()
        flow = torch.clamp(flow, -20.0, 20.0) / 20.0
        if flow.shape[1] != self.image_size or flow.shape[2] != self.image_size:
            flow = torch.nn.functional.interpolate(
                flow.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return flow


# ========== 辅助函数 ==========

def get_all_subjects(config: Dict) -> List[str]:
    """获取所有受试者 ID (仅 rich 格式，支持合并数据集)"""
    data_dir = Path(config.get("data.data_dir", "dataset"))
    dataset_name = config.get("data.dataset", "")
    frame_type = config.get("data.frame_type", "apex")

    # 解析源数据集 (合并数据集指向源目录)
    merge_cfg = DATASET_MERGE_CONFIG.get(dataset_name)
    source_name = merge_cfg['source'] if merge_cfg else dataset_name

    main_path = data_dir / source_name
    if not main_path.exists():
        logger.warning(f"数据集目录不存在: {main_path}")
        return []

    scan_path = main_path / ("flow" if frame_type == "flow" else "rgb")
    if not scan_path.exists():
        logger.error(f"扫描路径不存在: {scan_path}")
        return []

    subjects = sorted([d.name for d in scan_path.iterdir() if d.is_dir()])
    logger.info(f"数据集 {dataset_name} (源: {source_name}): 找到 {len(subjects)} 个受试者")
    return subjects


def _build_transform(config: Dict, is_train: bool):
    """构建数据增强 pipeline"""
    from torchvision import transforms

    aug_cfg = config.get("data.augmentation", {})
    image_size = config.get("data.image_size", 224)

    ops = [transforms.Resize((image_size, image_size))]

    if is_train and aug_cfg["enabled"]:
        if aug_cfg.get("horizontal_flip", False):
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
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


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """创建训练/测试 DataLoader"""
    data_dir = config.get("data.data_dir", "dataset")
    batch_size = config.get("data.batch_size", 64)
    num_workers = config.get("data.num_workers", 4)
    pin_memory = config.get("data.pin_memory", True)

    all_subjects = get_all_subjects(config)
    split_mode = config.get("data.split_mode", "fixed")

    # ---- 划分受试者 ----
    if split_mode == "loso":
        test_subject = config.get("data.test_subject")
        if not test_subject:
            raise ValueError("LOSO 模式必须提供 test_subject")
        test_subjects = [test_subject]
        train_subjects = [s for s in all_subjects if s != test_subject]
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
        logger.info(f"5-Fold 模式: Fold {fold_idx + 1}/5")

    else:  # fixed
        # 固定划分: 按受试者 hash 确定 train/test，确保可复现
        seed = config.get('base.seed', 1024)
        ratio = config.get("data.fixed_split", [0.8, 0.2])[0]
        random.seed(seed)
        # 按 hash 排序受试者，每次划分一致
        hashed = sorted(all_subjects, key=lambda s: hash(f"{seed}_{s}"))
        split_idx = int(len(hashed) * ratio)
        train_subjects = hashed[:split_idx]
        test_subjects = hashed[split_idx:]
        if not test_subjects:
            test_subjects = [hashed[-1]]
            train_subjects = hashed[:-1]

    logger.info(f"训练集受试者 ({len(train_subjects)}个): {train_subjects[:5]}...")
    logger.info(f"测试集受试者 ({len(test_subjects)}个): {test_subjects}")

    # ---- 构建 transform (训练集带增强, 测试集无增强) ----
    train_transform = _build_transform(config, is_train=True)
    test_transform = _build_transform(config, is_train=False)

    train_dataset = MEDataset(
        config=config, data_dir=data_dir, split="train",
        transform=train_transform, subject_list=train_subjects,
    )
    test_dataset = MEDataset(
        config=config, data_dir=data_dir, split="test",
        transform=test_transform, subject_list=test_subjects,
    )

    # flow / dual_flow 模式不适合多 worker (npy load 有 GIL 问题)
    # CPU 模式或 Windows spawn 模式下关闭多 worker，避免 multiprocessing 问题
    ft = config.get("data.frame_type", "apex")
    device = config.get("train.device", "cuda")
    is_cpu = device == "cpu" or (isinstance(device, str) and not device.startswith("cuda"))
    if is_cpu or not torch.cuda.is_available():
        actual_workers = 0
    elif ft in ("flow", "rgb_dual_flow"):
        actual_workers = 0
    else:
        actual_workers = num_workers

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=actual_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=actual_workers, pin_memory=pin_memory,
    )

    return train_loader, test_loader
