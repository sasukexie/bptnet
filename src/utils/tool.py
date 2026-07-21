import os

import psutil
import setproctitle
import torch
from src.data.dataset import get_all_subjects, MEDataset
from src.trainer import Trainer
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger, create_timestamped_log
from torchvision import transforms


def start_common_set(config=None):
    try:
        model_name = config['model']['name']
        dataset_name = config['data']['dataset']
        # set proc
        setproctitle.setproctitle("MER@" + model_name + "." + dataset_name)
        # check dir log,saved
        if not os.path.exists("./log"):
            os.makedirs("./log")
        if not os.path.exists("./saved"):
            os.makedirs("./saved")

        # 获取当前进程对象
        current_process = psutil.Process()
        print("current_process:", current_process)
    except Exception as e:
        pass


def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: torch.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device,
             or "CPU" when running on a non-CUDA device.
    """
    try:
        if device is None or device.type != "cuda":
            return "CPU"
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return "{:.2f} G/{:.2f} G".format(reserved, total)
    except (RuntimeError, AssertionError, AttributeError):
        return "CPU"


def check_fold_validity(config, train_subjects, test_subject):
    """
    检查LOSO fold是否有效（训练集是否包含测试集的所有类别）

    Args:
        config: 配置对象
        train_subjects: 训练集受试者列表
        test_subject: 测试集受试者

    Returns:
        tuple: (is_valid, missing_classes)
            - is_valid: bool, fold是否有效
            - missing_classes: list, 训练集缺失的类别列表
    """

    # 临时构建数据集以检查类别分布
    temp_config = config.copy() if isinstance(config, dict) else dict(config)
    temp_config['data.frame_type'] = temp_config.get('data.frame_type', 'apex')
    temp_config['data.image_size'] = temp_config.get('data.image_size', 224)

    # 简单的transform用于加载数据
    transform = transforms.Compose([
        transforms.Resize((temp_config['data.image_size'], temp_config['data.image_size'])),
        transforms.ToTensor(),
    ])

    try:
        # 构建训练集和测试集
        train_dataset = MEDataset(
            config=temp_config,
            data_dir=temp_config.get('data.data_dir', 'dataset'),
            split='train',
            transform=transform,
            subject_list=train_subjects,
            use_cache=False
        )

        test_dataset = MEDataset(
            config=temp_config,
            data_dir=temp_config.get('data.data_dir', 'dataset'),
            split='test',
            transform=transform,
            subject_list=[test_subject],
            use_cache=False
        )

        # 获取类别集合
        train_labels = set()
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            train_labels.add(label)

        test_labels = set()
        for i in range(len(test_dataset)):
            _, label = test_dataset[i]
            test_labels.add(label)

        # 检查缺失类别
        missing_classes = sorted(list(test_labels - train_labels))
        is_valid = len(missing_classes) == 0

        return is_valid, missing_classes

    except Exception as e:
        # 如果检查失败，默认认为有效（避免阻塞训练）
        print(f"警告: 检查fold有效性时出错: {e}，默认认为有效")
        return True, []

