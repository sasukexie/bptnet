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

from datetime import datetime

from main import main

if __name__ == '__main__':
    # 初始配置，优先级：*>cmd>init>model.yml>common.yml，*结尾标记最优先
    models = ['BPTNet']  # ['HTNet','LongShortActionFuseNet','MMNet','VITSRMCL','AlexNet','GoogLeNet','VGG16'] ###
    # > **MER：** 采用 CASME II、SAMM、SMIC、DFME 四个识别数据集。其中 CASME II、SAMM、SMIC 的评测同时遵循 MEGC2019 复合协议（3类标签映射），DFME 独立评测。
    datasets = ['casme2_7c', 'casme2_5c', 'casme2_4c', 'casme2_3c',
                # 'casme_8c', 'casme_4c', 'casme_3c',
                # 'casme_sq_8c', 'casme_sq_4c', 'casme_sq_3c',
                'mmew_7c', 'mmew_5c', 'mmew_3c',
                'samm_8c', 'samm_5c', 'samm_3c',
                'smic_hs_3c',
                'dfme_7c', 'dfme_5c', 'dfme_3c']  # ### 调试用: ['casme2_7c', 'mmew_7c', 'samm_8c', 'smic_hs_3c', 'dfme_7c']
    # Rich 数据集专用覆盖 (224×224 + 适配 batch_size)
    dataset_overrides = {
        'other': {'image_size': 224}
    }
    init_config = {
        'base': {
            'output_dir': 'saved',  # saved
        },
        'model': {
            'name': None
        },  # HTNet
        'data': {
            'data_dir': 'dataset',
            'dataset': None,
            'cache': {
                'enabled': True,  # 是否启用磁盘缓存
                'refresh': False  # true=强制重新扫描并覆盖缓存（数据集更新或逻辑变更时开启一次）
            },
            'num_classes': None,  # 分类数量
            'batch_size*': 32,
            'split_mode': "fixed",  # fixed,5fold,loso
            'frame_type': "apex",
            # frame_type: apex(单帧RGB) / flow(光流onset→apex) / rgb_triplet(三帧堆叠) / rgb_flow(RGB+光流) / rgb_dual_flow(RGB+双向光流)
        },
        'train': {
            'enabled': True,
            'epochs': 100,
            'learning_rate': 1e-4,  # 学习率 (5e-5),0.00005,0.001
            'weight_decay': 0.0001,  # 权重衰减 (1e-4)
            'device': 'cuda:0',  # 训练设备
        },
        'eval': {
            'enabled': True,  # True,False
        },
        'flag': {
            'id': datetime.now().strftime("%Y%m%d%H%M%S"),  # id，用于标识当前实验
            'train_flag': '1',  # 训练标记，用于标识当前实验所属批次
        }
    }
    split_modes = ['fixed', '5fold']  # ['fixed','5fold','loso'] ### 调试采用 fixed, 基线对比采用 5fold 或 loso
    frame_types = ['apex', 'flow', 'rgb_triplet', 'rgb_flow', 'rgb_dual_flow']  # apex(单帧RGB) / flow(光流onset→apex) / rgb_triplet(三帧堆叠) / rgb_flow(RGB+光流) / rgb_dual_flow(RGB+双向光流PhaseAware)
    model_to_frame_types = {
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
    for model in models:
        for dataset in datasets:
            for split_mode in split_modes:
                for frame_type in frame_types:
                    if model not in model_to_frame_types:
                        print(f"模型 {model} 不支持此 frame_type={frame_type} 选择")
                        continue
                    elif frame_type not in model_to_frame_types[model]:
                        print(f"模型 {model} 不适合此 frame_type={frame_type} 训练")
                        continue
                    init_config['model']['name'] = model
                    init_config['data']['dataset'] = dataset
                    init_config['data']['num_classes'] = int(dataset.split('_')[-1].replace('c', ''))
                    init_config['data']['split_mode'] = split_mode
                    init_config['data']['frame_type'] = frame_type
                    # 应用 rich 数据集专用覆盖 (image_size, batch_size)
                    if dataset in dataset_overrides:
                        init_config['data'].update(dataset_overrides[dataset])
                    else:
                        init_config['data'].update(dataset_overrides['other'])

                    main(init_config)
