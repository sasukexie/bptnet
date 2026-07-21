# 配置文件管理说明

## 配置架构

项目采用**分层配置管理**策略，将配置分为两层：

### 1. 公共配置 (common.yml)
包含所有模型共享的通用参数：
- **base**: 实验基本信息（种子、输出目录等）
- **data**: 数据配置（数据集、batch_size、图像尺寸等）
- **train**: 训练配置（epochs、学习率、优化器等）
- **eval**: 评估配置（指标、混淆矩阵等）
- **cross_val**: 交叉验证配置
- **log**: 日志配置
- **visualization**: 可视化配置

### 2. 模型独有配置 (模型名.yml)
每个模型只定义自己特有的参数：
- **model.name**: 模型名称（必须）
- **模型特定参数**: 如patch_size、embed_dim、heads等

## 配置合并逻辑

```
run_main.py 启动时：
1. 加载 common.yml（公共配置）
2. 加载 模型.yml（模型配置）
3. 模型配置覆盖 common.yml 中的同名参数
4. 生成最终配置字典
```

**优先级**：CMD > init_config> 模型配置 > common配置 > 默认值

## 配置文件清单

### common.yml (公共配置)
所有模型共享，修改会影响所有模型。

**主要参数**：
```yaml
data:
  dataset: "three_norm_u_v_os"    # 数据集名称
  batch_size: 64                  # 批次大小
  num_classes: 3                  # 分类数量
  image_size: 28                  # 输入图像尺寸

train:
  epochs: 200                     # 训练轮数
  learning_rate: 0.00005          # 学习率
  weight_decay: 0.0001            # 权重衰减
  device: "cuda:0"                # 训练设备
```

### htnet.yml (HTNet独有)
```yaml
model:
  name: "HTNet"
  # HTNet特有参数
  patch_size: 7                   # Patch大小
  dim: 256                        # 特征维度
  heads: 3                        # 注意力头数
  num_hierarchies: 3              # 层级数量
  block_repeats: [2, 2, 10]       # Transformer块重复次数
```

### long_short_fusenet.yml (LSAFN独有)
```yaml
model:
  name: "LongShortActionFuseNet"
  # LSAFN特有参数
  use_long_action: True           # 是否使用长时动作
  use_short_action: False         # 是否使用短时动作
  use_feature_concat: False       # 特征融合方式
  embed_dim: 128                  # 嵌入维度
  depth: 3                        # 网络深度
```

### mmnet.yml (MMNet独有)
```yaml
model:
  name: "MMNet"
  # MMNet没有额外的特有参数，使用默认配置
```

### vit_srmcl.yml (VITSRMCL独有)
```yaml
model:
  name: "VITSRMCL"
  # VITSRMCL特有参数
  patch_size: 4                   # Patch大小
  embed_dim: 256                  # 嵌入维度
  depth: 6                        # Transformer层数
  num_heads: 8                    # 注意力头数
  mlp_ratio: 4.0                  # MLP扩展比例
  mask_ratio: 0.0                 # 掩码比例
```

## 使用示例

### 示例1：修改全局学习率
影响所有模型：
```yaml
# 修改 src/config/common.yml
train:
  learning_rate: 0.0001    # 从0.00005改为0.0001
```

### 示例2：只为HTNet调整patch_size
只影响HTNet：
```yaml
# 修改 src/config/htnet.yml
model:
  patch_size: 14           # 只改变HTNet的patch大小
```

### 示例3：为所有模型增加训练轮数
```yaml
# 修改 src/config/common.yml
train:
  epochs: 300              # 所有模型都训练300轮
```

### 示例4：为特定模型单独设置epochs
```yaml
# 修改 src/config/vit_srmcl.yml
# 注意：模型配置中没有epochs字段，会自动使用common.yml中的值
# 如果需要不同，可以直接添加
train:
  epochs: 400              # VITSRMCL训练400轮，其他模型200轮
```

## 优势

1. **避免重复**：公共参数只需定义一次
2. **易于维护**：修改全局参数只需改common.yml
3. **清晰分离**：模型特有参数一目了然
4. **灵活配置**：可以为单个模型覆盖全局设置

## 添加新模型

1. 在 `src/model/` 创建模型文件（如 `my_model.py`）
2. 在 `src/config/` 创建配置文件（如 `my_model.yml`）
3. 配置文件中只需定义：
   ```yaml
   model:
     name: "MyModel"
     # 模型特有参数...
   ```
4. 公共参数自动从common.yml继承

## 注意事项

1. **model.name 必须定义**：每个模型配置文件必须包含模型名称
2. **参数覆盖**：模型配置中的参数会覆盖common.yml中的同名参数
3. **缺失参数**：如果模型配置中缺少某参数，会使用common.yml中的值
4. **注释说明**：建议在配置文件中添加注释，说明参数用途

## 配置加载流程

```python
# run_main.py 中的伪代码
def load_config(model_name):
    # 1. 加载公共配置
    common_config = load_yaml('src/config/common.yml')
    
    # 2. 加载模型配置
    model_config = load_yaml(f'src/config/{model_name}.yml')
    
    # 3. 合并配置（模型配置优先）
    final_config = deep_merge(common_config, model_config)
    
    return final_config
```
