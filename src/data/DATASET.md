# 数据集加载器使用说明

## 📂 支持的目录结构

`MEDataset` 通过**目录结构自动检测**格式，不依赖数据集名称。

### 丰富格式（rich，当前主格式）

由 `1_extract_dataset.py` 生成，包含 RGB 三帧 + 光流：

```
dataset/
└── casme2_7c/              # 任意数据集名均可
    ├── rgb/                # RGB 三帧
    │   ├── sub01/
    │   │   ├── 0/          # 类别 (数字目录)
    │   │   │   ├── xxx_onset.jpg
    │   │   │   ├── xxx_apex.jpg
    │   │   │   └── xxx_offset.jpg
    │   │   └── 1/
    │   └── sub02/
    ├── flow/               # onset→apex Farneback 光流
    │   └── sub01/0/xxx_flow.npy
    └── flow_ao/            # apex→offset 光流 (可选)
        └── sub01/0/xxx_flow_ao.npy
```
> **检测规则**: 目录下存在 `rgb/` 或 `flow/` 子目录 → 丰富格式。

### 新格式（apex/sequence，过渡格式）

已不再由提取脚本生成，仅向后兼容：

```
dataset/
└── xxx/
    ├── apex/               # 或 sequence/
    │   ├── sub01/
    │   │   ├── 0/
    │   │   │   └── xxx.jpg
    │   │   └── 1/
    │   └── sub02/
```
> **检测规则**: 无 `rgb/`/`flow/`，但有 `apex/` 或 `sequence/` → 新格式。加载时用 `*.jpg` 匹配。

### 旧格式（传统 baseline 数据集）

无子目录层级，直接 `subject/class/*.png`：

```
dataset/
└── casme2/
    ├── sub01/
    │   ├── 0/              # 类别0
    │   │   └── xxx.png
    │   └── 1/
    └── sub02/
```
> **检测规则**: 无任何子目录 → 旧格式。加载时用 `*.png` 匹配。

---

## ⚙️ 配置方法

格式由代码**自动检测目录结构**决定，无需在配置中标明。只需指定数据集名和 `frame_type`。

### 1. 丰富格式 — 单帧 apex

```yaml
# config.yml
data:
  dataset: "casme2_7c"    # 任意名称，代码自动检测 rgb/ 子目录
  frame_type: "apex"      # 扫描 rgb/*_apex.jpg
```
**行为**: 检测到 `rgb/` 子目录 → 走丰富格式分支，`frame_type="apex"` 仅加载 apex 帧。

### 2. 丰富格式 — 光流

```yaml
data:
  dataset: "casme2_7c"
  frame_type: "flow"      # 扫描 flow/*.npy
```
**行为**: 加载 `.npy` 光流，归一化到 [-1, 1]，bilinear resize 到 `image_size`。

### 3. 丰富格式 — RGB 三帧

```yaml
data:
  dataset: "casme2_7c"
  frame_type: "rgb_triplet"  # 加载 onset+apex+offset，堆叠为 (9,H,W)
```
**行为**: 以 `rgb/*_apex.jpg` 为锚，推导对应 `_onset.jpg` / `_offset.jpg`。

### 4. 丰富格式 — RGB + 光流

```yaml
data:
  dataset: "casme2_7c"
  frame_type: "rgb_flow"     # (RGB_apex, flow) 元组
```
**行为**: 从 `rgb/` 加载 apex 帧，从 `flow/` 加载对应光流，返回 `(rgb, flow)` 元组。

### 5. 丰富格式 — RGB + 双向光流 (Phase-Aware)

```yaml
data:
  dataset: "casme2_7c"
  frame_type: "rgb_dual_flow"  # (RGB_apex, flow_oa, flow_ao) 三元组
```
**行为**: 从 `rgb/` 加载 apex 帧，从 `flow/` 加载 onset→apex 光流，从 `flow_ao/` 加载 apex→offset 光流，返回 `(rgb, flow_oa, flow_ao)` 三元组。
> **适用模型**: BPTNet (Phase-Aware Dual-Flow 架构)  
> **前置条件**: 数据集需包含 `flow_ao/` 子目录（由 `1_extract_dataset.py` 生成，当前已存在）

### 6. 旧格式 / 新格式

```yaml
data:
  dataset: "casme2"       # 无 rgb/flow/apex/sequence 子目录
  frame_type: "apex"      # 此值仅影响缓存键，扫描路径为 dataset/casme2/
```
**行为**: 旧格式直接用 `*.png` 扫描；新格式 (`apex/`/`sequence/`) 用 `*.jpg` 扫描。

---

## 🚀 使用示例

### 示例1：训练HTNet（单帧输入）

```yaml
# htnet_casme2_7c.yml
data:
  dataset: "casme2_7c"
  frame_type: "apex"       # HTNet只需要apex帧
  batch_size: 64
  image_size: 28

model:
  name: "HTNet"
  num_classes: 7
  image_size: 28
```

```bash
python run_main.py --config htnet_casme2_7c
```

---

### 示例2：训练LSAFN（需要双帧差值）

```yaml
# lsafn_casme2_7c.yml
data:
  dataset: "casme2_7c"
  frame_type: "rgb_triplet"   # LSAFN从三帧中提取 onset 和 apex 计算差值
  batch_size: 32
  image_size: 224

model:
  name: "LongShortActionFuseNet"
  num_classes: 7
  image_size: 224
```

**注意**：LSAFN 的 `_parse_input` 方法从 `(9,H,W)` 三帧堆叠中提取 onset 和 apex（前 6 个通道）。

---

### 示例3：训练AlexNet/VGG16（迁移学习）

```yaml
# alexnet_casme2_7c.yml
data:
  dataset: "casme2_7c"
  frame_type: "apex"
  batch_size: 32
  image_size: 224          # AlexNet需要224x224

model:
  name: "AlexNet"
  pretrained: True
  image_size: 224
```

---

## 🔧 核心实现

### 1. 结构驱动的格式检测

```python
# MEDataset.__init__ (dataset.py L47-56)
main_path = self.data_dir / self.dataset_name
self.is_rich = (main_path / "rgb").exists() or (main_path / "flow").exists()
self.is_new  = (main_path / "apex").exists() or (main_path / "sequence").exists()
# 否则为旧格式
```

**不存在** `NEW_DATASETS` 常量或基于名称的白名单。任何目录名只要结构匹配即可。

### 2. 调度逻辑

```python
# _scan() (dataset.py L99-109)
if self.is_rich:
    return self._scan_rich(main_path)    # rgb/ 或 flow/ 驱动
elif self.is_new:
    scan_path = main_path / self.frame_type
    return self._scan_flat(scan_path, "*.jpg")
else:
    return self._scan_flat(main_path, "*.png")
```

### 3. 丰富格式 frame_type 分支

```python
# _scan_rich() (dataset.py L111-135)
if ft == "flow":
    → _scan_flat(main_path / "flow", "*.npy")
elif ft in ("rgb_triplet", "rgb_flow", "rgb_dual_flow"):
    → _scan_rgb_anchor(rgb_dir, main_path, ft)   # 以 apex 帧为锚
else:  # apex 等
    → _scan_flat(main_path / "rgb", "*_apex.jpg")
```

### 4. `_build_class_mapping` — 类名→数字

```python
# dataset.py L137-164
# 扫描所有 subject/class 目录名
# 全部可解析为 int → 直接用 int(name)
# 包含字符串 → 按字母序分配 0-based ID
```
当前数据集类目录均为数字（0~N），故直接返回 `int(name)`。

### 5. 缓存键

```python
# _build_cache_key() (dataset.py L89-95)
cache_key = f"{data_dir}|{dataset_name}|{frame_type}[|rich|new]"
```
不同数据目录/数据集名/frame_type/格式 完全隔离缓存。

---

## 📊 兼容性测试

### 测试旧格式
```python
# 扫描 dataset/casme2/sub*/class/*.png
config = {'data': {'dataset': 'casme2', 'frame_type': 'apex'}}
loader = create_dataloaders(config)
```

### 测试丰富格式 — apex
```python
# 扫描 dataset/casme2_7c/rgb/*_apex.jpg
config = {'data': {'dataset': 'casme2_7c', 'frame_type': 'apex'}}
loader = create_dataloaders(config)
```

### 测试丰富格式 — rgb_triplet
```python
# 扫描 dataset/casme2_7c/rgb/*_apex.jpg 并推导 onset/offset
config = {'data': {'dataset': 'casme2_7c', 'frame_type': 'rgb_triplet'}}
loader = create_dataloaders(config)
```

### 测试丰富格式 — flow
```python
# 扫描 dataset/casme2_7c/flow/*.npy
config = {'data': {'dataset': 'casme2_7c', 'frame_type': 'flow'}}
loader = create_dataloaders(config)
```

### 测试丰富格式 — rgb_dual_flow
```python
# 扫描 dataset/casme2_7c/rgb/*_apex.jpg + flow/*.npy + flow_ao/*.npy
config = {'data': {'dataset': 'casme2_7c', 'frame_type': 'rgb_dual_flow'}}
loader = create_dataloaders(config)
```

---

## ⚠️ 注意事项

### 1. 模型与 frame_type 匹配

| 模型 | 推荐 frame_type | 原因 |
|------|---------------|------|
| BPTNet | apex / flow / rgb_flow / **rgb_dual_flow** ★ | Phase-Aware Dual-Flow 架构，支持全部5种输入 |
| HTNet | apex / flow / rgb_flow | 单帧/光流/融合分类，通道拼接模式 |
| VITSRMCL | apex | MAE 重构单帧 |
| AlexNet/VGG16/GoogLeNet | apex | 标准图像分类 |
| LSAFN | rgb_triplet | 需要 onset-apex 差值（自行从三帧提取） |
| MMNet | rgb_triplet | 需要 onset-apex 差值 |

> **BPTNet 论文实验建议**:  
> 阶段1（消融）: `apex` → `flow` → `rgb_flow` → `rgb_dual_flow`，逐步验证各组件贡献  
> 阶段2（对照）: HTNet 跑 `apex` + `rgb_flow`，与 BPTNet 同配置对比  
> 阶段3（SOTA）: 其他模型各跑一种最优 frame_type（如上表）

### 2. 图片格式由目录结构决定

- **丰富格式** (`rgb/` 子目录): `.jpg` (RGB) + `.npy` (光流)
- **新格式** (`apex/`/`sequence/`): `.jpg`
- **旧格式** (无子目录): `.png`

> **rgb_dual_flow 特殊要求**: 除了 `rgb/` 和 `flow/`，还需要 `flow_ao/` 子目录（apex→offset 光流）。`1_extract_dataset.py` 已生成，无需额外操作。

### 3. 类别合并机制

通过 `DATASET_MERGE_CONFIG`（`dataset.py` L34），可将原始细粒度类别映射为粗粒度类别，**无需复制数据**——合并数据集与原始数据集共享同一份文件，仅在加载时做 label 重映射。

#### 合并依据

| 类别数 | 协议 | 合并逻辑 |
|:---:|------|------|
| **3** | MEGC2019 | Positive(Happiness), Negative(其余全部), Surprise |
| **4** | 原作者 note.txt | Positive(Happiness), Negative(Disgust+Sadness+Fear), Surprise, Others(Repression+Tense 等) |
| **5** | 论文主流 (15/17篇) | Happiness, Surprise, Disgust, Repression, Others(Sadness+Fear) |

#### 原始标签 → 合并映射

**CASME II (casme2_7c)** — 原始: 0=repression, 1=sadness, 2=others, 3=disgust, 4=surprise, 5=happiness, 6=fear

| 目标数据集 | 类别数 | 类别名 | label_map |
|---|:---:|---|---|
| casme2_7c | 7 | repression, sadness, others, disgust, surprise, happiness, fear | 原始，无映射 |
| casme2_5c | 5 | happiness, surprise, disgust, repression, others | {0→3, 1→4, 2→4, 3→2, 4→1, 5→0, 6→4} |
| casme2_4c | 4 | negative, others, positive, surprise | {0→1, 1→0, 2→1, 3→0, 4→3, 5→2, 6→0} |
| casme2_3c | 3 | negative, positive, surprise | {0→0, 1→0, 2→0, 3→0, 4→2, 5→1, 6→0} |

**CASME (casme_8c)** — 原始: 0=tense, 1=disgust, 2=repression, 3=surprise, 4=happiness, 5=sadness, 6=fear, 7=contempt

| 目标数据集 | 类别数 | 类别名 | label_map |
|---|:---:|---|---|
| casme_8c | 8 | 原始8类 | 无映射 |
| casme_4c | 4 | negative, others, positive, surprise | {0→1, 1→0, 2→1, 3→3, 4→2, 5→0, 6→0, 7→0} |
| casme_3c | 3 | negative, positive, surprise | {0→0, 1→0, 2→0, 3→2, 4→1, 5→0, 6→0, 7→0} |

**CAS(ME)^2 (casme_sq_8c)** — 原始: 0=happiness, 1=disgust, 2=anger, 3=surprise, 4=fear, 5=sadness, 6=helpless, 7=pain

| 目标数据集 | 类别数 | 类别名 | label_map |
|---|:---:|---|---|
| casme_sq_8c | 8 | 原始8类 | 无映射 |
| casme_sq_4c | 4 | negative, others, positive, surprise | {0→2, 1→0, 2→0, 3→3, 4→0, 5→0, 6→1, 7→1} |
| casme_sq_3c | 3 | negative, positive, surprise | {0→1, 1→0, 2→0, 3→2, 4→0, 5→0, 6→0, 7→0} |

#### 实验优先级：应该跑哪些类别数？

| 数据集 | 必跑 | 建议跑 | 可跳过 | 理由 |
|---|---|---|---|---|
| casme2 | **5c, 3c** | **7c, 4c** | — | 5c=论文主流对标标准(15/17篇)，3c=MEGC2019跨库评测协议，7c展示全类能力，4c原作者推荐且比5类更均衡 |
| casme | **3c** | **4c** | 8c | 8c样本仅186且不均衡比66:1，8类几乎不可能收敛；4c/3c可对齐casme2做联合对比 |
| casme_sq | **3c** | 4c | 8c | 仅53个样本，8类每类不足10个，4类勉强可行，3类是唯一合理选择 |

> **核心原则**: 3类是跨数据集/跨论文的通用协议，5类是 CASME II 专属主流标准，4类是原作者推荐且比5类更均衡。不要自行发明合并规则——FACS 体系下的约定是领域共识。

#### 使用方法

```yaml
# 只需改 dataset 名称，数据目录和文件无需任何修改
data:
  dataset: "casme2_5c"    # 自动从 casme2_7c 读取数据，label 按 5 类映射
  frame_type: "apex"

model:
  num_classes: 5           # 必须与合并后类别数匹配
```

#### 实现细节

```python
# dataset.py MEDataset.__init__
self._merge_cfg = DATASET_MERGE_CONFIG.get(self.dataset_name)
self._source_name = self._merge_cfg['source'] if self._merge_cfg else self.dataset_name
self._label_map = self._merge_cfg['label_map'] if self._merge_cfg else None

# 扫描使用源数据集目录
main_path = self.data_dir / self._source_name   # e.g. casme2_5c → 扫描 casme2_7c/

# subject 过滤时应用 label_map
label = self._label_map[s["label"]] if self._label_map else s["label"]

# __getitem__ 使用已映射的 self.labels[idx]
label = self.labels[idx]   # 不是 sample["label"]
```

- **缓存**: `_build_cache_key` 使用 `_source_name`，合并数据集与原始数据集**共享扫描缓存**
- **`get_all_subjects`**: 同样解析源数据集目录
- **`num_classes`**: 由 `run_main.py` 从数据集名自动推导（`casme2_5c` → `5`）

### 4. 类别标签范围 (原始数据集)

| 数据集          | 类别数 | 标签范围 | 备注 |
|--------------|:---:|:------|------|
| casme2 (旧)   | 3   | 0-2 | legacy baseline |
| casme2_7c    | 7   | 0-6 | 见数据准备说明 §4.1 |
| casme_8c     | 8   | 0-7 | 见数据准备说明 §4.2 |
| casme_sq_8c  | 8   | 0-7 | 微表情过滤后实际 8 类 |
| casme_sq_10c | 10  | 0-9 | 放开宏表情过滤后（暂无数据） |

**重要**：确保 `model.num_classes` 与合并后的数据集匹配！详见上方「类别合并机制」。

### 5. LOSO评估

```yaml
# LOSO模式下，frame_type同样生效
data:
  dataset: "casme2_7c"
  split_mode: "loso"
  test_subject: "sub01"
  frame_type: "apex"  # 仍然需要指定
```

---

## 🐛 常见问题

### Q1: 找不到数据目录
```
FileNotFoundError: 数据集必须是 rich 格式 (包含 rgb/ 或 flow/ 子目录): <workspace>/dataset/casme2_7c
```

**解决**：
1. 共享数据集位于工作区根目录 `mer/dataset/`（多个模型共用，已相对 `mer/` 解析，见 `common.yml` 的 `data.data_dir` 与 `config.py` 的 `workspace_root`）。确认 `mer/dataset/casme2_7c/{rgb,flow,flow_ao}` 存在
2. 数据由仓库外的预处理脚本一次性提取成 rich 格式；输出目录应指向 `<data-root>/`
3. 检查 `data.frame_type` 是否有效（`apex` / `flow` / `rgb_triplet` / `rgb_flow` / `rgb_dual_flow`）
4. 确认 `data.dataset` 名称拼写正确
4. 丰富格式需存在 `rgb/` 或 `flow/` 子目录

### Q2: 加载了0个样本
```
加载 train 集: 0 个样本, 受试者数: 0
```

**解决**：
1. 检查目录结构是否符合预期
2. 确认图片格式正确（.jpg vs .png）
3. 查看日志中的扫描路径是否正确

### Q3: 类别数量不匹配
```
RuntimeError: size mismatch, m1: [64 x 4096], m2: [1000 x 7]
```

**解决**：
修改配置文件中的 `model.num_classes`：
```yaml
model:
  num_classes: 7  # 必须与数据集一致
```

---

## 📝 总结

✅ **结构驱动检测**：自动根据 `rgb/`/`flow/`/`apex/`/`sequence/` 子目录存在性判断格式  
✅ **向后兼容**：旧格式（`.png`）无需修改配置  
✅ **灵活 frame_type**：`apex` / `flow` / `rgb_triplet` / `rgb_flow` / `rgb_dual_flow` 按需切换  
✅ **类别合并**：`DATASET_MERGE_CONFIG` 支持 3/4/5 类映射，不改数据只改配置  
✅ **缓存隔离**：不同格式/数据集/frame_type 独立缓存，合并数据集共享源数据集缓存

现在你只需修改配置文件，数据集加载器自动适配！🎉
