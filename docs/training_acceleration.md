# LaneATT 训练加速方案

## 当前训练配置概览

| 配置项 | 当前值 | 位置 |
|--------|--------|------|
| batch_size | 8 | experiments/laneatt_r18_tusimple/config.yaml |
| epochs | 100 | experiments/laneatt_r18_tusimple/config.yaml |
| optimizer | Adam, lr=0.0003 | experiments/laneatt_r18_tusimple/config.yaml |
| lr_scheduler | CosineAnnealingLR, T_max=45400 | experiments/laneatt_r18_tusimple/config.yaml |
| num_workers | 8 | lib/runner.py:191 |
| pin_memory | 未设置（默认 False） | lib/runner.py:188-192 |
| persistent_workers | 未设置（默认 False） | lib/runner.py:188-192 |
| 混合精度 | 无 | lib/runner.py:48-59 |
| checkpoint 间隔 | 每 1 epoch | experiments/laneatt_r18_tusimple/config.yaml |
| validation 间隔 | 每 10 epoch | experiments/laneatt_r18_tusimple/config.yaml |
| train split | train+val | experiments/laneatt_r18_tusimple/config.yaml |
| root | datasets/ | experiments/laneatt_r18_tusimple/config.yaml |
| 精度 | FP32 | 全局 |

---

## 方法 1：DataLoader 参数优化

### 预计加速：10-15%

### 当前代码 (`lib/runner.py:186-193`)

```python
def get_train_dataloader(self):
    train_dataset = self.cfg.get_dataset('train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=self.cfg['batch_size'],
                                               shuffle=True,
                                               num_workers=8,
                                               worker_init_fn=self._worker_init_fn_)
    return train_loader
```

### 优化方案

**DataLoader 参数优化：**

```python
def get_train_dataloader(self):
    train_dataset = self.cfg.get_dataset('train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=self.cfg['batch_size'],
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True,
                                               persistent_workers=True,
                                               prefetch_factor=4,
                                               worker_init_fn=self._worker_init_fn_)
    return train_loader
```

**训练循环中配合 `non_blocking=True`（必须同步修改）：**

```python
images = images.to(self.device, non_blocking=True)
labels = labels.to(self.device, non_blocking=True)
```

> **重要**：`pin_memory=True` 必须配合 `.to(device, non_blocking=True)` 才能发挥真正效果。不加 `non_blocking=True`，`.to()` 仍然是同步传输，`pin_memory` 分配的页锁定内存几乎没有收益。

**同步修改 `get_test_dataloader` 和 `get_val_dataloader`：** 这两个方法（`runner.py:195-211`）也应添加 `pin_memory=True`、`persistent_workers=True`、`prefetch_factor=4` 参数，以加速 validation 和 test 阶段的数据加载。

四个参数的作用：

**`pin_memory=True`**：在 CPU 端分配页锁定（page-locked）内存。默认情况下，CPU tensor 存放在可分页内存中，传输到 GPU 时需要先拷贝到页锁定内存再通过 DMA 传输。开启 pin_memory 后跳过这一步，直接通过 DMA 传输，减少一次内存拷贝。

**`non_blocking=True`**：配合 `pin_memory` 使用。开启后 `.to(device)` 调用立即返回，CPU→GPU 传输在后台异步执行，可以与 GPU 计算重叠。不开启则 `.to()` 会阻塞等待传输完成，`pin_memory` 的加速效果大打折扣。

**`persistent_workers=True`**：默认情况下，每个 epoch 结束后 DataLoader 会销毁所有 worker 进程，下一个 epoch 重新创建。每次创建 worker 涉及进程 fork、Python 解释器初始化、数据集对象加载（包括 LaneATT 的标注数据和 imgaug 增强管线）。开启后 worker 进程在整个训练期间保持存活，省去反复启动的开销。对 100 个 epoch 的训练来说尤为明显。

**`prefetch_factor=4`**：每个 worker 预取的 batch 数。默认值为 2，增大到 4 意味着每个 worker 提前准备 4 个 batch 的数据。这样在 GPU 计算当前 batch 时，CPU 端有更多数据等待传输，减少 GPU 空等数据的时间。

### 对训练结果的影响

**零影响**。这四个参数仅改变数据传输和进程管理策略，不改变任何数据内容或计算逻辑。每一个 batch 送入模型的 tensor 完全相同。

### 注意事项

- `pin_memory=True` 会额外占用约 200-500 MB 主机内存（取决于 batch_size 和 prefetch 数量）
- `persistent_workers=True` 使 8 个 worker 进程常驻内存，需确保容器内存配额充足
- 如果容器内存紧张，可以将 `num_workers` 从 8 降到 4 并配合 `persistent_workers=True`，总体效果往往更好

---

## 方法 2：混合精度训练 (AMP)

### 预计加速：20-30%（同时减少约 40% 显存占用）

### 当前代码 (`lib/runner.py:44-62`)

```python
for epoch in trange(...):
    model.train()
    for i, (images, labels, _) in enumerate(pbar):
        images = images.to(self.device)
        labels = labels.to(self.device)
        outputs = model(images, **self.cfg.get_train_parameters())
        loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
```

### 优化方案

> **注意**：容器当前使用 PyTorch 2.4.0a0 (NVIDIA nv24.5)。`torch.cuda.amp.GradScaler()` 和 `torch.cuda.amp.autocast()` 在 PyTorch 2.4 中已标记为 deprecated，以下使用新 API。

```python
# 训练开始前初始化
scaler = torch.amp.GradScaler('cuda')

for epoch in trange(...):
    model.train()
    for i, (images, labels, _) in enumerate(pbar):
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # 前向传播在 autocast 上下文中执行
        with torch.amp.autocast('cuda'):
            outputs = model(images, **self.cfg.get_train_parameters())
            loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

        # 反向传播使用 GradScaler
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

### NMS CUDA 扩展的 FP16 兼容性风险

**这是 AMP 方案中最需要关注的问题。** `forward()` 的最后一步调用了 `self.nms()`，NMS 内部调用自定义 CUDA 扩展 `nms(proposals, scores, ...)`（`lib/nms/`）。在 `autocast()` 上下文中，`forward()` 的输出 `proposals` 可能是 FP16 tensor，而自定义 CUDA kernel 通常只为 FP32 编写。如果 NMS kernel 不做类型检查就直接按 FP32 解释 FP16 数据，**会产生错误结果而不是报错**。

虽然 NMS 在 `torch.no_grad()` 中执行（`laneatt.py:120`），但 `torch.no_grad()` 不会取消 `autocast` 的类型转换。

**安全方案 A**（推荐）：缩小 autocast 作用域，只包裹前向传播到 NMS 之前的部分，在 `laneatt.py` 的 `forward()` 中，进入 NMS 前显式转回 FP32：

```python
# laneatt.py forward() 中，调用 self.nms() 之前
reg_proposals = reg_proposals.float()  # 确保 NMS 接收 FP32
proposals_list = self.nms(reg_proposals, attention_matrix, nms_thres, nms_topk, conf_threshold)
```

**安全方案 B**：将 `autocast` 只包裹 `forward()`，`loss()` 在 autocast 外执行：

```python
with torch.amp.autocast('cuda'):
    outputs = model(images, **self.cfg.get_train_parameters())
# loss 在 autocast 外以 FP32 计算（更安全，但损失部分 AMP 加速）
loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)
```

**实施前必须验证**：

```python
# 验证 NMS 是否兼容 FP16 输入
with torch.amp.autocast('cuda'):
    outputs = model(dummy_input, conf_threshold=None, nms_thres=15., nms_topk=3000)
    for proposals, _, _, _ in outputs:
        print(f"proposals dtype: {proposals.dtype}")  # 如果是 float16，需要使用安全方案
```

### 原理

AMP（Automatic Mixed Precision）让 PyTorch 自动选择每个算子的计算精度：

- **FP16 执行的操作**：卷积（ResNet backbone）、矩阵乘法（attention bmm、cls_layer、reg_layer）、线性层。这些操作在 FP16 下速度约为 FP32 的 2 倍（利用 Tensor Core）。
- **保持 FP32 的操作**：Softmax、损失函数（FocalLoss、SmoothL1Loss）、BatchNorm 的统计量累积。这些操作对精度敏感，AMP 会自动保持 FP32。
- **GradScaler**：FP16 的动态范围较小（最小正数约 6e-8），梯度可能下溢为零。GradScaler 在反向传播前将 loss 放大，反向传播后将梯度缩小，防止梯度下溢。

### 对训练结果的影响

**几乎无影响**（F1 差异通常 < 0.1%）。原因：

1. LaneATT 的核心计算（卷积、线性层、attention）在 FP16 下有足够精度。ResNet18 的权重分布适合半精度表示。
2. 对精度最敏感的 FocalLoss（包含 log 和 pow 运算）在 AMP 下自动以 FP32 执行，不会因精度不足导致梯度异常。
3. SmoothL1Loss 的车道线回归目标是像素级偏移量（范围 0-640），FP16 能精确表示。
4. NMS 过程在 `torch.no_grad()` 上下文中执行（`laneatt.py:120`），不参与梯度计算，但**仍需确保输入为 FP32**（见上文兼容性风险）。

### 注意事项

- **Tensor Core 架构要求**：Tensor Core 从 **Volta 架构（V100）** 开始才有，不是 Pascal（GTX 1080）。Pascal GPU 上 AMP 仍可工作（FP16 计算 + FP32 累加），但无法利用 Tensor Core 的 2x 加速，实际加速约 10-15%。Turing（RTX 2080）及以上架构有 Tensor Core，可获得完整加速效果
- 极少数情况下 GradScaler 可能因 inf/nan 跳过 optimizer step，训练初期可能出现 1-2 次，属正常行为
- 显存减少约 40% 意味着可以进一步增大 batch_size（见方法 4）

---

## 方法 3：减少 checkpoint 和 validation 频率

### 预计加速：1-3%

### 当前配置

```yaml
model_checkpoint_interval: 1   # 每 epoch 存一次 checkpoint
val_every: 10                  # 每 10 epoch 做一次 validation
```

### 优化方案

```yaml
model_checkpoint_interval: 5   # 每 5 epoch 存一次 checkpoint
val_every: 10                  # 保持不变，已经是较低频率
```

### 逐项分析

**Checkpoint**：每次调用 `torch.save()` 保存完整的 model state_dict + optimizer state_dict + scheduler state_dict。对 ResNet18 模型，每个 checkpoint 约 50 MB。`torch.save()` 是同步 I/O 操作，在容器中写入磁盘时会阻塞训练进程。100 个 epoch 改为每 5 epoch 存一次，从 100 次减少到 20 次。但需注意，50 MB 在 SSD 上写入耗时 <100ms，100 epoch 省下的总时间不到 10 秒，主要收益来自减少磁盘占用。

**Validation**：每次 validation 需要：加载整个 val 数据集、模型切换到 eval 模式、前向推理所有 val 样本、计算 metrics、再切换回 train 模式。TuSimple val 集约 358 条数据，batch_size=8，需约 45 次前向推理，耗时较短。如果使用更大的数据集（如 CARLA），validation 的时间开销会更显著。

### 对训练结果的影响

**零影响**。checkpoint 和 validation 不参与训练过程，仅影响：

- 训练中断后的恢复粒度（最多丢失 4 个 epoch 的训练进度）
- 中间结果的可观测性（不能逐 epoch 追踪 val metrics 变化曲线）

### 建议

训练初期调试时保持高频率（interval=1, val_every=2），确认训练正常收敛后再切换为低频率进行完整训练。

---

## 方法 4：增大 batch_size

### 预计加速：30-50%（从 bs=8 增大到 bs=16 或 bs=32）

### 为什么能加速

GPU 的并行计算单元（CUDA Core / Tensor Core）在小 batch 时利用率低。以 ResNet18 为例：

| batch_size | GPU 利用率（估计） | 每秒处理帧数 | 每 epoch 用时（77k 样本） |
|------------|-------------------|-------------|------------------------|
| 8 | ~40% | ~300 | ~4.3 min |
| 16 | ~65% | ~480 | ~2.7 min |
| 32 | ~80% | ~580 | ~2.2 min |

增大 batch 不是线性加速，因为显存带宽和计算资源终会饱和。

### 配合 AMP 的显存计算

当前 batch_size=8 FP32 显存占用约 3-4 GB。开启 AMP 后：
- bs=8 AMP → 约 2-2.5 GB
- bs=16 AMP → 约 4-5 GB
- bs=32 AMP → 约 8-9 GB

需根据 GPU 显存容量选择。

### 对训练结果的影响

**需要同步调整超参数**，否则会影响收敛质量。

**学习率缩放**：经典的线性缩放法则（linear scaling rule）建议 batch_size 翻倍时 lr 也翻倍。当前 bs=8, lr=0.0003，则：
- bs=16 → lr=0.0006
- bs=32 → lr=0.0012

**scheduler T_max 调整**：CosineAnnealingLR 的 `T_max` 是总迭代次数。batch_size 翻倍后每 epoch 的迭代次数减半，`T_max` 也应减半：
- 当前：`T_max = 45400 = 100 epochs × 454 iterations/epoch`（train+val split 约 3632 样本，bs=8）
- bs=16：`T_max = 22700 = 100 × 227`

**大 batch 的泛化影响**：

- 研究表明大 batch SGD 训练倾向于收敛到 loss landscape 中较"尖锐"的极小值，泛化能力可能略差（Sharp Minima 问题）。但对于 Adam 优化器，这个效应没有 SGD 那么明显。
- LaneATT 使用 Adam 优化器，对 batch_size 变化的鲁棒性较好。
- 经验上 bs=16 通常与 bs=8 结果持平（F1 差异 < 0.3%），bs=32 可能出现 0.5-1% 的 F1 下降。
- 如果 bs=32 效果下降明显，可使用 warmup 策略：前 5 epoch lr 从 0 线性增长到目标值。

### 具体修改

`experiments/laneatt_r18_tusimple/config.yaml`:

```yaml
batch_size: 16
optimizer:
  name: Adam
  parameters:
    lr: 0.0006
lr_scheduler:
  name: CosineAnnealingLR
  parameters:
    T_max: 22700  # 原 T_max=45400 / 2（batch_size 翻倍，迭代次数减半）
```

**T_max 需根据实际数据集大小重新计算**：当前 T_max=45400 对应 train+val split 约 3632 样本（454 iterations × bs=8）。从 `cfgs/laneatt_tusimple_resnet18.yml` 的历史注释可以看到不同数据集组合下 iteration 数差异很大（454 ~ 10814）。T_max 的通用计算公式为：`T_max = epochs × ceil(样本总数 / batch_size)`。修改 batch_size 或切换数据集时，务必同步更新 T_max。

---

## 方法 5：减少数据增强重试次数

### 预计加速：10-20%（取决于数据集质量）

### 当前代码 (`lib/datasets/lane_dataset.py:307-321`)

```python
for i in range(30):
    img, line_strings = self.transform(image=img_org.copy(), line_strings=line_strings_org)
    line_strings.clip_out_of_image_()
    new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
    try:
        label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
        break
    except:
        if (i + 1) == 30:
            self.logger.critical('Transform annotation failed 30 times :(')
            exit()
```

### 失败原因分析

`transform_annotation()` 内部调用 `scipy.interpolate.InterpolatedUnivariateSpline` 对车道线做样条插值。失败场景：

1. 增强后车道线被裁剪到只剩 0-1 个点，无法拟合样条
2. 增强导致车道线点全部超出图像边界
3. 样条拟合数值不稳定（共线点、重复点）

每次失败后会重新随机增强（因为 imgaug 有随机性），直到增强结果不触发异常。

### 优化方案

```python
for i in range(5):  # 从 30 减少到 5
    img, line_strings = self.transform(image=img_org.copy(), line_strings=line_strings_org)
    line_strings.clip_out_of_image_()
    new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
    try:
        label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
        break
    except:
        if (i + 1) == 5:
            # 回退：使用不增强的原图
            try:
                new_anno = {'path': item['path'],
                            'lanes': self.linestrings_to_lanes(line_strings_org)}
                label = self.transform_annotation(new_anno,
                            img_wh=(self.img_w, self.img_h))['label']
                img = cv2.resize(img_org, (self.img_w, self.img_h))
            except:
                # 原始标注本身也可能拟合失败（如只有 1 个点），
                # 此时使用空标注避免程序崩溃
                self.logger.warning('Fallback annotation also failed for %s', item['path'])
                label = np.ones((self.dataset.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                                dtype=np.float32) * -1e5
                label[:, 0] = 1
                label[:, 1] = 0
                img = cv2.resize(img_org, (self.img_w, self.img_h))
```

> **注意**：回退路径必须有异常保护。原始标注也可能存在无法拟合样条的情况（如车道线只有 1 个点），不加 try-except 会导致程序崩溃。

### 对训练结果的影响

**轻微影响**（F1 差异 < 0.2%）。

- 减少重试次数意味着某些"难增强"的样本更容易回退到原图（无增强），等价于对这些样本降低了 aug_chance。
- 这些"难增强"的样本通常是车道线较短或接近图像边缘的 case，它们在增强后容易被裁剪消失，本身就不是典型的训练样本。
- 对整体数据分布的影响很小：假设 5% 的样本需要 >5 次重试，这些样本回退使用原图，相当于 aug_chance 从 1.0 降到 0.95。

### 进阶优化

在 `transform_annotations()` 调用**之前**预过滤原始标注（`lane_dataset.py` 的 `__init__` 中，第 56 行之前）：

```python
# lane_dataset.py 的 __init__ 中，self.transform_annotations() 调用之前
self.dataset.annotations = [
    anno for anno in self.dataset.annotations
    if len(anno['lanes']) > 0 and all(len(lane) >= 2 for lane in anno['lanes'])
]
self.transform_annotations()  # 已有代码
```

提前去除无法拟合样条的标注（点数 <2 的车道线），从根源减少重试次数。

> **注意**：过滤会改变 dataset 的长度（`__len__()` 返回值），如果 evaluation 代码依赖 annotation index 与原始数据集的一一对应关系（如 `draw_annotation(idx)` 通过 idx 查找原始图片路径），过滤后 index 会错位。建议仅在训练集上过滤，或者记录被过滤掉的样本数以便排查。

---

## 方法 6：torch.compile 模型编译

### 预计加速：10-20%（首 epoch 会变慢）

### 实施方法

```python
# runner.py train() 方法中，model.to(device) 之后
model = torch.compile(model, mode='reduce-overhead')
```

三种编译模式：
- `default`：平衡编译时间和运行时性能
- `reduce-overhead`：更激进优化，适合固定 shape 的输入
- `max-autotune`：最大性能，编译时间最长

### 原理

`torch.compile` 将 Python 级别的 PyTorch 操作编译为优化的 GPU 内核：
- 算子融合（Operator Fusion）：将多个小 kernel 合并为一个大 kernel，减少 GPU kernel launch 开销和显存读写
- 内存规划（Memory Planning）：优化中间 tensor 的内存分配和复用
- 自动调优（Auto-tuning）：对卷积等操作自动选择最优算法

### 对训练结果的影响

**理论上零影响**。torch.compile 保证数值等价性，编译后的计算图与原始 eager mode 产生相同的结果。

但在实践中有两点需要注意：
1. FP32 下的算子融合可能改变浮点运算顺序，导致微小的数值差异（最后几位有效位），但不影响收敛结果。
2. 如果涉及自定义 CUDA 扩展（本项目的 NMS），编译器可能无法优化甚至不兼容。

### 兼容性风险

**这是本方案中风险最高的方法**：

1. **NMS CUDA 扩展**：LaneATT 使用自定义的 `lib/nms/` CUDA C++ 扩展。`torch.compile` 可能无法追踪这个自定义算子，导致 graph break（回退到 eager mode 执行）。Graph break 不会导致错误，但会削弱编译优化效果。

2. **Python for 循环**：`cut_anchor_features()` 方法（`laneatt.py:236`）中有 `for batch_idx, img_features in enumerate(features)` 逐 batch 处理的 Python 循环，这也是一个 graph break 点，编译器无法将其向量化。

3. **动态 shape**：NMS 输出的 proposal 数量每个 batch 不同。`torch.compile` 对动态 shape 的支持在 PyTorch 2.1+ 有所改善，但仍可能触发频繁重编译，反而变慢。

4. **PyTorch 版本**：容器当前为 PyTorch 2.4.0a0 (NVIDIA nv24.5)，版本满足要求。

### 建议

先测试可行性：

```python
# 快速测试编译是否成功
model = torch.compile(model, mode='default')
dummy_input = torch.randn(8, 3, 360, 640).cuda()
output = model(dummy_input, conf_threshold=None, nms_thres=15., nms_topk=3000)
print("Compile OK")
```

如果报错或 graph break 过多，放弃此方法。

---

## 综合方案与实施顺序

### 推荐方案：方法 1 + 2 + 3 组合

| 步骤 | 方法 | 预计加速 | 对结果影响 | 风险 |
|------|------|---------|-----------|------|
| 1 | DataLoader 优化 | 10-15% | 零 | 无 |
| 2 | 混合精度 (AMP) | 20-30% | < 0.1% | 低（需验证 NMS FP16 兼容性） |
| 3 | 降低 checkpoint/val 频率 | 1-3% | 零 | 无 |
| **合计** | | **20-35%** | **< 0.1%** | **低** |

> **注意**：各方法针对不同瓶颈（数据加载、GPU 计算、I/O），加速不能简单相加。如果 GPU 计算已经是瓶颈，DataLoader 优化的实际收益会低于预估。

如果需要更大加速，在上述基础上叠加：

| 步骤 | 方法 | 额外加速 | 对结果影响 | 风险 |
|------|------|---------|-----------|------|
| 4 | batch_size 8→16 | 30-50% | 需调参 | 中 |
| 5 | 减少 augmentation 重试 | 10-20% | < 0.2% | 低 |
| 6 | torch.compile | 10-20% | 零(理论) | 高（NMS 扩展 + Python 循环导致多处 graph break） |

### 预期效果

以当前 TuSimple train+val split（约 3632 样本）、100 epoch 为例的粗略估计：

| 配置 | 每 epoch 耗时 | 100 epoch 总耗时 |
|------|-------------|-----------------|
| 当前（bs=8, FP32） | ~1 min | ~1.7 h |
| 方法 1+2+3（bs=8, AMP） | ~0.7 min | ~1.2 h |
| 方法 1+2+3+4（bs=16, AMP） | ~0.5 min | ~0.8 h |

如果使用更大数据集（如 CARLA 合并后约 10814 iterations/epoch，bs=8），耗时会成比例增加：

| 配置 | 每 epoch 耗时 | 100 epoch 总耗时 |
|------|-------------|-----------------|
| 当前（bs=8, FP32） | ~5 min | ~8.3 h |
| 方法 1+2+3（bs=8, AMP） | ~3.5 min | ~5.8 h |
| 方法 1+2+3+4（bs=16, AMP） | ~2 min | ~3.3 h |

（以上为估计值，实际取决于 GPU 型号和数据集大小）
