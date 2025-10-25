---
layout: post
title: "Faster R-CNN：结构总览与 MMDetection 源码路径（含张量形状速查）"
date: 2025-10-26
tags: [Faster R-CNN, MMDetection, 目标检测, RPN, RoIAlign]
mathjax: true
---

> **关键词**：Faster R-CNN，RPN，RoIAlign，BBoxHead，MMDetection 源码路径  
> **摘要**：一图梳理 Faster R-CNN 的数据流，给出张量形状速查表与 MMDetection 源码定位。并用一个常见输入（800×1333，Pad→800×1344）给出可落地的数字例子，最后附调试打印与常见坑排查清单。
<!--more-->

## 目录
* 目录
{:toc}

## 1. 结构总览（数据流一图速览）

```
Image → Backbone(ResNet) → Neck(FPN: P2…P6)
                        → RPN (cls: 前景/背景, reg: Δ)
                      proposals(Top-K) ──→ RoIAlign(选层/对齐采样 7×7)
                                           → BBox Head (分类 cls + 回归 reg)
                                           → Δ 解码 → per-class NMS → det_bboxes, det_labels
```

- RPN：各金字塔层每个网格点，对 A 个锚框做二分类（前景/背景）与 4 维回归（Δ）。
- RoIAlign：把 proposals 映射回 FPN 相应层，裁成 7×7 特征块。
- BBox Head：输出 `cls_score`（含背景类）与 `bbox_pred`（类相关/类无关）。

> *可放图位*：  
> `![结构总览]({{ '/assets/img/2025-10-26/fasterrcnn-arch.png' | relative_url }})`

---

## 2. Backbone

### 2.1 ResNet50（Stem → Stages）
- **输入**：RGB 图像（如 224×224）
- **Stem**：7×7 Conv (stride=2, out=64) → BN → ReLU → 3×3 MaxPool (s=2)
- **Stages**：4 个 stage（Bottleneck 堆叠），常见层数 (3, 4, 6, 3)
- **Bottleneck**：`1×1 降维 → 3×3 提特征 → 1×1 升维`，残差连接相加后 ReLU。

**BN 公式（训练期）**：
$$
\mu=\frac{1}{m}\sum_{i=1}^m x_i,\quad
\sigma^2=\frac{1}{m}\sum_{i=1}^m (x_i-\mu)^2,\\
\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad
y=\gamma\hat x+\beta
$$

---

## 3. Neck（FPN）

### 3.1 输入/输出
- 以 ResNet50 为例，输入通道约 `[256, 512, 1024, 2048]`（对应 `C2..C5`）。
- 输出将通道统一到 `out_channels=256`，得到 `P2..P5`；若 `num_outs=5` 再由 `P5` 下采样出 `P6`。

### 3.2 侧向 & 自顶向下融合（代码片段）
```python
# 侧向 1x1 conv
laterals = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]

# top-down 融合
for i in range(len(laterals) - 1, 0, -1):
    size = laterals[i - 1].shape[2:]
    laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=size, **self.upsample_cfg)
```

- 输出形状示例（B=1）：`P2..P6 = [1, 256, 128,128] … [1,256, 8,8]`（以 1024×1024 为例）。

> *可放图位*：`![FPN 融合示意]({{ '/assets/img/2025-10-26/fpn.png' | relative_url }})`

---

## 4. RPN

### 4.1 前向（单层）
```python
def forward_single(self, x):
    x = self.rpn_conv(x); x = F.relu(x)
    rpn_cls_score = self.rpn_cls(x)  # (N, A, H, W)  —— use_sigmoid=True → A；Softmax → 2A
    rpn_bbox_pred = self.rpn_reg(x)  # (N, 4A, H, W)
    return rpn_cls_score, rpn_bbox_pred
```

- **通道说明**：Sigmoid 情况下分类通道为 `A`；Softmax 为 `2A`。回归恒为 `4A`。  
- **形状例**（A=3）：`cls=(1,3,128,128)` 到 `cls=(1,3,8,8)`；`reg=(1,12,128,128)` 到 `reg=(1,12,8,8)`。

### 4.2 训练（loss）
1. 生成锚框 `grid_priors(featmap_sizes)` → 过滤图内锚 `inside_flags` → `assigner` → `sampler`  
2. **unmap 回填**：把只在有效锚上计算的 targets/weights 回到“全部锚”长度  
3. **损失**：
   - 分类（Sigmoid）  
     $$\ell_{\text{cls}} = -\frac{1}{\text{avg\_factor}}\sum_i \big[y_i\log p_i+(1-y_i)\log (1-p_i)\big]$$
   - 回归（Smooth L1, \(\beta=1/9\)）  
     $$\ell_{\beta}(x)=\begin{cases}\frac{0.5x^2}{\beta}&|x|<\beta\\ |x|-0.5\beta&\text{otherwise}\end{cases}$$

### 4.3 推理
- `permute+reshape` → Sigmoid 得分 → `nms_pre` 预筛 → `delta` 解码 → 过滤小框 → NMS → `max_per_img` 截断。

> *可放图位*：`![RPN 流程]({{ '/assets/img/2025-10-26/rpn-flow.png' | relative_url }})`

---

## 5. RoI / RCNN Head

### 5.1 输入
- FPN 特征 `x=(P2..P6)`  
- RPN proposals（`InstanceData`）：`bboxes(n_i,4)`（xyxy）+ 可选 `scores`  
- 数据集元信息：`img_shape、ori_shape、scale_factor、pad_shape...`

### 5.2 选层与对齐（SingleRoIExtractor）
```python
# 选层：小框→低层（细），大框→高层（粗）
target_lvl = clip(floor(log2(sqrt(w*h)/finest_scale)), 0, L-1)   # 常用 finest_scale=56

# RoIAlign 得到 (N, C, 7, 7)
roi_feats = self.roi_layers[i](feats[i], rois_)
```

### 5.3 BBoxHead
- Flatten → 共享 FC（如 2×1024）→  
  - `cls_score`: `(N, K+1)`（Softmax 多类+背景）  
  - `bbox_pred`: `(N, 4)`（类无关）/ `(N, 4K)`（类相关）

**Δ 编码：**
$$
t_x=\frac{x-x_a}{w_a},\quad
t_y=\frac{y-y_a}{h_a},\quad
t_w=\log\frac{w}{w_a},\quad
t_h=\log\frac{h}{h_a}
$$

### 5.4 测试后处理
- `multiclass_nms`：把 `(N,4K)` reshape 为 `(N,K,4)`，丢背景列，分类别做（Soft-）NMS，输出：  
  `det_bboxes:(M,4+1)`、`det_labels:(M,)`。

> *可放图位*：`![RoIAlign 示意]({{ '/assets/img/2025-10-26/roi-align.png' | relative_url }})`

---

## 6. 具体数字例子（800×1333，Pad→800×1344）

> 短边缩放到 800，并 **Pad 到 32 的倍数**。例：`H=800, W=1333 → PadW=1344`。

| 层 | stride | 空间尺寸 (Hi×Wi) | 每层锚框数 `A*Hi*Wi`（A=3） |
|---|---:|---:|---:|
| P2 | 4  | 200×336 | 201,600 |
| P3 | 8  | 100×168 | 50,400 |
| P4 | 16 | 50×84   | 12,600 |
| P5 | 32 | 25×42   | 3,150 |
| P6 | 64 | 13×21   | 819 |
| **合计** |  |  | **268,569 anchors** |

说明：P6 常由 P5 以 `k=3,s=2,p=1` 再下采样得到（25×42 → 13×21）。RPN 会按 `nms_pre` 先做每层预筛，再 NMS，最终拼成 proposals（如 `max_per_img≈1000`）。

---

## 7. 在 MMDetection 的源码位置（3.x）

- **Detector（两阶段框架）**  
  `mmdet/models/detectors/two_stage.py`；`mmdet/models/detectors/faster_rcnn.py`
- **RPN**  
  Head：`mmdet/models/rpn_heads/rpn_head.py`；  
  Anchor：`mmdet/models/task_modules/prior_generators/anchor_generator.py`；  
  编解码：`mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py`
- **RoI / BBox 分支**  
  RoIHead：`mmdet/models/roi_heads/standard_roi_head.py`；  
  RoIAlign：`mmdet/models/roi_layers/roi_align.py`；  
  提取器：`mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py`；  
  BBoxHead：`mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py`
- **后处理**  
  NMS：`mmdet/structures/bbox/ops.py`（`multiclass_nms`）
- **训练样本分配/采样**  
  分配器：`mmdet/models/task_modules/assigners/max_iou_assigner.py`；  
  采样器：`mmdet/models/task_modules/samplers/random_sampler.py`

---

## 8. “配置 → 形状”的快速映射

```python
# 典型 MMDetection 配置片段（Python 字典风格）
model = dict(
    type='FasterRCNN',
    neck=dict(type='FPN', num_outs=5),  # → P2..P6, strides=[4, 8, 16, 32, 64]
    rpn_head=dict(
        type='RPNHead',
        anchor_generator=dict(
            scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
        ),
        # → 每层 A = len(scales) * len(ratios) = 3
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder', target_stds=[0.1, 0.1, 0.2, 0.2]
        ),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=NUM,
            reg_class_agnostic=False  # False→bbox_pred.shape[1] = 4*NUM；True→4
        ),
    ),
)
```

- `cls_score.shape = [R, NUM + 1]`（softmax 多分类 + 背景）
- `bbox_pred.shape = [R, 4 * NUM]`（`reg_class_agnostic=False`，类相关回归），或 `[R, 4]`（`True`，类无关回归）
- RPN 输出通道：`rpn_cls: A`，`rpn_reg: 4A`；`A = len(scales) * len(ratios)`

**Δ 编码：**
$$
t_x=\frac{x-x_a}{w_a},\quad
t_y=\frac{y-y_a}{h_a},\quad
t_w=\log\frac{w}{w_a},\quad
t_h=\log\frac{h}{h_a}
$$

---

## 9. 常见坑与快速排查

1. 背景通道：`cls_score=[R, NUM+1]`；确保标签与背景列对齐。  
2. 尺寸以 Pad 后为准：计算 `Hi, Wi` 用 **Pad 后** 尺寸，否则层与层会对不上。  
3. Anchor 数巨量：理论 ~26.86 万（此示例），实际会被 `nms_pre` 先筛、再 NMS。  
4. `reg_class_agnostic`：决定 `bbox_pred` 维度是 `4` 还是 `4*NUM`。  
5. 类别 id 与数据集：COCO 类别从 1 起，背景 0；自定义数据注意与 `num_classes` 一致。  
6. 阈值影响：`score_thr`、`nms.iou_threshold`、`max_per_img` 对误检/漏检影响显著。  
7. 训练/测试增强一致：`Resize/Normalize` 等配置保持一致，避免评测偏差。

---

## 10. 最少量调试打印（直接插到本地源码）

**在 `RPNHead.forward()` 中**
```python
print("[RPN] feat shapes:", [f.shape for f in x])
print("[RPN] cls per level:", [s.shape for s in cls_scores])
print("[RPN] reg per level:", [b.shape for b in bbox_preds])
```

**在 `StandardRoIHead._bbox_forward()` / `predict()` 中**
```python
print("[RoI] rois:", rois.shape)               # (N,5) = [batch,x1,y1,x2,y2]
print("[RoI] bbox_feats:", bbox_feats.shape)   # (N,256,7,7)
print("[RoI] cls_score:", cls_score.shape, "bbox_pred:", bbox_pred.shape)
```

**NMS 前后对比（单图）**
```python
print("[Post] before nms:", bboxes.shape, scores.shape)
print("[Post] after nms:", results.bboxes.shape, results.scores.shape)
```

---

*完*
