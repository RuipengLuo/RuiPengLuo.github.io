---
layout: post
title: "Faster R-CNN：结构总览与 MMDetection 源码路径（含张量形状速查）"
date: 2025-10-25
tags: [Faster R-CNN, MMDetection, 目标检测, RPN, RoIAlign]
mathjax: true
---

> **关键词**：Faster R-CNN，RPN，RoIAlign，BBoxHead，MMDetection 源码路径  
> **摘要**：一图梳理 Faster R-CNN 的数据流，给出张量形状速查表与 MMDetection 源码定位。并用一个常见输入（800×1333，Pad→800×1344）给出可落地的数字例子，最后附调试打印与常见坑排查清单。
<!--more-->

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

---

## 2. 张量形状速查（通用形式）

- FPN（单图 N=1）：`P2…P6` 的空间尺寸约为 `H/4, H/8, H/16, H/32, H/64`（以 **Pad 后** 尺寸为准）。
- RPN 逐层输出：
  - `rpn_cls: [N, A, Hi, Wi]`（sigmoid 二分类）
  - `rpn_reg: [N, 4A, Hi, Wi]`
- RoIAlign：`roi_feats: [R, C', 7, 7]`
- BBox Head：
  - `cls_score: [R, num_classes + 1]`（含背景）
  - `bbox_pred: [R, 4]`（`reg_class_agnostic=True`），或 `[R, 4 * num_classes]`（`False`）

---

## 3. 具体数字例子（800×1333，Pad→800×1344）

> 默认短边缩放到 800，并 **Pad 到 32 的倍数**。例：`H=800, W=1333 → PadW=1344`。

| 层 | stride | 空间尺寸 (Hi×Wi) | 每层锚框数 `A*Hi*Wi`（A=3） |
|---|---:|---:|---:|
| P2 | 4  | 200×336 | 201,600 |
| P3 | 8  | 100×168 | 50,400 |
| P4 | 16 | 50×84   | 12,600 |
| P5 | 32 | 25×42   | 3,150 |
| P6 | 64 | 13×21   | 819 |
| **合计** |  |  | **268,569 anchors** |

说明：
- P6 常由 P5 以 `k=3,s=2,p=1` 再下采样得到（25×42 → 13×21）。
- RPN 会按 `nms_pre` 先做每层预筛，再 NMS，最终拼成 proposals（如 `max_per_img≈1000`）。

---

## 4. 在 MMDetection 的源码位置（3.x）

- **Detector（两阶段框架）**
  - `mmdet/models/detectors/two_stage.py`
  - `mmdet/models/detectors/faster_rcnn.py`
- **RPN**
  - Head：`mmdet/models/rpn_heads/rpn_head.py`
  - Anchor 生成：`mmdet/models/task_modules/prior_generators/anchor_generator.py`
  - Δ 编/解码：`mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py`
- **RoI / BBox 分支**
  - RoIHead：`mmdet/models/roi_heads/standard_roi_head.py`
  - RoIAlign：`mmdet/models/roi_layers/roi_align.py`
  - RoI 特征提取器：`mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py`
  - BBoxHead（Shared2FC/ConvFC）：`mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py`
- **后处理**
  - NMS：`mmdet/structures/bbox/ops.py`（`multiclass_nms`）
- **训练样本分配/采样**
  - 分配器：`mmdet/models/task_modules/assigners/max_iou_assigner.py`
  - 采样器：`mmdet/models/task_modules/samplers/random_sampler.py`

---

## 5. 训练与推理的调用路径（极简伪代码）

**训练（`TwoStageDetector.loss`）**
```python
feats = extract_feat(img)  # backbone -> FPN
# RPN
rpn_losses, rpn_results_list = rpn_head.loss_and_predict(
    feats, batch_data, rpn_train_cfg
)
# RoI/BBox
roi_losses = roi_head.loss(
    feats, rpn_results_list, batch_data, rcnn_train_cfg
)
loss = {**rpn_losses, **roi_losses}
```

**推理（`TwoStageDetector.predict`）**
```python
feats = extract_feat(img)
proposals = rpn_head.predict(feats, rpn_test_cfg)
det_bboxes, det_labels = roi_head.predict(
    feats, proposals, rcnn_test_cfg, rescale=True
)
```

---

## 6. “配置 → 形状”的快速映射

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
t_x=\\frac{x-x_a}{w_a},\\quad
t_y=\\frac{y-y_a}{h_a},\\quad
t_w=\\log\\frac{w}{w_a},\\quad
t_h=\\log\\frac{h}{h_a}
$$

---

## 7. 常见坑与快速排查

1. 背景通道：`cls_score=[R, NUM+1]`；确保标签与背景列对齐。  
2. 尺寸以 Pad 后为准：计算 `Hi, Wi` 用 **Pad 后** 尺寸，否则层与层会对不上。  
3. Anchor 数巨量：理论 ~26.86 万（此示例），实际会被 `nms_pre` 先筛、再 NMS。  
4. `reg_class_agnostic`：决定 `bbox_pred` 维度是 `4` 还是 `4*NUM`。  
5. 类别 id 与数据集：COCO 类别从 1 起，背景 0；自定义数据注意与 `num_classes` 一致。  
6. 阈值影响：`score_thr`、`nms.iou_threshold`、`max_per_img` 对误检/漏检影响显著。  
7. 训练/测试增强一致：`Resize/Normalize` 等配置保持一致，避免评测偏差。

---

## 8. 最少量调试打印（直接插到本地源码）

**在 `RPNHead.forward()` 中**
```python
print("[RPN] feat shapes:", [f.shape for f in x])
print("[RPN] cls per level:", [s.shape for s in cls_scores])
print("[RPN] reg per level:", [b.shape for b in bbox_preds])
```

**在 `StandardRoIHead.forward_train()/predict()` 中**
```python
print("[RoI] #proposals:", sum(len(p) for p in rpn_results_list))
print("[RoI] roi_feats:", roi_feats.shape)
print("[RoI] cls_score:", cls_score.shape, "bbox_pred:", bbox_pred.shape)
```

**在 NMS 前（per image）**
```python
print("[Post] bboxes:", bboxes.shape, "scores:", scores.shape)
```

---

## 9. 阅读源码的推荐顺序

1. `mmdet/models/detectors/two_stage.py`  
2. `mmdet/models/rpn_heads/rpn_head.py`  
3. `mmdet/models/roi_heads/standard_roi_head.py`  
4. `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py`  
5. `mmdet/models/task_modules/assigners/*`, `samplers/*`, `coders/delta_xywh_bbox_coder.py`  
6. `mmdet/structures/bbox/ops.py`（`multiclass_nms`）
