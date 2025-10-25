---
layout: post
title: "Faster R-CNN：结构总览与 MMDetection 源码路径（含张量形状速查）"
date: 2025-10-25
tags:
  - Faster R-CNN
  - MMDetection
  - 目标检测
  - RPN
  - RoIAlign
mathjax: true
---
关键词：Faster R-CNN，RPN，RoIAlign，BBoxHead，MMDetection 源码路径
摘要：一图梳理 Faster R-CNN 的数据流，给出张量形状速查表与 MMDetection 源码定位。并用一个常见尺寸（800×1333，Pad 至 800×1344）给出可落地的数字例子，最后附上调试打印与常见坑排查清单。

<!--more-->
1. 结构总览（数据流一图速览）
Image → Backbone(ResNet) → Neck(FPN P2…P6)
                        → RPN(cls:前景/背景, reg:Δ)
                      proposals(Top-K) ──→  RoIAlign(从FPN选层, 对齐采样7×7)
                                           → BBox Head(分类cls + 回归reg)
                                           → Δ解码 → per-class NMS → det_bboxes, det_labels


RPN：在各金字塔层的每个网格点上，对 A 个锚框做二分类（前景/背景）与 4 维回归（Δ）。

RoIAlign：把 proposals 映射回金字塔相应层，裁成 7×7 的特征块，交给 BBox Head。

BBox Head：输出 cls_score（包含背景类）与 bbox_pred（类相关或类无关回归）。

2. 张量形状速查（通用形式）

FPN（单图 N=1）：
P2…P6 的空间尺寸约为 H/4, H/8, H/16, H/32, H/64（对齐到整除，实际按 Pad 后尺寸计算）。

RPN 逐层输出：
rpn_cls: [N, A, Hi, Wi]（sigmoid 二分类）；
rpn_reg: [N, 4A, Hi, Wi]。

RoIAlign：roi_feats: [R, C', 7, 7]。

BBox Head：
cls_score: [R, num_classes + 1]（含背景类）；
bbox_pred: [R, 4]（reg_class_agnostic=True），或 [R, 4 * num_classes]（False）。

3. 具体数字例子（800×1333，Pad→800×1344）

MMDetection 默认把短边缩放到 800，并 Pad 到 32 的倍数。
例：H=800, W=1333 → PadW=1344。

层	stride	空间尺寸 (Hi×Wi)	每层锚框数 A*Hi*Wi（A=3）
P2	4	200×336	3×200×336 = 201,600
P3	8	100×168	50,400
P4	16	50×84	12,600
P5	32	25×42	3,150
P6	64	13×21	819
合计			268,569 个 anchors

说明：P6 由 P5 再下采样得到，卷积 k=3,s=2,p=1 下输出 ⌊(in+2p−k)/s⌋+1，所以 25×42 → 13×21。
RPN 会先按 nms_pre（常见 1000）做 每层预筛，然后 NMS、拼成 proposals（常见 max_per_img≈1000）。

4. MMDetection 源码定位（3.x）

总装配（两阶段框架）
mmdet/models/detectors/two_stage.py（TwoStageDetector）
mmdet/models/detectors/faster_rcnn.py（FasterRCNN 仅做配置化继承）

RPN
头：mmdet/models/rpn_heads/rpn_head.py
先验框：mmdet/models/task_modules/prior_generators/anchor_generator.py
Δ编解码：mmdet/models/task_modules/coders/delta_xywh_bbox_coder.py

RoI / BBox 分支
RoIHead：mmdet/models/roi_heads/standard_roi_head.py
RoIAlign：mmdet/models/roi_layers/roi_align.py
提取器：mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
BBoxHead：mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py

训练时样本分配/采样
分配器：mmdet/models/task_modules/assigners/max_iou_assigner.py
采样器：mmdet/models/task_modules/samplers/random_sampler.py

后处理
NMS：mmdet/structures/bbox/ops.py（multiclass_nms）

5. 训练与推理的调用路径（极简伪代码）

训练（TwoStageDetector.loss）：

feats = extract_feat(img)  # backbone -> FPN
# RPN
rpn_losses, rpn_results_list = rpn_head.loss_and_predict(feats, batch_data, rpn_train_cfg)
# RoI/BBox
roi_losses = roi_head.loss(feats, rpn_results_list, batch_data, rcnn_train_cfg)
loss = {**rpn_losses, **roi_losses}


推理（TwoStageDetector.predict）：

feats = extract_feat(img)
proposals = rpn_head.predict(feats, rpn_test_cfg)
det_bboxes, det_labels = roi_head.predict(feats, proposals, rcnn_test_cfg, rescale=True)

6. “配置 → 形状”的快速映射
model = dict(
  type='FasterRCNN',
  neck=dict(type='FPN', num_outs=5),  # → P2..P6, strides=[4,8,16,32,64]
  rpn_head=dict(
    type='RPNHead',
    anchor_generator=dict(scales=[8], ratios=[0.5,1.0,2.0], strides=[4,8,16,32,64]),
    # → 每层 A = len(scales)*len(ratios) = 3
    bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_stds=[0.1,0.1,0.2,0.2]),
  ),
  roi_head=dict(
    type='StandardRoIHead',
    bbox_roi_extractor=dict(type='SingleRoIExtractor',
                            roi_layer=dict(type='RoIAlign', output_size=7),
                            featmap_strides=[4,8,16,32]),
    bbox_head=dict(
      type='Shared2FCBBoxHead', num_classes=NUM,
      reg_class_agnostic=False,  # False→bbox_pred.shape[1]=4*NUM；True→4
    )
  )
)


cls_score.shape = [R, NUM + 1]（多分类 + 背景）。

bbox_pred.shape = [R, 4 * NUM]（类相关回归，默认 False），或 [R, 4]（类无关回归）。

RPN 输出通道：rpn_cls: A，rpn_reg: 4A；A = len(scales) * len(ratios)。

Δ 编码（训练用）与解码（推理解）：
$$
t_x=\frac{x-x_a}{w_a},\quad t_y=\frac{y-y_a}{h_a},\quad
t_w=\log\frac{w}{w_a},\quad t_h=\log\frac{h}{h_a}
$$


7. 常见坑与快速排查

背景通道：cls_score=[R, NUM+1]；别忘了背景那一列。

Pad 到 32 的倍数：尺寸计算用 Pad 后 的 H_padded×W_padded，否则各层 Hi、Wi 会对不上。

锚框总数巨大：理论 anchor≈26.86 万个（见上表），实际 RPN 会按 nms_pre 先筛（如 1000/2000），再 NMS。

reg_class_agnostic：决定 bbox_pred 的第二维是 4 还是 4*NUM。

类别 id 与数据集：COCO 从 1 开始，背景是 0；自定义数据要检查类别顺序与 num_classes。

阈值导致误检/漏检：score_thr、nms.iou_threshold、max_per_img 影响较大。

训练/测试增强不一致：如 Resize、Normalize 的均值/方差要一致。

8. 最少量调试打印（插到本地源码里就能看）

RPNHead.forward()：

print("[RPN] feat shapes:", [f.shape for f in x])
print("[RPN] cls per level:", [s.shape for s in cls_scores])
print("[RPN] reg per level:", [b.shape for b in bbox_preds])


StandardRoIHead.forward_train()/predict()：

print("[RoI] #proposals:", sum(len(p) for p in rpn_results_list))
print("[RoI] roi_feats:", roi_feats.shape)
print("[RoI] cls_score:", cls_score.shape, "bbox_pred:", bbox_pred.shape)


NMS 前：

print("[Post] bboxes:", bboxes.shape, "scores:", scores.shape)  # per image

9. 阅读顺序（建议）

1）detectors/two_stage.py → 2）rpn_heads/rpn_head.py → 3）roi_heads/standard_roi_head.py
→ 4）bbox_heads/convfc_bbox_head.py → 5）assigners/samplers/coders → 6）bbox/ops.py (NMS)

