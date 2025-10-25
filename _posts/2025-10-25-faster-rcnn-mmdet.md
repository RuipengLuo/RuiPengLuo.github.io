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
<!-- MathJax：最简单可用法——在文章里直接引入脚本（只影响本页） -->
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
  svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

> 关键词：Faster R-CNN，RPN，RoIAlign，BBoxHead，MMDetection 源码路径

## 1. 结构一图速览
Image -> Backbone -> FPN -> RPN -> Proposals -> RoIAlign -> BBox Head (cls+reg) -> NMS -> dets


## 2. 关键张量形状速查
- FPN: `[N, C, H/4.., W/4..]`
- RPN: `cls [N, A, Hi, Wi]`, `reg [N, 4A, Hi, Wi]`
- RoIAlign: `[R, C', 7, 7]`
- BBoxHead: 
  - `cls_score [R, num_classes+1]`（含背景）
  - `bbox_pred [R, 4*num_classes]`（若 `reg_class_agnostic=False`），或 `[R, 4]`（若 `True`）

## 3. 在 MMDetection 的源码位置（3.x）
- Detector: `mmdet/models/detectors/two_stage.py`
- FasterRCNN: `mmdet/models/detectors/faster_rcnn.py`
- RPNHead: `mmdet/models/rpn_heads/rpn_head.py`
- RoIHead: `mmdet/models/roi_heads/standard_roi_head.py`
- RoIAlign: `mmdet/models/roi_layers/roi_align.py`
- BBoxHead: `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py`
- NMS: `mmdet/structures/bbox/ops.py`

## 4. 配置到形状映射（示例）
设 `num_classes = 2`：
- `cls_score = [R, 3]`（2 类 + 背景）
- `reg_class_agnostic=False` → `bbox_pred = [R, 8]`

**Δ编码：**
$$
t_x=\frac{x-x_a}{w_a},\quad t_y=\frac{y-y_a}{h_a},\quad 
t_w=\log\frac{w}{w_a},\quad t_h=\log\frac{h}{h_a}
$$

## 5. 训练/推理调用顺序（极简伪代码）
```python
feats = backbone_neck(img)                # [P2..P5]
proposals = rpn_head.predict(feats)       # Top-K + NMS
roi_feats = RoIAlign(feats, proposals)    # 7x7
cls_score, bbox_pred = bbox_head(roi_feats)
dets = decode_and_nms(cls_score, bbox_pred)

