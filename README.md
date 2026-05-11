# LiquidIdentification

Bottle liquid-level identification with YOLO OBB.

## 中文说明

本项目使用 YOLO OBB 模型识别瓶中液体液位

### 数据集

OBB 数据集配置文件是：

```bash
bottle_obb.yaml
```

配置文件中的数据集路径为：

```bash
/mnt/e/LiquidIdentification/bottleDataset
```

数据集目录结构应为：

```text
bottleDataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

每个标签文件使用 YOLO OBB 格式：

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

其中 `class` 是类别编号，后面 8 个数是旋转框 4 个顶点的归一化坐标。

### 训练

默认使用 `yolo11m-obb.pt` 训练：

```bash
python train_obb.py
```

常用参数示例：

```bash
python train_obb.py --epochs 200 --imgsz 640 --batch 2 --device cpu
```

如果 GPU 可用，可以使用：

```bash
python train_obb.py --device 0
```

如果训练中断，可以继续训练：

```bash
python train_obb.py --resume
```

训练结果默认保存在 `runs/obb/` 下。脚本会自动使用带时间戳的目录名，避免覆盖旧结果，例如：

```bash
runs/obb/bottle_yolo11m_obb_20260511_103000
```

如果想手动指定目录名：

```bash
python train_obb.py --name bottle_yolo11m_obb
```

如果确认要写入已有目录：

```bash
python train_obb.py --name bottle_yolo11m_obb --exist-ok
```

训练完成后，常见输出文件含义如下：

```text
runs/obb/<run_name>/
  weights/
    best.pt      # 验证集指标最好的模型权重，通常用于最终预测
    last.pt      # 最后一个 epoch 保存的模型权重，可用于继续训练
  args.yaml      # 本次训练使用的参数配置
  results.csv    # 每个 epoch 的 loss、precision、recall、mAP 等指标表格
  results.png    # 训练指标曲线总览图
  BoxP_curve.png # Precision 曲线
  BoxR_curve.png # Recall 曲线
  BoxF1_curve.png
                # F1 曲线，用于观察不同置信度阈值下的综合表现
  BoxPR_curve.png
                # Precision-Recall 曲线
  confusion_matrix.png
                # 混淆矩阵，查看类别之间的误判情况
  confusion_matrix_normalized.png
                # 归一化混淆矩阵
  labels.jpg     # 数据集中标签分布的可视化
  train_batch*.jpg
                # 训练 batch 可视化，用来检查图片和标注是否正常
  val_batch*_labels.jpg
                # 验证集真实标签可视化
  val_batch*_pred.jpg
                # 验证集预测结果可视化
```

一般最常用的是：

```text
weights/best.pt
results.csv
results.png
val_batch*_pred.jpg
```

如果只是拿训练好的模型去预测，优先使用：

```bash
runs/obb/<run_name>/weights/best.pt
```

### 预测

使用预训练权重预测单张图片：

```bash
yolo predict model=yolo11m-obb.pt source=/mnt/e/LiquidIdentification/testDataset/images/image4.jpg
```

预测整个文件夹：

```bash
yolo predict model=yolo11m-obb.pt source=/mnt/e/LiquidIdentification/testDataset/images
```

训练完成后，可以使用最佳权重进行预测：

```bash
yolo predict model=runs/obb/<run_name>/weights/best.pt source=/mnt/e/LiquidIdentification/testDataset/images
```

## English

## Environment

This project is currently run from WSL with the `torchforge` virtual environment:

```bash
cd /mnt/e/LiquidIdentification
source /root/envs/torchforge/bin/activate
```

## Dataset

The OBB dataset config is:

```bash
bottle_obb.yaml
```

It points to:

```bash
/mnt/e/LiquidIdentification/bottleDataset
```

Expected layout:

```text
bottleDataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each label file should use YOLO OBB format:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

## Train

Train with `yolo11m-obb.pt`:

```bash
python train_obb.py
```

Common options:

```bash
python train_obb.py --epochs 200 --imgsz 640 --batch 2 --device cpu
```

Use GPU if available:

```bash
python train_obb.py --device 0
```

Resume an interrupted run:

```bash
python train_obb.py --resume
```

Training outputs are saved under `runs/obb/`. By default, the script creates a timestamped run directory to avoid overwriting old results, for example:

```bash
runs/obb/bottle_yolo11m_obb_20260511_103000
```

To choose a fixed run name:

```bash
python train_obb.py --name bottle_yolo11m_obb
```

To write into an existing run directory:

```bash
python train_obb.py --name bottle_yolo11m_obb --exist-ok
```

Common training outputs:

```text
runs/obb/<run_name>/
  weights/
    best.pt      # Best checkpoint on the validation set
    last.pt      # Last epoch checkpoint, useful for resume
  args.yaml      # Training arguments
  results.csv    # Epoch-by-epoch metrics
  results.png    # Training metric plots
  BoxP_curve.png # Precision curve
  BoxR_curve.png # Recall curve
  BoxF1_curve.png
                # F1 curve
  BoxPR_curve.png
                # Precision-recall curve
  confusion_matrix.png
                # Confusion matrix
  confusion_matrix_normalized.png
                # Normalized confusion matrix
  labels.jpg     # Label distribution visualization
  train_batch*.jpg
                # Training batch visualization
  val_batch*_labels.jpg
                # Validation labels visualization
  val_batch*_pred.jpg
                # Validation predictions visualization
```

For prediction, usually use:

```bash
runs/obb/<run_name>/weights/best.pt
```

## Predict

Predict one image with the pretrained weight:

```bash
yolo predict model=yolo11m-obb.pt source=/mnt/e/LiquidIdentification/testDataset/images/image4.jpg
```

Predict a folder:

```bash
yolo predict model=yolo11m-obb.pt source=/mnt/e/LiquidIdentification/testDataset/images
```

After training, use the best checkpoint:

```bash
yolo predict model=runs/obb/<run_name>/weights/best.pt source=/mnt/e/LiquidIdentification/testDataset/images
```
