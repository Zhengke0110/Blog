---
title: 直方图与模板匹配
date: 2025-07-02
type: notes-opencv
---

## 核心概念

直方图是图像像素强度分布的图形表示，提供了图像中像素值分布的统计信息。模板匹配是一种在较大图像中搜索和查找模板图像位置的技术。

## 流程图

```mermaid
graph LR
    A[原始图像] --> B[计算直方图]
    B --> C[统计像素分布]
    C --> D[分析图像特征]
    D --> E[模板匹配]
    E --> F[目标检测]

    style A fill:#FF3D71,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style B fill:#00D4FF,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style C fill:#00C851,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style D fill:#FF9500,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style E fill:#9C27B0,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style F fill:#E91E63,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
```

## 位运算

### `cv2.bitwise_and(src1, src2, dst=None, mask=None)`

> **功能**: 对两个图像进行位 AND 运算
>
> **参数**:
>
> - `src1`: 第一个输入图像 (numpy.ndarray)
> - `src2`: 第二个输入图像 (numpy.ndarray)
> - `dst`: 输出图像（可选）
> - `mask`: 掩模图像（可选）
>
> **返回值**: 位 AND 运算结果 (numpy.ndarray)

## 绘图函数

### `cv2.rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0)`

> **功能**: 在图像上绘制矩形
>
> **参数**:
>
> - `img`: 目标图像 (numpy.ndarray)
> - `pt1`: 矩形左上角坐标 (tuple): (x, y)
> - `pt2`: 矩形右下角坐标 (tuple): (x, y)
> - `color`: 矩形颜色 (tuple): (B, G, R) 或标量
> - `thickness`: 线条粗细，-1 表示填充 (int)
> - `lineType`: 线条类型
> - `shift`: 坐标点小数位数
>
> **返回值**: None（直接修改输入图像）

## 直方图相关函数

### `cv2.calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=False)`

> **功能**: 计算图像直方图
>
> **参数**:
>
> - `images`: 输入图像列表 (list[numpy.ndarray])
> - `channels`: 通道索引列表 (list[int])
>   - 灰度图：[0]
>   - 彩色图：[0], [1], [2] 分别对应 B, G, R 通道
> - `mask`: 掩模图像 (numpy.ndarray) 或 None
> - `histSize`: 直方图 bins 数量列表 (list[int])
> - `ranges`: 像素值范围列表 (list[float])
> - `hist`: 输出直方图（可选）
> - `accumulate`: 是否累积直方图（可选）
>
> **返回值**: 直方图数组 (numpy.ndarray)

### `cv2.equalizeHist(src, dst=None)`

> **功能**: 直方图均衡化
>
> **参数**:
>
> - `src`: 输入灰度图像 (numpy.ndarray)
> - `dst`: 输出图像（可选）
>
> **返回值**: 均衡化后的图像 (numpy.ndarray)

### `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`

> **功能**: 创建 CLAHE（对比度受限的自适应直方图均衡化）对象
>
> **参数**:
>
> - `clipLimit`: 对比度限制阈值 (float)
>   - 值越大，对比度增强越明显
>   - 典型值：1.0-4.0
> - `tileGridSize`: 图像分块大小 (tuple): (width, height)
>   - 较小的块：更强的局部对比度
>   - 较大的块：更平滑的过渡
>
> **返回值**: CLAHE 对象
>
> **CLAHE 对象方法**:
>
> - `apply(src)`: 应用 CLAHE 到图像

## 模板匹配函数

### `cv2.matchTemplate(image, templ, method, result=None, mask=None)`

> **功能**: 模板匹配
>
> **参数**:
>
> - `image`: 搜索图像 (numpy.ndarray)
> - `templ`: 模板图像 (numpy.ndarray)
> - `method`: 匹配方法 (int)
>   - `cv2.TM_CCOEFF`: 相关系数匹配
>   - `cv2.TM_CCOEFF_NORMED`: 归一化相关系数匹配（推荐）
>   - `cv2.TM_CCORR`: 相关匹配
>   - `cv2.TM_CCORR_NORMED`: 归一化相关匹配
>   - `cv2.TM_SQDIFF`: 平方差匹配
>   - `cv2.TM_SQDIFF_NORMED`: 归一化平方差匹配
> - `result`: 输出结果矩阵（可选）
> - `mask`: 模板掩模（可选）
>
> **返回值**: 匹配结果矩阵 (numpy.ndarray)

### `cv2.minMaxLoc(src, mask=None)`

> **功能**: 查找矩阵中的最大值和最小值及其位置
>
> **参数**:
>
> - `src`: 输入单通道矩阵 (numpy.ndarray)
> - `mask`: 掩模矩阵（可选）
>
> **返回值**: 四元组 (minVal, maxVal, minLoc, maxLoc)
>
> - `minVal`: 最小值 (float)
> - `maxVal`: 最大值 (float)
> - `minLoc`: 最小值位置 (tuple): (x, y)
> - `maxLoc`: 最大值位置 (tuple): (x, y)

## 基础实现示例

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 显示图像的辅助函数
def cv_show(name, img):
    """显示图像的辅助函数"""
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间，毫秒级，0表示任意键终止
    cv2.destroyAllWindows()

# 基础使用示例
def basic_example():
    """基础使用示例"""
    # 读取图像 - OpenCV 4.x 推荐使用完整路径
    img = cv2.imread('/images/notes/opencv/lena.jpg', cv2.IMREAD_COLOR)        # 读取彩色图像
    gray = cv2.imread('/images/notes/opencv/lena.jpg', cv2.IMREAD_GRAYSCALE)   # 读取灰度图像

    # 检查图像是否成功读取
    if img is None or gray is None:
        print("无法读取图像文件")
        return

    # 颜色空间转换
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 直方图计算
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # 直方图均衡化
    equ = cv2.equalizeHist(gray)

    # CLAHE (对比度受限的自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)

    # 模板匹配
    template = cv2.imread('/images/notes/opencv/face.jpg', cv2.IMREAD_GRAYSCALE)
    if template is not None:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 绘制矩形 - 确保坐标类型正确
        h, w = template.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    return img, gray, hist, equ, clahe_img

# 运行基础示例
if __name__ == "__main__":
    basic_example()
```

## 掩模操作

掩模允许我们只统计图像特定区域的直方图：

```mermaid
graph LR
    A[原始图像] --> B[创建掩模]
    B --> C[应用掩模]
    C --> D[计算区域直方图]

    style A fill:#E91E63,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style B fill:#FF5722,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style C fill:#8BC34A,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style D fill:#9C27B0,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
```
