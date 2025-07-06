---
title: Harris 角点检测
date: 2025-07-02
type: notes-opencv
---

![Harris角点检测示意图](/public/images/notes/opencv/harris_1.png)

## 核心原理

Harris 角点检测基于图像的**局部自相关函数**，通过分析像素点周围的梯度变化来识别角点。角点具有以下特性：

- 在多个方向上都有较大的梯度变化
- 是图像中的稳定特征点
- 对噪声具有一定的鲁棒性

![Harris角点检测原理](/public/images/notes/opencv/harris_2.png)

## 数学公式

### 自相关函数

$$E(u,v) = \sum_{x,y} w(x,y)[I(x+u,y+v) - I(x,y)]^2$$

其中：

- $w(x,y)$ 是窗口函数（通常为高斯函数）
- $I(x,y)$ 是图像强度函数
- $(u,v)$ 是位移量

### Harris 矩阵（结构张量）

$$M = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$$

其中：

- $I_x = \frac{\partial I}{\partial x}$，$I_y = \frac{\partial I}{\partial y}$ 是图像梯度

### Harris 响应函数

$$R = \det(M) - k \cdot \text{trace}(M)^2$$
$$R = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2$$

其中：

- $k \in [0.04, 0.06]$ 是 Harris 参数
- $\lambda_1, \lambda_2$ 是矩阵 $M$ 的特征值

### 角点判定条件

- $R > 0$：角点
- $R < 0$：边缘
- $R \approx 0$：平坦区域

## 流程图

```mermaid
graph LR
    A[输入图像] --> B[转换为灰度图]
    B --> C[计算图像梯度]
    C --> D[构建Harris矩阵]
    D --> E[计算响应函数R]
    E --> F[阈值化处理]
    F --> G[非极大值抑制]
    G --> H[标记角点]

    style A fill:#FF3D71,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style B fill:#00D4FF,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style C fill:#00C851,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style D fill:#FF9500,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style E fill:#9C27B0,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style F fill:#E91E63,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style G fill:#FF5722,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style H fill:#4CAF50,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
```

## OpenCV 实现

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 基本Harris角点检测
def harris_corner_detection(image_path):
    """Harris角点检测基本实现"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris角点检测
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # 标记角点
    img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 红色标记

    return img

# 使用示例    img_path = '/public/images/notes/opencv/test_1.jpg'
result = harris_corner_detection(img_path)
if result is not None:
    cv2.imshow('Harris Corners', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 函数详解

#### `cv2.cornerHarris(src, blockSize, ksize, k, dst=None, borderType=cv2.BORDER_DEFAULT)`

> **功能**: 执行 Harris 角点检测
>
> **参数**:
>
> - `src`: 输入的灰度图像 (numpy.ndarray)
> - `blockSize`: 角点检测窗口大小 (int)
>   - 取值范围：通常为 2-7
>   - 较小值：检测更多细节，但可能包含噪声
>   - 较大值：检测更稳定的角点
> - `ksize`: Sobel 算子窗口大小 (int)
>   - 取值范围：1, 3, 5, 7
>   - 必须为奇数
>   - 影响梯度计算的精度
> - `k`: Harris 响应函数参数 (float)
>   - 取值范围：0.04-0.06
>   - 影响角点检测的灵敏度
> - `dst`: 输出图像 (可选)
> - `borderType`: 边界类型 (int, 可选)
>
> **返回值**: Harris 响应图像 (numpy.ndarray)，数据类型为 float32
>
> **注意**: 返回的是响应值图像，需要进一步处理才能获得角点坐标

#### `cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, corners=None, mask=None, blockSize=3, useHarrisDetector=False, k=0.04)`

> **功能**: 检测图像中的强角点，可以使用 Harris 检测器
>
> **参数**:
>
> - `image`: 输入的灰度图像 (numpy.ndarray)
> - `maxCorners`: 返回的最大角点数 (int)
> - `qualityLevel`: 角点质量水平 (float)
>   - 取值范围：0-1
>   - 相对于最强角点的质量比例
> - `minDistance`: 角点之间的最小欧氏距离 (float)
> - `corners`: 输出角点数组 (可选)
> - `mask`: 感兴趣区域掩码 (可选)
> - `blockSize`: 角点检测窗口大小 (int, 默认 3)
> - `useHarrisDetector`: 是否使用 Harris 检测器 (bool, 默认 False)
> - `k`: Harris 检测器参数 (float, 默认 0.04)
>
> **返回值**: 角点坐标数组 (numpy.ndarray)，形状为(N, 1, 2)

## 参数调优与示例

### 不同参数组合比较

```python
def harris_parameter_comparison(image_path):
    """Harris角点检测参数比较"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 不同参数组合的测试
    parameters = [
        {'blockSize': 2, 'ksize': 3, 'k': 0.04, 'title': 'blockSize=2, k=0.04'},
        {'blockSize': 3, 'ksize': 3, 'k': 0.05, 'title': 'blockSize=3, k=0.05'},
        {'blockSize': 5, 'ksize': 5, 'k': 0.06, 'title': 'blockSize=5, k=0.06'},
    ]

    plt.figure(figsize=(15, 5))

    for i, params in enumerate(parameters):
        # Harris角点检测
        dst = cv2.cornerHarris(gray,
                              blockSize=params['blockSize'],
                              ksize=params['ksize'],
                              k=params['k'])

        # 创建结果图像
        img_copy = img.copy()
        img_copy[dst > 0.01 * dst.max()] = [0, 0, 255]

        # 显示结果
        plt.subplot(1, 3, i+1)
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.title(params['title'])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 使用示例
harris_parameter_comparison('/public/images/notes/opencv/test_1.jpg')
```

### 改进的 Harris 角点检测

```python
def enhanced_harris_detection(image_path, threshold_ratio=0.01):
    """增强版Harris角点检测"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 膨胀操作，增强角点
    dst = cv2.dilate(dst, None)

    # 找到角点位置
    corners = np.where(dst > threshold_ratio * dst.max())

    # 绘制角点
    result = img.copy()
    for y, x in zip(corners[0], corners[1]):
        cv2.circle(result, (x, y), 3, (0, 255, 0), -1)

    return result, len(corners[0])

# 使用示例
result, corner_count = enhanced_harris_detection('/public/images/notes/opencv/test_1.jpg')
if result is not None:
    print(f"检测到 {corner_count} 个角点")
    cv2.imshow('Enhanced Harris Corners', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

![Harris角点检测参数对比](/public/images/notes/opencv/harris_3.png)

## 优缺点分析

### 优点

1. **计算效率高**：相比其他特征检测算法，Harris 检测速度快
2. **对噪声鲁棒**：对一定程度的噪声具有抵抗能力
3. **理论基础扎实**：基于图像梯度的数学原理清晰
4. **实现简单**：算法逻辑简单，易于理解和实现

### 缺点

1. **尺度敏感**：对图像缩放敏感，缺乏尺度不变性
2. **旋转敏感**：对图像旋转敏感，缺乏旋转不变性
3. **只检测角点**：只能检测角点，无法检测其他类型的特征
4. **无特征描述**：只提供角点位置，不提供特征描述符

## 应用场景

### 适用场景

- **简单几何图形检测**：矩形、三角形等几何形状的角点
- **建筑物检测**：建筑物的角点和边缘
- **实时应用**：对速度要求高的场景
- **预处理步骤**：作为其他算法的预处理步骤

### 不适用场景

- **复杂纹理匹配**：需要更复杂的特征描述
- **尺度变化大**：图像存在显著尺度变化
- **旋转变化大**：图像存在显著旋转变化
- **光照变化大**：光照条件变化剧烈

## 实际应用示例

### 棋盘格角点检测

```python
def detect_chessboard_corners(image_path):
    """棋盘格角点检测"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 方法1：使用Harris角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 方法2：使用goodFeaturesToTrack结合Harris检测器
    corners = cv2.goodFeaturesToTrack(gray,
                                     maxCorners=100,
                                     qualityLevel=0.01,
                                     minDistance=10,
                                     useHarrisDetector=True,
                                     k=0.04)

    # 绘制角点
    result = img.copy()
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel().astype(int)
            cv2.circle(result, (x, y), 3, (0, 255, 0), -1)

    return result

# 使用示例
result = detect_chessboard_corners('/public/images/notes/opencv/chessboard.jpg')
if result is not None:
    cv2.imshow('Chessboard Corners', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 建筑物角点检测

```python
def detect_building_corners(image_path):
    """建筑物角点检测"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 预处理：增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Harris角点检测
    dst = cv2.cornerHarris(enhanced, 3, 3, 0.04)

    # 阈值化和非极大值抑制
    threshold = 0.01 * dst.max()
    corners = np.where(dst > threshold)

    # 绘制角点
    result = img.copy()
    for y, x in zip(corners[0], corners[1]):
        cv2.circle(result, (x, y), 5, (0, 0, 255), -1)

    return result

# 使用示例
result = detect_building_corners('/public/images/notes/opencv/building.jpg')
if result is not None:
    cv2.imshow('Building Corners', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

![棋盘格角点检测示例](/public/images/notes/opencv/harris_4.png)

## 实用技巧与最佳实践

### 角点检测优化技巧

```python
def optimized_harris_detection(image_path, **kwargs):
    """优化的Harris角点检测"""
    # 默认参数
    params = {
        'blockSize': 2,
        'ksize': 3,
        'k': 0.04,
        'threshold_ratio': 0.01,
        'use_clahe': True,
        'dilate_kernel': None
    }
    params.update(kwargs)

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 可选的对比度增强
    if params['use_clahe']:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Harris角点检测
    dst = cv2.cornerHarris(gray,
                          params['blockSize'],
                          params['ksize'],
                          params['k'])

    # 可选的膨胀操作
    if params['dilate_kernel'] is not None:
        dst = cv2.dilate(dst, params['dilate_kernel'])

    # 阈值化处理
    threshold = params['threshold_ratio'] * dst.max()
    corners = np.where(dst > threshold)

    # 创建结果图像
    result = img.copy()
    for y, x in zip(corners[0], corners[1]):
        cv2.circle(result, (x, y), 3, (0, 255, 0), -1)

    return result, len(corners[0])

# 使用示例
result, count = optimized_harris_detection('/public/images/notes/opencv/test_1.jpg',
                                          threshold_ratio=0.02,
                                          use_clahe=True)
if result is not None:
    print(f"检测到 {count} 个角点")
```
