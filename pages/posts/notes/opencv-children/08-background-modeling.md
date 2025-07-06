---
title: 背景建模
date: 2025-07-06
type: notes-opencv
---

## 核心概念

背景建模是从视频序列中分离前景目标和背景的关键技术，广泛应用于视频监控、目标跟踪和运动分析。

![背景建模示意图](/images/notes/opencv/bg_2.png)

## 流程图

```mermaid
graph LR
    A[视频输入] --> B{选择算法}
    B --> C[帧差法]
    B --> D[混合高斯模型]

    C --> E[像素差分]
    E --> F[阈值判断]
    F --> G[前景掩码]

    D --> H[高斯模型训练]
    H --> I[参数更新]
    I --> J[GMM匹配]
    J --> K[前景检测]

    G --> L[轮廓检测]
    K --> L
    L --> M[目标框定]

    style A fill:#FF3D71,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style B fill:#00D4FF,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style C fill:#00C851,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style D fill:#FF9500,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style E fill:#9C27B0,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style F fill:#E91E63,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style G fill:#FF5722,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style H fill:#4CAF50,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style I fill:#2196F3,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style J fill:#FF6F00,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style K fill:#8BC34A,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style L fill:#3F51B5,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style M fill:#009688,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
```

## 帧差法 (Frame Differencing)

### 数学原理

对连续两帧图像进行差分运算：

$$D(x,y) = |I_t(x,y) - I_{t-1}(x,y)|$$

其中：

- $I_t(x,y)$ 为当前帧像素值
- $I_{t-1}(x,y)$ 为前一帧像素值
- 判断条件：$D(x,y) > \tau$ (阈值)

### 算法步骤

1. **图像预处理**: 灰度化、去噪
2. **帧差计算**: $D(x,y) = |I_t(x,y) - I_{t-1}(x,y)|$
3. **二值化**: $B(x,y) = \begin{cases} 255 & \text{if } D(x,y) > \tau \\ 0 & \text{otherwise} \end{cases}$
4. **形态学处理**: 去除噪声和填充空洞

### 优缺点

**优点**:

- 计算简单，实时性好，内存占用小
- 对快速运动物体检测效果好
- 实现简单，易于理解

**缺点**:

- 产生噪声和空洞，对光照变化敏感
- 无法处理静止目标
- 容易受到摄像头抖动影响

## 混合高斯模型 (Gaussian Mixture Model)

![混合高斯模型原理](/images/notes/opencv/bg_3.png)

### 数学原理

每个像素点的背景分布建模为 K 个高斯分布的混合：

$$P(X_t) = \sum_{k=1}^{K} \omega_{k,t} \cdot \mathcal{N}(X_t; \mu_{k,t}, \Sigma_{k,t})$$

其中：

- $\omega_{k,t}$ 是第 k 个高斯分量的权重，满足 $\sum_{k=1}^{K} \omega_{k,t} = 1$
- $\mu_{k,t}$ 是均值向量
- $\Sigma_{k,t}$ 是协方差矩阵
- $\mathcal{N}$ 是多元高斯分布函数

高斯分布函数定义为：
$$\mathcal{N}(X; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)$$

![高斯分布示意](/images/notes/opencv/bg_4.png)

### 参数更新公式

对于匹配的高斯分量，参数更新为：
$$\omega_{k,t} = (1-\alpha)\omega_{k,t-1} + \alpha M_{k,t}$$
$$\mu_{k,t} = (1-\rho)\mu_{k,t-1} + \rho X_t$$
$$\sigma_{k,t}^2 = (1-\rho)\sigma_{k,t-1}^2 + \rho(X_t-\mu_{k,t})^T(X_t-\mu_{k,t})$$

其中：

- $\alpha$ 为学习率 (通常取 0.005-0.01)
- $\rho = \alpha \cdot \mathcal{N}(X_t; \mu_{k,t}, \sigma_{k,t}^2 I)$
- $M_{k,t}$ 为匹配指示变量

### 前景检测判断

像素点与模型匹配条件：
$$|X_t - \mu_{k,t}| < T \cdot \sigma_{k,t}$$

其中 $T$ 通常取 2.5，对应 99.3%的置信区间。

![背景建模结果](/images/notes/opencv/bg_5.png)

### 函数详解

#### `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)`

> **功能**: 创建基于混合高斯模型的背景减除器
>
> **参数**:
>
> - `history`: 用于建立背景模型的历史帧数 (int)
>   - 取值范围：通常为 200-500
>   - 较大值：背景模型更稳定，但适应性较差
>   - 较小值：背景模型适应性好，但可能不够稳定
> - `varThreshold`: 马氏距离阈值 (float)
>   - 取值范围：通常为 16-50
>   - 较大值：检测到更多前景，但噪声也增加
>   - 较小值：检测结果更准确，但可能遗漏前景
> - `detectShadows`: 是否检测阴影 (bool)
>   - True: 检测阴影并标记为灰色值 (127)
>   - False: 不检测阴影
>
> **返回值**: BackgroundSubtractorMOG2 对象

#### `cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)`

> **功能**: 创建基于 K-最近邻的背景减除器
>
> **参数**:
>
> - `history`: 用于建立背景模型的历史帧数 (int)
> - `dist2Threshold`: K-最近邻距离阈值 (float)
>   - 取值范围：通常为 300-500
>   - 影响前景检测的敏感度
> - `detectShadows`: 是否检测阴影 (bool)
>
> **返回值**: BackgroundSubtractorKNN 对象

#### `backgroundSubtractor.apply(image, learningRate=-1)`

> **功能**: 对输入图像应用背景减除
>
> **参数**:
>
> - `image`: 输入图像 (numpy.ndarray)
> - `learningRate`: 学习率 (float)
>   - -1: 使用默认学习率 (1/history)
>   - 0: 不更新背景模型
>   - 0-1: 自定义学习率
>
> **返回值**: 前景掩码 (numpy.ndarray)，前景为 255，背景为 0

#### `backgroundSubtractor.getBackgroundImage()`

> **功能**: 获取当前背景模型图像
>
> **参数**: 无
>
> **返回值**: 背景图像 (numpy.ndarray)

### 基础实现

```python
import cv2
import numpy as np

def basic_background_subtraction(video_path):
    """基础背景减除实现"""
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    # 创建背景减除器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 应用背景减除
        fg_mask = bg_subtractor.apply(frame)

        # 获取背景图像
        bg_image = bg_subtractor.getBackgroundImage()

        # 显示结果
        cv2.imshow('Original', frame)
        cv2.imshow('Foreground Mask', fg_mask)
        if bg_image is not None:
            cv2.imshow('Background Model', bg_image)

        # 按 'q' 键退出
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
basic_background_subtraction('videos/traffic.mp4')
```

### 帧差法实现

```python
def frame_differencing(video_path, threshold=30):
    """帧差法背景减除"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 转换为灰度图
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 计算帧差
        diff = cv2.absdiff(prev_gray, curr_gray)

        # 阈值化
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 显示结果
        cv2.imshow('Original', curr_frame)
        cv2.imshow('Frame Difference', diff)
        cv2.imshow('Binary Mask', binary)

        # 更新前一帧
        prev_gray = curr_gray.copy()

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
frame_differencing('videos/traffic.mp4', threshold=25)
```

### 参数优化示例

```python
def optimized_background_subtraction(video_path):
    """优化的背景减除"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    # 创建多个背景减除器进行比较
    bg_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True
    )

    bg_knn = cv2.createBackgroundSubtractorKNN(
        history=500,
        dist2Threshold=400,
        detectShadows=True
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 应用不同的背景减除器
        fg_mog2 = bg_mog2.apply(frame)
        fg_knn = bg_knn.apply(frame)

        # 形态学操作优化结果
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 对MOG2结果进行优化
        fg_mog2_clean = cv2.morphologyEx(fg_mog2, cv2.MORPH_OPEN, kernel)
        fg_mog2_clean = cv2.morphologyEx(fg_mog2_clean, cv2.MORPH_CLOSE, kernel)

        # 对KNN结果进行优化
        fg_knn_clean = cv2.morphologyEx(fg_knn, cv2.MORPH_OPEN, kernel)
        fg_knn_clean = cv2.morphologyEx(fg_knn_clean, cv2.MORPH_CLOSE, kernel)

        # 显示结果
        cv2.imshow('Original', frame)
        cv2.imshow('MOG2 Raw', fg_mog2)
        cv2.imshow('MOG2 Clean', fg_mog2_clean)
        cv2.imshow('KNN Raw', fg_knn)
        cv2.imshow('KNN Clean', fg_knn_clean)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
optimized_background_subtraction('videos/traffic.mp4')
```

## 算法比较

### 不同算法特点对比

| 算法         | 计算复杂度 | 内存占用 | 适应性 | 鲁棒性 | 适用场景               |
| ------------ | ---------- | -------- | ------ | ------ | ---------------------- |
| 帧差法       | 低         | 低       | 差     | 差     | 简单场景，实时性要求高 |
| 混合高斯模型 | 中         | 中       | 好     | 好     | 复杂场景，光照变化     |
| KNN          | 高         | 高       | 很好   | 很好   | 复杂场景，高精度要求   |

## 应用场景

### 视频监控

```python
def video_surveillance(video_path):
    """视频监控应用"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # 创建背景减除器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=True
    )

    # 设置最小轮廓面积
    min_area = 500

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景减除
        fg_mask = bg_subtractor.apply(frame)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制边界框
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Area: {int(area)}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示结果
        cv2.imshow('Surveillance', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
video_surveillance('videos/surveillance.mp4')
```

### 交通流量统计

```python
def traffic_counting(video_path):
    """交通流量统计"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # 创建背景减除器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=20,
        detectShadows=True
    )

    # 设置计数线
    line_y = 300  # 计数线的y坐标
    vehicle_count = 0
    tracked_objects = {}
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景减除
        fg_mask = bg_subtractor.apply(frame)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制计数线
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 0, 0), 2)

        # 检测车辆
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 最小车辆面积
                x, y, w, h = cv2.boundingRect(contour)
                center_y = y + h // 2

                # 检查是否穿过计数线
                if abs(center_y - line_y) < 5:  # 允许一定的误差
                    vehicle_count += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示计数
        cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 显示结果
        cv2.imshow('Traffic Counting', frame)
        cv2.imshow('Foreground Mask', fg_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"总车辆数: {vehicle_count}")

# 使用示例
traffic_counting('videos/traffic.mp4')
```
