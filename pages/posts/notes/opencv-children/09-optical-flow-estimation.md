---
title: 光流估计
date: 2025-07-06
type: notes-opencv
---

## 核心概念

光流是描述图像中像素运动的瞬时速度矢量场，基于亮度恒定假设来估计物体运动。光流估计在计算机视觉中具有重要应用，包括目标跟踪、运动分析、视频压缩等。

![光流估计原理](/images/notes/opencv/lk_2.png)

## 流程图

```mermaid
graph LR
    A[视频序列] --> B[特征点检测]
    B --> C[Harris/Shi-Tomasi]
    C --> D[Lucas-Kanade算法]
    D --> E[构建方程组]
    E --> F[最小二乘求解]
    F --> G[光流矢量]
    G --> H[目标跟踪]
    H --> I[应用输出]

    style A fill:#FF3D71,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style B fill:#00D4FF,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style C fill:#00C851,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style D fill:#FF9500,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style E fill:#9C27B0,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style F fill:#E91E63,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style G fill:#FF5722,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style H fill:#4CAF50,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
    style I fill:#2196F3,stroke:#FFFFFF,stroke-width:3px,color:#FFFFFF
```

![光流场示意图](/images/notes/opencv/lk_1.png)

## 光流基本假设

1. **亮度恒定假设**:
   $$I(x,y,t) = I(x+dx, y+dy, t+dt)$$

2. **小运动假设**:
   运动位移足够小，可以用一阶泰勒展开近似

3. **空间一致性假设**:
   邻近像素具有相同的运动模式

## 光流约束方程

通过泰勒展开和亮度恒定假设，得到光流约束方程：

$$I(x+dx, y+dy, t+dt) = I(x,y,t) + \frac{\partial I}{\partial x}dx + \frac{\partial I}{\partial y}dy + \frac{\partial I}{\partial t}dt + \epsilon$$

由于 $I(x,y,t) = I(x+dx, y+dy, t+dt)$，可得：

$$\frac{\partial I}{\partial x}\frac{dx}{dt} + \frac{\partial I}{\partial y}\frac{dy}{dt} + \frac{\partial I}{\partial t} = 0$$

简化为：
$$I_x u + I_y v + I_t = 0$$

其中：

- $u = \frac{dx}{dt}, v = \frac{dy}{dt}$ 为光流速度分量
- $I_x = \frac{\partial I}{\partial x}, I_y = \frac{\partial I}{\partial y}, I_t = \frac{\partial I}{\partial t}$ 为图像梯度

## Lucas-Kanade 算法

![Lucas-Kanade算法示意](/images/notes/opencv/lk_3.png)

### 数学推导

假设窗口内所有像素具有相同运动，构建方程组：

$$
\begin{bmatrix}
I_{x1} & I_{y1} \\
I_{x2} & I_{y2} \\
\vdots & \vdots \\
I_{xn} & I_{yn}
\end{bmatrix}
\begin{bmatrix}
u \\ v
\end{bmatrix} =
-\begin{bmatrix}
I_{t1} \\
I_{t2} \\
\vdots \\
I_{tn}
\end{bmatrix}
$$

矩阵形式：$\mathbf{A}\mathbf{d} = \mathbf{b}$

### 最小二乘解

通过最小化残差平方和：
$$\mathbf{d} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b}$$

展开为：
$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} \sum I_x^2 & \sum I_xI_y \\ \sum I_xI_y & \sum I_y^2 \end{bmatrix}^{-1} \begin{bmatrix} -\sum I_xI_t \\ -\sum I_yI_t \end{bmatrix}$$

### 算法可靠性判断

通过特征值判断解的可靠性：

- 设 $\mathbf{Z} = \mathbf{A}^T\mathbf{A} = \begin{bmatrix} \sum I_x^2 & \sum I_xI_y \\ \sum I_xI_y & \sum I_y^2 \end{bmatrix}$
- 特征值：$\lambda_1, \lambda_2$
- 判断条件：$\lambda_2 > \tau$ 且 $\frac{\lambda_1}{\lambda_2} < r$

![Lucas-Kanade结果示例](/images/notes/opencv/lk_4.png)

## OpenCV 实现

### 函数详解

#### `cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status=None, err=None, winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01), flags=0, minEigThreshold=1e-4)`

> **功能**: 使用金字塔 Lucas-Kanade 算法计算稀疏光流
>
> **参数**:
>
> - `prevImg`: 前一帧灰度图像 (numpy.ndarray)
> - `nextImg`: 当前帧灰度图像 (numpy.ndarray)
> - `prevPts`: 前一帧中的特征点坐标 (numpy.ndarray)
>   - 形状为 (N, 1, 2) 的浮点数组
> - `nextPts`: 当前帧中对应的特征点坐标 (numpy.ndarray, 可选)
>   - 如果为 None，将自动计算
> - `status`: 输出状态数组 (numpy.ndarray, 可选)
> - `err`: 输出误差数组 (numpy.ndarray, 可选)
> - `winSize`: 搜索窗口大小 (tuple)
>   - 默认 (21, 21)
>   - 较大值：更鲁棒，但计算量大
>   - 较小值：计算快，但可能不准确
> - `maxLevel`: 金字塔最大层数 (int)
>   - 0: 不使用金字塔
>   - 值越大，能处理更大的运动
> - `criteria`: 终止条件 (tuple)
>   - 通常为 (type, maxCount, epsilon)
> - `flags`: 计算标志 (int, 可选)
> - `minEigThreshold`: 最小特征值阈值 (float)
>   - 用于判断特征点质量
>
> **返回值**:
>
> - `nextPts`: 当前帧中的特征点坐标 (numpy.ndarray)
> - `status`: 状态数组，1 表示找到对应点，0 表示丢失 (numpy.ndarray)
> - `err`: 每个特征点的误差 (numpy.ndarray)

#### `cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance, corners=None, mask=None, blockSize=3, useHarrisDetector=False, k=0.04)`

> **功能**: 检测图像中适合跟踪的角点特征
>
> **参数**:
>
> - `image`: 输入灰度图像 (numpy.ndarray)
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

#### `cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)`

> **功能**: 使用 Farneback 算法计算稠密光流
>
> **参数**:
>
> - `prev`: 前一帧灰度图像 (numpy.ndarray)
> - `next`: 当前帧灰度图像 (numpy.ndarray)
> - `flow`: 输出光流场 (numpy.ndarray, 可选)
> - `pyr_scale`: 金字塔缩放比例 (float)
>   - 典型值：0.5
> - `levels`: 金字塔层数 (int)
>   - 典型值：3
> - `winsize`: 平均窗口大小 (int)
>   - 典型值：15
> - `iterations`: 每层的迭代次数 (int)
>   - 典型值：3
> - `poly_n`: 像素邻域大小 (int)
>   - 典型值：5 或 7
> - `poly_sigma`: 高斯标准差 (float)
>   - 典型值：1.2
> - `flags`: 操作标志 (int)
>
> **返回值**: 光流场 (numpy.ndarray)，形状为 (H, W, 2)

### 基础实现

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lucas_kanade_tracking(video_path):
    """Lucas-Kanade光流跟踪基础实现"""
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    # Lucas-Kanade光流参数
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # 特征点检测参数
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # 读取第一帧
    ret, old_frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return None

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # 检测初始特征点
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # 创建随机颜色用于轨迹绘制
    colors = np.random.randint(0, 255, (100, 3))

    # 创建轨迹图像
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            # 选择好的点
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # 绘制轨迹
                for i, (tr, to) in enumerate(zip(good_new, good_old)):
                    a, b = tr.ravel().astype(int)
                    c, d = to.ravel().astype(int)
                    mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)

                img = cv2.add(frame, mask)

                cv2.imshow('Frame', img)

                # 更新前一帧和点
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
lucas_kanade_tracking('videos/traffic.mp4')
```

### 稠密光流实现

```python
def farneback_optical_flow(video_path):
    """Farneback稠密光流实现"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    # 读取第一帧
    ret, frame1 = cap.read()
    if not ret:
        print("无法读取视频帧")
        return None

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # 创建HSV图像用于可视化
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 转换为极坐标表示
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 设置HSV图像的色调和亮度
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # 转换为BGR显示
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('Original', frame2)
        cv2.imshow('Optical Flow', bgr)

        prvs = next_frame

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
farneback_optical_flow('videos/traffic.mp4')
```

### 实时光流检测

```python
def real_time_optical_flow():
    """实时光流检测"""
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None

    # 参数设置
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # 读取第一帧
    ret, old_frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        return None

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # 创建轨迹
    mask = np.zeros_like(old_frame)
    colors = np.random.randint(0, 255, (100, 3))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 每30帧重新检测特征点
        if frame_count % 30 == 0:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)  # 清除轨迹

        # 计算光流
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            if p1 is not None:
                # 选择好的点
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # 绘制轨迹
                for i, (tr, to) in enumerate(zip(good_new, good_old)):
                    a, b = tr.ravel().astype(int)
                    c, d = to.ravel().astype(int)
                    mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)

                img = cv2.add(frame, mask)
                cv2.imshow('Real-time Optical Flow', img)

                # 更新
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
real_time_optical_flow()
```

## 算法比较

### Lucas-Kanade vs Farneback

| 特性     | Lucas-Kanade | Farneback    |
| -------- | ------------ | ------------ |
| 类型     | 稀疏光流     | 稠密光流     |
| 计算量   | 低           | 高           |
| 精度     | 高（局部）   | 中等（全局） |
| 应用场景 | 目标跟踪     | 运动分析     |
| 实时性   | 好           | 一般         |

### 优缺点分析

#### Lucas-Kanade 算法

**优点**:

- 计算效率高，适合实时应用
- 对特征点跟踪精度高
- 实现简单，参数较少
- 对局部运动分析效果好

**缺点**:

- 只能跟踪特征点，无法获得全局运动信息
- 对大运动敏感，需要金字塔处理
- 特征点可能丢失
- 依赖于良好的特征点检测

#### Farneback 算法

**优点**:

- 提供稠密光流场信息
- 能够分析全局运动模式
- 对纹理丰富区域效果好
- 不依赖特征点检测

**缺点**:

- 计算量大，实时性较差
- 对参数敏感，需要仔细调优
- 在纹理稀少区域效果不佳
- 内存消耗大

## 应用场景

### 适用场景

**Lucas-Kanade 光流适用于**:

- 实时目标跟踪
- 视频监控系统
- 手势识别
- 人脸跟踪
- 移动端应用

**Farneback 光流适用于**:

- 运动分析
- 视频稳定
- 背景建模
- 运动检测
- 科学研究

### 目标跟踪

```python
def object_tracking_with_optical_flow(video_path):
    """基于光流的目标跟踪"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # 读取第一帧并选择目标
    ret, frame = cap.read()
    if not ret:
        return None

    # 手动选择跟踪区域（也可以用检测器自动选择）
    bbox = cv2.selectROI("Select Object", frame, False)
    cv2.destroyWindow("Select Object")

    # 在选择区域内检测特征点
    x, y, w, h = [int(i) for i in bbox]
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 特征点检测
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=None, **feature_params)

    if p0 is not None:
        # 调整坐标到全图
        p0[:, :, 0] += x
        p0[:, :, 1] += y

    # 光流参数
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            # 计算光流
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            if p1 is not None:
                # 选择好的点
                good_new = p1[st == 1]

                if len(good_new) > 0:
                    # 计算目标位置
                    center_x = int(np.mean(good_new[:, 0]))
                    center_y = int(np.mean(good_new[:, 1]))

                    # 绘制跟踪结果
                    cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), 2)
                    cv2.rectangle(frame,
                                (center_x - w//2, center_y - h//2),
                                (center_x + w//2, center_y + h//2),
                                (0, 255, 0), 2)

                    # 绘制特征点
                    for pt in good_new:
                        x_pt, y_pt = pt.ravel().astype(int)
                        cv2.circle(frame, (x_pt, y_pt), 3, (255, 0, 0), -1)

                    p0 = good_new.reshape(-1, 1, 2)

        cv2.imshow('Object Tracking', frame)
        old_gray = frame_gray.copy()

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 使用示例
object_tracking_with_optical_flow('videos/object.mp4')
```

### 运动分析

```python
def motion_analysis(video_path):
    """运动分析应用"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    ret, frame1 = cap.read()
    if not ret:
        return None

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # 运动统计
    motion_history = []

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 计算运动强度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # 运动统计
        motion_intensity = np.mean(mag)
        motion_history.append(motion_intensity)

        # 可视化运动区域
        motion_mask = mag > np.mean(mag) + 2 * np.std(mag)
        frame2[motion_mask] = [0, 0, 255]  # 标记高运动区域为红色

        # 显示运动强度
        cv2.putText(frame2, f'Motion: {motion_intensity:.2f}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Motion Analysis', frame2)
        prvs = next_frame

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 绘制运动趋势
    plt.figure(figsize=(10, 6))
    plt.plot(motion_history)
    plt.title('Motion Intensity Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Motion Intensity')
    plt.grid(True)
    plt.show()

# 使用示例
motion_analysis('videos/sports.mp4')
```
