---
title: 安全微伴自动化脚本开发与实践
date: 2025-08-01
draft: true
lang: zh
# duration: 5min
---

![首页](/images/blog/anquan/home.png)

## 为什么要开发?

安全微伴是一个交互式的课堂互动平台，需要用户在指定课程中完成互动操作，在触发对应的互动事件后，系统会自动调用对应的 API，完成相应的进度，最终完成课程的学习。

由于课程包含大量互动事件，需要开发自动化脚本来处理这些交互，帮助用户快速完成课程学习，简化互动操作，提升用户体验。

## 开发思路

首先，点击某一个课程进入学习，观察其 HTML 结构，有如下发现：

1. 课程的学习页面采用的是调用第三方制作的学习内容，使用到了 iframe
2. 研究发现 iframe 中的 下一步按钮的点击事件会触发 iframe 中的 next 事件
3. 观察 iframe 中的 next 事件，发现其会调用 API，记录课程的进度
4. 比较页面的 section 结构发现，使用的是预加载 HTML 结构，通过点击下一步/下一页对 section 添加 page-activate 类属性，触发 section 的激活事件，从而完成页面的激活切换
5. 查看下一页/下一步按的的 HTML 结构，`body > section.page-WH.page-container > section.page-WH.page-item.page-0.page-active > div > a.base-an.btn-base.btn-next > img`，包含有`page-active`、`btn-next`等属性
6. 对比多个页面发现，其结构统一，可以考虑使用 JavaScript 定时器添加 click 事件，实现自动点击下一步/下一页功能。

![next1](/images/blog/anquan/next-1.png)
![next2](/images/blog/anquan/next-2.png)

## 代码实现

### 元素点击功能模块

在实践中发现，有些页面有非 next 的元素需要点击，来实现课程的交互，为了解决这个问题，我创建了一个全局变量 clickedElements，用于记录已经点击的元素，并创建了一个函数 shouldSkipElement，用于检查元素是否应该被跳过。
![special](/images/blog/anquan/special.png)

```javascript
// 全局点击记录，防止重复点击
const clickedElements = new Set();
const clickCountMap = new Map(); // 记录每个元素的点击次数

// 生成元素唯一标识
function getElementId(element) {
  const tag = element.tagName.toLowerCase();
  const className = element.className || "no-class";
  const text = element.textContent?.substring(0, 20) || "no-text";
  const parent = element.parentElement?.tagName.toLowerCase() || "no-parent";
  return `${tag}-${className}-${text}-${parent}`;
}

// 检查元素是否应该被跳过
function shouldSkipElement(element) {
  // 1. 检查类名中是否包含 prev
  if (element.className && element.className.includes("prev")) {
    console.log(`【跳过】包含 prev 的元素: ${element.className}`);
    return true;
  }

  // 2. 检查 ID 中是否包含 prev
  if (element.id && element.id.includes("prev")) {
    console.log(`【跳过】ID 包含 prev 的元素: ${element.id}`);
    return true;
  }

  // 3. 检查文本内容中是否包含"上一步"等关键词
  const text = element.textContent?.toLowerCase() || "";
  const skipKeywords = ["上一步", "previous", "prev"];

  if (skipKeywords.some((keyword) => text.includes(keyword))) {
    console.log(`【跳过】包含禁用关键词的元素: ${text.substring(0, 20)}`);
    return true;
  }

  return false;
}
```

![next3](/images/blog/anquan/next-3.png)

同时维护了一个优先级数组，确保点击顺序，避免无效点击。针对无法被规则检测到的按钮，可以在最高优先级中进行配置，确保能够被正确检测和点击。

```javascript
const clickableSelectors = [
  // 最高优先级: 课程列表
  "ul.img-texts-list li.img-texts-item:not(.passed)",

  // 最高优先级: 下一课相关按钮
  "#app > section > div > section.comment-container > section.rating-course > section > a:nth-child(2)",
  "a.comment-wklist-item",

  // 高优先级: page-WH 专用规则
  'section.page-WH.page-item[class*="page-active"] div.box.active-left a.base-an.btn-base[class*="btn-next"]',
  'section.page-WH.page-item[class*="page-active"] > div > div',

  // 中优先级: 初始和常规流程
  "section.page-start.page-active > div > span.pri-start-btn",
  "section.btn-next-prev > div > div.btn-next",

  // 低优先级: 通用和特定规则
  'section[class*="page-active"] a[class*="btn-base"]',
  "section.page-active > div > div",

  // 最低优先级: 通用图片规则
  'section.page-active img.element[class*="p"]',
];
```

### 答题模块

![problem](/images/blog/anquan/problem.png)

在实践中发现，平台的问题比较特殊，采用图片的形式进行展示，而不是常见的文字题型，方案一是通过 OCR 的方式识别图片，但经过验证后发现，可以绕过答题页面直接激活回答正确页面，触发对应的 next 事件，确保进度被记录。
![problem](/images/blog/anquan/problem-2.png)

```javascript
const questionPage = document.querySelector(
  'section[class*="page-aq"].page-active'
);
if (questionPage) {
  const match = questionPage.className.match(/page-aq(\d+)/);
  if (match && match[1]) {
    const questionNumber = match[1];
    const correctButtonSelector = `section.page-at${questionNumber} a.btn-at`;
    const correctButton = document.querySelector(correctButtonSelector);
    if (correctButton) {
      correctButton.click();
      return;
    }
  }
}
```

### 拖拽交互模块

课程中存在一些拖拽交互，如：滑块、图片拖拽等，针对这些交互，我创建了一个通用的拖拽处理函数，通过模拟鼠标按下、移动和抬起事件，实现拖拽交互。

```javascript
// 通用化处理所有可拖动滑块
const allDragSliders = document.querySelectorAll(
  'section.page-active div[class*="box"] div[class*="drag"]'
);
for (const dragSlider of allDragSliders) {
  // 检查滑块是否存在且尚未被拖动
  if (dragSlider && !dragSlider.style.left) {
    const trackElement = dragSlider.parentElement; // 获取滑块的容器（轨道）
    if (!trackElement) continue;

    const sliderRect = dragSlider.getBoundingClientRect();
    const trackRect = trackElement.getBoundingClientRect();

    // 从滑块中心点开始
    const startX = sliderRect.left + sliderRect.width / 2;
    const startY = sliderRect.top + sliderRect.height / 2;

    // 目标位置为轨道的末端，确保拖动到位
    const endX = trackRect.right - sliderRect.width / 2;

    // 1. 模拟鼠标按下
    dragSlider.dispatchEvent(
      new MouseEvent("mousedown", {
        bubbles: true,
        cancelable: true,
        clientX: startX,
        clientY: startY,
      })
    );

    // 2. 模拟鼠标移动
    dragSlider.dispatchEvent(
      new MouseEvent("mousemove", {
        bubbles: true,
        cancelable: true,
        clientX: endX,
        clientY: startY,
      })
    );

    // 3. 模拟鼠标松开
    dragSlider.dispatchEvent(
      new MouseEvent("mouseup", {
        bubbles: true,
        cancelable: true,
        clientX: endX,
        clientY: startY,
      })
    );
    return; // 每次只处理一个滑块
  }
}
```

### 页面静音模块

全面静音页面中的所有音频和视频元素，使用多种方式 确保能够完全静音。

```javascript
function muteCurrentTab() {
  try {
    // 方法1: 直接设置所有音频和视频元素静音
    document.querySelectorAll("video, audio").forEach((media) => {
      media.muted = true;
      media.volume = 0;
    });

    // 方法2: 使用Web Audio API静音整个页面
    if (window.AudioContext || window.webkitAudioContext) {
      const audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      const gainNode = audioContext.createGain();
      gainNode.gain.setValueAtTime(0, audioContext.currentTime);
      gainNode.connect(audioContext.destination);
      console.log("【Tampermonkey】使用Web Audio API静音页面");
    }

    // 方法3: 监听并拦截所有新创建的媒体元素
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) {
            if (node.tagName === "VIDEO" || node.tagName === "AUDIO") {
              node.muted = true;
              node.volume = 0;
            }
            // 也检查新添加元素的子元素
            const mediaElements =
              node.querySelectorAll && node.querySelectorAll("video, audio");
            if (mediaElements) {
              mediaElements.forEach((media) => {
                media.muted = true;
                media.volume = 0;
              });
            }
          }
        });
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  } catch (error) {
    console.log(`静音功能出错: ${error.message}`);
  }
}
```

### 弹窗屏蔽模块

课程中可能会出现一些 alert 的警告，通过屏蔽弹窗的代码，能够实现自动屏蔽页面中的 alert、confirm 和 prompt 弹窗，避免打断自动化流程。

```javascript
function blockAlerts() {
  try {
    // 保存原始的alert函数，以备日志使用
    const originalAlert = unsafeWindow.alert;
    const originalConfirm = unsafeWindow.confirm;
    const originalPrompt = unsafeWindow.prompt;

    // 拦截并屏蔽alert
    unsafeWindow.alert = function (message) {
      console.log(`已屏蔽alert警告: ${message}`);
      return undefined; // 不显示alert对话框
    };

    // 拦截并自动确认confirm对话框
    unsafeWindow.confirm = function (message) {
      console.log(`已自动确认confirm对话框: ${message}`);
      return true; // 自动点击"确定"
    };

    // 拦截并自动处理prompt对话框
    unsafeWindow.prompt = function (message, defaultText) {
      console.log(`已自动处理prompt对话框: ${message}`);
      return defaultText || ""; // 返回默认值或空字符串
    };

    console.log("alert/confirm/prompt 屏蔽功能已启动");
  } catch (error) {
    console.log(`屏蔽alert功能出错: ${error.message}`);
  }
}
```

### 视频播放拦截与加速模块

有一些课程的内容是视频+交互结合，为了加快学习速度又不影响进度的记录，可以适当修改视频的播放速度，这个模块的核心功能是拦截网页中的视频播放，并自动设置为静音和 4 倍速播放。

```javascript
const originalVideo = unsafeWindow.HTMLVideoElement.prototype.play;
unsafeWindow.HTMLVideoElement.prototype.play = function () {
  console.log("【Tampermonkey】拦截到 video.play()");
  this.muted = true;

  // 设置播放速度为最快，让视频快速自然播放完成
  this.playbackRate = 4;
  console.log("【Tampermonkey】设置视频4倍速播放，保持进度记录");

  // 监听视频结束事件，确保能够正确触发后续流程
  this.addEventListener("ended", function () {}, { once: true });

  return originalVideo.call(this);
};
```

### 主循环控制模块

整个脚本的核心控制逻辑，定时执行各种自动化操作，可以将数值调的更低，但完成过快可能会不记录学习进度。

```javascript
function autoProcess() {
  // 1. 自动播放视频并确保静音
  // 2. 处理拖拽滑块
  // 3. 处理进度条
  // 4. 智能答题
  // 5. 特殊按钮处理
  // 6. 常规元素点击
}

// 每500ms执行一次自动化处理
setInterval(autoProcess, 500);
```

## 如何使用?

1. 选用 Chrome/Edge 浏览器，并安装 Tampermonkey 插件。
2. 新建一个脚本，并复制粘贴脚本代码。
3. 保存并运行脚本，进入课程的学习页面，刷新后可看到脚本已加载的提示信息。

## 为什么选用 Tampermonkey?

由于浏览器的安全限制，经过 iframe 处理后的页面，脚本无法直接访问网页的 DOM 元素，因此需要使用 Tampermonkey 插件来执行脚本。Tampermonkey 插件允许脚本访问网页的 DOM 元素，并执行脚本。**没有 Tampermonkey 插件，脚本就无法执行。**

## 下载脚本

通过网盘分享的文件：autoV2.js
链接: https://pan.baidu.com/s/1IAAvrPwaPv1qEbFh0RTflA?pwd=6i5k 提取码: 6i5k
--来自百度网盘超级会员 v6 的分享
