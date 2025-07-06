---
title: 浏览器自动化：基于 WebUI 与 DeepSeek V3 的深度集成
date: 2025-03-10
draft: true
lang: zh
---

GitHub 上有一个值得关注的开源项目 browser-use，目前已收获 8.3K Star。该项目基于 Playwright 构建，通过 LLM 实现智能浏览器自动化，支持复杂的网页交互和数据采集任务。

最近，社区开发者基于 browser-use 开发了一个开源的 WebUI 界面（browser-use-webui），显著降低了使用门槛，并优化了以下关键特性：

## **browser-use WebUI 主要功能**

1. **交互式操作界面**

   - 可视化任务配置
   - 实时执行状态监控
   - 支持断点调试和重试机制

2. **多模型适配**  
   目前已完成对接：

   - DeepSeek V3（推荐，支持中文效果最好）
   - Gemini Pro
   - GPT-3.5/4
   - Claude 2

3. **本地浏览器集成**

   - 支持 Chrome/Edge/Firefox
   - 保留已登录状态和 Cookie
   - 内置录屏与重放功能
   - 支持多浏览器实例

4. **智能交互增强**
   - 优化的 prompt 模板
   - 自动错误重试
   - 上下文感知的操作链
   - 支持自定义动作扩展

---

## **安装指南**

项目已在 GitHub 开源，感兴趣的朋友可以尝试体验。项目基于 **Python** 开发，要求版本 **3.11 及以上**。

- 推荐使用 [pyenv](https://github.com/pyenv/pyenv) 或 [uv](https://docs.astral.sh/uv/) 管理 Python 版本。
- 由于本机已安装[Anaconda](https://www.anaconda.com/download/success), 我将使用 Conda 创建虚拟环境。
- 根据个人习惯选择合适的工具，无需纠结具体实现方式。

### 第一步: 拉取项目到本地

```bash
git clone https://github.com/browser-use/web-ui.git
cd web-ui
```

### 第二步: 创建虚拟环境

```bash
conda create -n browser-use # 创建一个虚拟环境
source activate browser-use # 激活虚拟环境
```

### 第三步: 安装依赖包

```bash
pip3 install browser-use # 安装browser-use包

playwright install # 安装playwright

pip3 install -r requirements.txt # 安装依赖包
```

### 第四步: 配置环境信息

```bash
cp .env.example .env

# 配置示例（根据实际路径修改）：
CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME_USER_DATA="/Users/username/Library/Application Support/Google/Chrome"
PLAYWRIGHT_BROWSERS_PATH="/Users/username/.cache/ms-playwright"
DEBUG_MODE=false
```

### 第五步: 运行项目

```bash
python3 webui.py --ip 127.0.0.1 --port 7788
```

提示如下信息时, 项目运行成功

```bash
(browser-use) ➜  web-ui git:(main) python3 webui.py --ip 127.0.0.1 --port 7788

INFO     [browser_use] BrowserUse logging setup complete with level info
INFO     [root] Anonymized telemetry enabled. See https://docs.browser-use.com/development/telemetry for more information.
* Running on local URL:  http://127.0.0.1:7788

To create a public link, set `share=True` in `launch()`.
```

![运行成功](/images/blog/browser-use/success-info.png)

## **使用指南**

浏览器输入: http://127.0.0.1:7788/ 看到界面后则代表成功
![OP1](/images/blog/browser-use/op-1.png)

进入[DeepSeek 官网](https://www.deepseek.com/) 点击 API 开发平台, 适当充值后,生成 API 密钥

![OP2](/images/blog/browser-use/op-2.png)
![OP3](/images/blog/browser-use/op-3.png)
在界面上填入 API 密钥即可进行下一步
![OP4](/images/blog/browser-use/op-4.png)
点击 Run Agent 输入任务的描述信息，随后点击 `Run Agent` 按钮，你可以看到命令行窗口输入了如下内容
![OP5](/images/blog/browser-use/op-5.png)
![OP6](/images/blog/browser-use/op-6.png)
当出现如下提示时，任务运行成功
![OP7](/images/blog/browser-use/op-7.png)

点击 Results 你可以看到运行结果，效果还是非常不错的，ta 会帮你记录使用过程的录像，你可以随时查看

![OP8](/images/blog/browser-use/op-8.png)

## **常见问题**

1. 如遇到浏览器启动失败，请检查：

   - Chrome 路径是否正确配置
   - 用户数据目录权限是否正常
   - 是否有其他 Chrome 实例占用

2. DeepSeek API 相关：

   - 支持 V3 及以上版本
   - 按量计费，建议先充值 10 元测试

3. 性能优化：
   - 支持多线程并行执行
   - 建议限制单次任务时长
   - 可配置自动重试次数
