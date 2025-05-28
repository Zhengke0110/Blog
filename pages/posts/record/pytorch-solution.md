---
title: 记录Pytorch等使用Bug
date: 2025-05-28
type: record
---

# Win11 下 d2l 安装报错解决

```bash
...
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

```

临时解决方案:

```bash
pip install d2l==1.0.0-alpha0
```
