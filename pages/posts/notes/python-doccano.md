---
title: doccano 文本安装工具安装踩坑
date: 2025-07-14
type: notes
---

_Link[doccano Github 地址](https://github.com/doccano/doccano)_

环境信息

- Python: 3.9.13(Conda 创建的虚拟环境)

执行命令`doccano init`报错，由于`marshmallow`版本过低，导致报错，但当我执行 `pip install --upgrade marshmallow`升级安装 marshmallow 又报错，最后通过重新配置虚拟环境`conda create -n doccano_env python=3.8`解决， 项目作者要求 Python 3.8 以上版本， 但实测 3.9 以上存在会有兼容性问题。

```bash
(doccano_env) C:\Users\xxx>doccano init
Traceback (most recent call last):
  File "C:\Users\xxx\.conda\envs\doccano_env\lib\runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Users\xxx\.conda\envs\doccano_env\lib\runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "C:\Users\xxx\.conda\envs\doccano_env\Scripts\doccano.exe\__main__.py", line 4, in <module>
  File "C:\Users\xxx\.conda\envs\doccano_env\lib\site-packages\backend\cli.py", line 10, in <module>
    from environs import Env
  File "C:\Users\xxx\.conda\envs\doccano_env\lib\site-packages\environs\__init__.py", line 58, in <module>
    _SUPPORTS_LOAD_DEFAULT = ma.__version_info__ >= (3, 13)
AttributeError: module 'marshmallow' has no attribute '__version_info__'
```
