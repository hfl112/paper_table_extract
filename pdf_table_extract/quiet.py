"""把第三方库的日志噪声关掉。

docling / paddleocr / paddlex / RapidOCR 会往 stderr 刷大量 INFO 行和进度条
（"Model files already exist"、"Loading weights"、"Using engine_name" 之类）。
早先的做法是让调用方用 grep 过滤，那是把库的毛病转嫁给使用者 ——
每条命令都得挂一串管道，也没法直接复制粘贴。这里在进程内一次性关掉。

只压 INFO/DEBUG，**WARNING 及以上照常输出** —— 真出问题时不能瞎。
"""

from __future__ import annotations

import logging
import os
import warnings

_NOISY = (
    "paddle",
    "paddleocr",
    "paddlex",
    "docling",
    "docling_core",
    "docling_ibm_models",
    "RapidOCR",
    "rapidocr",
    "pypdfium2",
    "torch",
    "PIL",
    "matplotlib",
)

_env_done = False


def silence() -> None:
    """设环境变量 + 压 logger。**必须在任何引擎 import 之前**调用一次。"""
    global _env_done
    if not _env_done:
        _env_done = True
        # 进度条（docling 的 "Loading weights"、huggingface 的 "Fetching N files"）
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        # paddlex 每次启动都去探测模型源，几秒且刷屏
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("GLOG_minloglevel", "2")
        warnings.filterwarnings("ignore")
    hush_loggers()


def hush_loggers() -> None:
    """把噪声 logger 压到 WARNING。**每次 import 完引擎库后都要再调一次。**

    原因（踩过的坑）：`paddlex` 在 import 时会调自己的 `setup_logging()`，
    把 `paddlex` logger 的级别**显式设回 INFO**，盖掉提前设的 WARNING；
    RapidOCR 同样在 import 时装自己的 handler 和级别。
    所以 import 前设一次是不够的，import 后必须补一刀 —— 这个函数是幂等的。
    """
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
        # 子 logger 已存在时要逐个压，否则它们自己的 level 会盖过父级
        prefix = name + "."
        for existing in list(logging.Logger.manager.loggerDict):
            if existing.startswith(prefix):
                logging.getLogger(existing).setLevel(logging.WARNING)
