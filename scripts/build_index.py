# -*- coding: utf-8 -*-
"""
scripts/build_index.py - 索引构建快捷入口

使用：
    export DASHSCOPE_API_KEY=xxx
    export AGICTO_API_KEY=xxx
    python scripts/build_index.py
"""
import os
import sys

# 把工程根加入 path，使 app.* 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.index.builder import build_and_save


if __name__ == "__main__":
    build_and_save(progress_callback=lambda stage, current, total, message: print(f"[{stage} {current}/{total}] {message}", flush=True))
