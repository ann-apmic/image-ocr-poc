#!/usr/bin/env python3
"""
OCR測試專案主程式
"""

import os
import sys

from test_runner import main

# 添加專案路徑到Python路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    exit(main())
