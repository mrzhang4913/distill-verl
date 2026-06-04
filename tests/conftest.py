# tests/conftest.py
# pytest 自动加载此文件，将项目根目录加入 Python 路径
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
