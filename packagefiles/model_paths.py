"""提供统一的模型路径解析工具，确保不同模块可在任意工作目录下找到 ONNX 资源。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

# 项目根目录：``packagefiles`` 与 ``model`` 位于同一级目录
_REPO_ROOT = Path(__file__).resolve().parents[1]

# 模型主目录以及常用子目录
MODEL_ROOT = _REPO_ROOT / "model"
OCR_MODEL_ROOT = MODEL_ROOT / "ocr_model"
YOLO_MODEL_ROOT = MODEL_ROOT / "yolo_model"


def _join_path(base: Path, parts: Iterable[str]) -> Path:
    """在给定基础目录上拼接路径片段。"""
    return base.joinpath(*parts)


def model_path(*parts: str) -> str:
    """返回 ``model`` 目录下资源的绝对路径字符串。"""
    return str(_join_path(MODEL_ROOT, parts))


def ocr_model_path(*parts: str) -> str:
    """返回 ``model/ocr_model`` 子目录下资源的绝对路径字符串。"""
    return str(_join_path(OCR_MODEL_ROOT, parts))


def yolo_model_path(*parts: str) -> str:
    """返回 ``model/yolo_model`` 子目录下资源的绝对路径字符串。"""
    return str(_join_path(YOLO_MODEL_ROOT, parts))

