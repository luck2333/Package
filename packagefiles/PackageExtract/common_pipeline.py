"""封装提取流程中共用的辅助函数集合。"""

from __future__ import annotations

import os
import shutil
from typing import Iterable, Tuple

import numpy as np

from packagefiles.PackageExtract.DETR_BGA import DETR_BGA
from packagefiles.PackageExtract.function_tool import empty_folder, set_Image_size
from packagefiles.PackageExtract.onnx_use import Run_onnx_det
from packagefiles.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2 import (
    begain_output_pairs_data_location,
)


# 默认需要处理的视图顺序，保持与原流程一致。
DEFAULT_VIEWS: Tuple[str, ...] = ("top", "bottom", "side", "detailed")


def prepare_workspace(
    data_dir: str,
    data_copy_dir: str,
    data_bottom_crop_dir: str,
    onnx_output_dir: str,
    opencv_output_dir: str,
    image_views: Iterable[str] = DEFAULT_VIEWS,
) -> None:
    """初始化提取流程所需的临时目录，并统一输入图片尺寸。

    该函数完整复刻了旧版 ``front_loading_work`` 的处理步骤：
    1. 清空上一次推理的中间产物目录；
    2. 遍历多个视图，确保图片尺寸符合推理要求；
    3. 将视图图像备份到 ``data_copy``，再还原到 ``data``，保证后续步骤在干净的副本上运行。
    """

    # 重置存放检测结果的临时目录。
    empty_folder(onnx_output_dir)
    os.makedirs(onnx_output_dir, exist_ok=True)

    empty_folder(data_bottom_crop_dir)
    os.makedirs(data_bottom_crop_dir, exist_ok=True)

    # 逐个视图调整图片尺寸，缺失图片时保留提示信息。
    for view_name in image_views:
        filein = os.path.join(data_dir, f"{view_name}.jpg")
        fileout = filein
        try:
            set_Image_size(filein, fileout)
        except Exception:
            print("文件", filein, "不存在")

    # 备份视图图片，保留当前状态。
    empty_folder(data_copy_dir)
    os.makedirs(data_copy_dir, exist_ok=True)
    if os.path.isdir(data_dir):
        for file_name in os.listdir(data_dir):
            shutil.copy(os.path.join(data_dir, file_name), os.path.join(data_copy_dir, file_name))

    # 清空 OpenCV 的输出目录。
    empty_folder(opencv_output_dir)
    os.makedirs(opencv_output_dir, exist_ok=True)

    # 使用备份重新构建 ``data`` 目录，确保后续步骤在一致的数据上运行。
    empty_folder(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    if os.path.isdir(data_copy_dir):
        for file_name in os.listdir(data_copy_dir):
            shutil.copy(os.path.join(data_copy_dir, file_name), os.path.join(data_dir, file_name))


def dbnet_get_text_box(img_path: str) -> np.ndarray:
    """运行 DBNet，获取指定图片的文本框坐标。"""

    location_cool = Run_onnx_det(img_path)
    dbnet_data = np.empty((len(location_cool), 4))  # [x1,x2,x3,x4]
    for i in range(len(location_cool)):
        dbnet_data[i][0] = min(location_cool[i][2], location_cool[i][0])
        dbnet_data[i][1] = min(location_cool[i][3], location_cool[i][1])
        dbnet_data[i][2] = max(location_cool[i][2], location_cool[i][0])
        dbnet_data[i][3] = max(location_cool[i][3], location_cool[i][1])

    dbnet_data = np.around(dbnet_data, decimals=2)
    return dbnet_data


def yolo_classify(img_path: str, package_classes: str):
    """调用 YOLO 系列检测器，返回图像元素的坐标信息。"""

    if package_classes == "BGA":
        # BGA 封装需要额外合并 DETR 结果，强化 PIN 及边框的检测质量。
        (
            yolox_pairs,
            yolox_num,
            yolox_serial_num,
            pin,
            other,
            pad,
            border,
            angle_pairs,
            BGA_serial_num,
            BGA_serial_letter,
        ) = begain_output_pairs_data_location(img_path, package_classes)
        (
            _,
            _,
            _,
            pin,
            _,
            _,
            border,
            _,
            BGA_serial_num,
            BGA_serial_letter,
        ) = DETR_BGA(img_path, package_classes)
        print("yolox_pairs", yolox_pairs)
        print("yolox_num", yolox_num)
        print("yolox_serial_num", yolox_serial_num)
        print("pin", pin)
        print("other", other)
        print("pad", pad)
        print("border", border)
        print("angle_pairs", angle_pairs)
        print("BGA_serial_num", BGA_serial_num)
        print("BGA_serial_letter", BGA_serial_letter)
    else:
        (
            yolox_pairs,
            yolox_num,
            yolox_serial_num,
            pin,
            other,
            pad,
            border,
            angle_pairs,
            BGA_serial_num,
            BGA_serial_letter,
        ) = begain_output_pairs_data_location(img_path, package_classes)

        yolox_pairs = np.around(yolox_pairs, decimals=2)
        yolox_num = np.around(yolox_num, decimals=2)
        angle_pairs = np.around(angle_pairs, decimals=2)

    return (
        yolox_pairs,
        yolox_num,
        yolox_serial_num,
        pin,
        other,
        pad,
        border,
        angle_pairs,
        BGA_serial_num,
        BGA_serial_letter,
    )


def get_data_location_by_yolo_dbnet(
    package_path: str, package_classes: str, view_names: Iterable[str] = DEFAULT_VIEWS
):
    """结合 YOLO 与 DBNet 的结果，汇总指定视图的检测数据。"""

    L3 = []
    empty_data = np.empty((0, 4))

    # 使用字典暂存每个视图的检测结果，便于后续统一展开成 L3 列表。
    view_results = {}
    for view in view_names:
        img_path = os.path.join(package_path, f"{view}.jpg")
        if os.path.exists(img_path):
            dbnet_data = dbnet_get_text_box(img_path)
            (
                yolox_pairs,
                yolox_num,
                yolox_serial_num,
                pin,
                other,
                pad,
                border,
                angle_pairs,
                BGA_serial_num,
                BGA_serial_letter,
            ) = yolo_classify(img_path, package_classes)
        else:
            dbnet_data = empty_data
            yolox_pairs = empty_data
            yolox_num = empty_data
            yolox_serial_num = empty_data
            pin = empty_data
            other = empty_data
            pad = empty_data
            border = empty_data
            angle_pairs = empty_data
            BGA_serial_num = empty_data
            BGA_serial_letter = empty_data
        view_results[view] = {
            "dbnet_data": dbnet_data,
            "yolox_pairs": yolox_pairs,
            "yolox_num": yolox_num,
            "yolox_serial_num": yolox_serial_num,
            "pin": pin,
            "other": other,
            "pad": pad,
            "border": border,
            "angle_pairs": angle_pairs,
            "BGA_serial_num": BGA_serial_num,
            "BGA_serial_letter": BGA_serial_letter,
        }

    for view in view_names:
        results = view_results[view]
        for key in ("dbnet_data", "yolox_pairs", "yolox_num", "yolox_serial_num", "pin", "other", "pad", "border", "angle_pairs"):
            L3.append({"list_name": f"{view}_{key}", "list": results[key]})
        if view == "bottom":
            L3.append({"list_name": "bottom_BGA_serial_letter", "list": results["BGA_serial_letter"]})
            L3.append({"list_name": "bottom_BGA_serial_num", "list": results["BGA_serial_num"]})

    # 返回与旧流程一致的 L3 数据结构，方便直接替换原有实现。
    return L3
