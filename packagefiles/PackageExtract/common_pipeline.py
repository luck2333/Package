"""封装提取流程中共用的辅助函数集合。"""

from __future__ import annotations

import os
import shutil
from typing import Iterable, Tuple

import numpy as np

from packagefiles.PackageExtract.DETR_BGA import DETR_BGA
from packagefiles.PackageExtract import get_pairs_data_present5_test as _pairs_module
from packagefiles.PackageExtract.function_tool import (
    empty_folder,
    find_list,
    recite_data,
    set_Image_size,
)
from packagefiles.PackageExtract.onnx_use import Run_onnx_det
from packagefiles.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2 import (
    begain_output_pairs_data_location,
)

# 目录定位：保持与用户提供的脚本一致，便于在不同 IDE 中运行。
file_dir = str(os.path.abspath(__file__))
current_dir = str(os.path.dirname(file_dir))
parent_dir = str(os.path.dirname(current_dir))
root_dir = str(os.path.dirname(parent_dir))

current_dir = current_dir.replace("\\", "//")
parent_dir = parent_dir.replace("\\", "//")
root_dir = root_dir.replace("\\", "//")

# 全局路径，兼容原脚本中的写法。
DATA = root_dir + "//" + "Result//Package_extract//data"
DATA_BOTTOM_CROP = root_dir + "//" + "Result//Package_extract//data_bottom_crop"
DATA_COPY = root_dir + "//" + "Result//Package_extract//data_copy"
ONNX_OUTPUT = root_dir + "//" + "Result//Package_extract//onnx_output"
OPENCV_OUTPUT = root_dir + "//" + "Result//Package_extract//opencv_output"
OPENCV_OUTPUT_LINE = root_dir + "//" + "Result//Package_extract//opencv_output_yinXian"
YOLO_DATA = root_dir + "//" + "Result//Package_extract//yolox_data"


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
    """初始化提取流程所需的临时目录，并统一输入图片尺寸。"""

    empty_folder(onnx_output_dir)
    os.makedirs(onnx_output_dir, exist_ok=True)

    empty_folder(data_bottom_crop_dir)
    os.makedirs(data_bottom_crop_dir, exist_ok=True)

    for view_name in image_views:
        filein = os.path.join(data_dir, f"{view_name}.jpg")
        fileout = filein
        try:
            set_Image_size(filein, fileout)
        except Exception:
            print("文件", filein, "不存在")

    empty_folder(data_copy_dir)
    os.makedirs(data_copy_dir, exist_ok=True)
    if os.path.isdir(data_dir):
        for file_name in os.listdir(data_dir):
            shutil.copy(os.path.join(data_dir, file_name), os.path.join(data_copy_dir, file_name))

    empty_folder(opencv_output_dir)
    os.makedirs(opencv_output_dir, exist_ok=True)

    empty_folder(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    if os.path.isdir(data_copy_dir):
        for file_name in os.listdir(data_copy_dir):
            shutil.copy(os.path.join(data_copy_dir, file_name), os.path.join(data_dir, file_name))


def dbnet_get_text_box(img_path: str) -> np.ndarray:
    """运行 DBNet，获取指定图片的文本框坐标。"""

    location_cool = Run_onnx_det(img_path)
    dbnet_data = np.empty((len(location_cool), 4))
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
            print("未找到视图,返回空值")
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
        for key in (
            "dbnet_data",
            "yolox_pairs",
            "yolox_num",
            "yolox_serial_num",
            "pin",
            "other",
            "pad",
            "border",
            "angle_pairs",
        ):
            L3.append({"list_name": f"{view}_{key}", "list": results[key]})
        if view == "bottom":
            L3.append({"list_name": "bottom_BGA_serial_letter", "list": results["BGA_serial_letter"]})
            L3.append({"list_name": "bottom_BGA_serial_num", "list": results["BGA_serial_num"]})

    return L3


def remove_other_annotations(L3):
    """F4.6：剔除 YOLO/DBNet 输出中的 OTHER 类型框。"""

    for view in ("top", "bottom", "side", "detailed"):
        yolox_key = f"{view}_yolox_num"
        dbnet_key = f"{view}_dbnet_data"
        other_key = f"{view}_other"

        yolox_num = find_list(L3, yolox_key)
        dbnet_data = find_list(L3, dbnet_key)
        other_data = find_list(L3, other_key)

        filtered_yolox = _pairs_module.delete_other(other_data, yolox_num)
        filtered_dbnet = _pairs_module.delete_other(other_data, dbnet_data)

        recite_data(L3, yolox_key, filtered_yolox)
        recite_data(L3, dbnet_key, filtered_dbnet)

    return L3


def enrich_pairs_with_lines(L3, image_root: str, test_mode: int):
    """F4.6：为尺寸线补齐对应的标尺界限。"""

    empty_data = np.empty((0, 13))
    for view in ("top", "bottom", "side", "detailed"):
        yolox_pairs = find_list(L3, f"{view}_yolox_pairs")
        img_path = os.path.join(image_root, f"{view}.jpg")

        if os.path.exists(img_path):
            pairs_length = _pairs_module.find_pairs_length(img_path, yolox_pairs, test_mode)
        else:
            pairs_length = empty_data

        recite_data(L3, f"{view}_yolox_pairs_length", pairs_length)

    return L3


def preprocess_pairs_and_text(L3, key: int):
    """F4.7：整理尺寸线与文本，生成初始配对候选。"""

    top_yolox_pairs = find_list(L3, "top_yolox_pairs")
    bottom_yolox_pairs = find_list(L3, "bottom_yolox_pairs")
    side_yolox_pairs = find_list(L3, "side_yolox_pairs")
    detailed_yolox_pairs = find_list(L3, "detailed_yolox_pairs")
    top_dbnet_data = find_list(L3, "top_dbnet_data")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    side_dbnet_data = find_list(L3, "side_dbnet_data")
    detailed_dbnet_data = find_list(L3, "detailed_dbnet_data")

    (
        top_yolox_pairs,
        bottom_yolox_pairs,
        side_yolox_pairs,
        detailed_yolox_pairs,
        top_yolox_pairs_copy,
        bottom_yolox_pairs_copy,
        side_yolox_pairs_copy,
        detailed_yolox_pairs_copy,
        top_dbnet_data_all,
        bottom_dbnet_data_all,
    ) = _pairs_module.get_better_data_1(
        top_yolox_pairs,
        bottom_yolox_pairs,
        side_yolox_pairs,
        detailed_yolox_pairs,
        key,
        top_dbnet_data,
        bottom_dbnet_data,
        side_dbnet_data,
        detailed_dbnet_data,
    )

    recite_data(L3, "top_yolox_pairs", top_yolox_pairs)
    recite_data(L3, "bottom_yolox_pairs", bottom_yolox_pairs)
    recite_data(L3, "side_yolox_pairs", side_yolox_pairs)
    recite_data(L3, "detailed_yolox_pairs", detailed_yolox_pairs)
    recite_data(L3, "top_dbnet_data", top_dbnet_data)
    recite_data(L3, "bottom_dbnet_data", bottom_dbnet_data)
    recite_data(L3, "side_dbnet_data", side_dbnet_data)
    recite_data(L3, "detailed_dbnet_data", detailed_dbnet_data)
    recite_data(L3, "top_yolox_pairs_copy", top_yolox_pairs_copy)
    recite_data(L3, "bottom_yolox_pairs_copy", bottom_yolox_pairs_copy)
    recite_data(L3, "side_yolox_pairs_copy", side_yolox_pairs_copy)
    recite_data(L3, "detailed_yolox_pairs_copy", detailed_yolox_pairs_copy)
    recite_data(L3, "top_dbnet_data_all", top_dbnet_data_all)
    recite_data(L3, "bottom_dbnet_data_all", bottom_dbnet_data_all)

    return L3


def run_svtr_ocr(L3):
    """F4.7：执行 SVTR OCR 推理，将文本候选加入 L3。"""

    top_dbnet_data_all = find_list(L3, "top_dbnet_data_all")
    bottom_dbnet_data_all = find_list(L3, "bottom_dbnet_data_all")
    side_dbnet_data = find_list(L3, "side_dbnet_data")
    detailed_dbnet_data = find_list(L3, "detailed_dbnet_data")

    _, _, top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data = _pairs_module.SVTR(
        top_dbnet_data_all,
        bottom_dbnet_data_all,
        side_dbnet_data,
        detailed_dbnet_data,
    )

    recite_data(L3, "top_ocr_data", top_ocr_data)
    recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    recite_data(L3, "side_ocr_data", side_ocr_data)
    recite_data(L3, "detailed_ocr_data", detailed_ocr_data)

    return L3


def normalize_ocr_candidates(L3, key: int):
    """F4.7：OCR 文本后处理，规整最大/中值/最小候选。"""

    top_dbnet_data = find_list(L3, "top_dbnet_data")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    side_dbnet_data = find_list(L3, "side_dbnet_data")
    detailed_dbnet_data = find_list(L3, "detailed_dbnet_data")
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    detailed_ocr_data = find_list(L3, "detailed_ocr_data")
    top_yolox_num = find_list(L3, "top_yolox_num")
    bottom_yolox_num = find_list(L3, "bottom_yolox_num")
    side_yolox_num = find_list(L3, "side_yolox_num")
    detailed_yolox_num = find_list(L3, "detailed_yolox_num")

    (
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
    ) = _pairs_module.data_wrangling(
        key,
        top_dbnet_data,
        bottom_dbnet_data,
        side_dbnet_data,
        detailed_dbnet_data,
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        top_yolox_num,
        bottom_yolox_num,
        side_yolox_num,
        detailed_yolox_num,
    )

    recite_data(L3, "top_ocr_data", top_ocr_data)
    recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    recite_data(L3, "side_ocr_data", side_ocr_data)
    recite_data(L3, "detailed_ocr_data", detailed_ocr_data)

    return L3


def extract_pin_serials(L3, package_classes: str):
    """F4.8：提取序号/PIN 相关信息，兼容 BGA/QFP 等封装。"""

    top_yolox_serial_num = find_list(L3, "top_yolox_serial_num")
    bottom_yolox_serial_num = find_list(L3, "bottom_yolox_serial_num")
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")

    if package_classes in {"QFP", "QFN", "SOP", "SON"}:
        (
            top_serial_numbers_data,
            bottom_serial_numbers_data,
            top_ocr_data,
            bottom_ocr_data,
        ) = _pairs_module.find_PIN(
            top_yolox_serial_num,
            bottom_yolox_serial_num,
            top_ocr_data,
            bottom_ocr_data,
        )

        recite_data(L3, "top_serial_numbers_data", top_serial_numbers_data)
        recite_data(L3, "bottom_serial_numbers_data", bottom_serial_numbers_data)
        recite_data(L3, "top_ocr_data", top_ocr_data)
        recite_data(L3, "bottom_ocr_data", bottom_ocr_data)

    if package_classes == "BGA":
        bottom_serial_numbers = find_list(L3, "bottom_BGA_serial_num")
        bottom_serial_letters = find_list(L3, "bottom_BGA_serial_letter")
        bottom_border = find_list(L3, "bottom_border")
        bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")

        bottom_dbnet_data_serial = np.array(bottom_dbnet_data)

        try:
            pin_map, color_map = _pairs_module.time_save_find_pinmap(bottom_border)
        except Exception as exc:
            print("time_save_find_pinmap 执行失败:", exc)
            pin_map = [[1 for _ in range(10)] for _ in range(10)]
            color_map = [[2 for _ in range(10)] for _ in range(10)]

        (
            serial_numbers_coords,
            serial_letters_coords,
            filtered_bottom_dbnet,
        ) = _pairs_module.find_serial_number_letter(
            bottom_serial_numbers,
            bottom_serial_letters,
            bottom_dbnet_data_serial,
        )

        bottom_img_path = os.path.join(DATA, "bottom.jpg")
        bottom_ocr_strings = _pairs_module.ocr_data(bottom_img_path, bottom_dbnet_data_serial)

        (
            serial_numbers_data,
            serial_letters_data,
            filtered_bottom_ocr,
        ) = _pairs_module.filter_bottom_ocr_data(
            bottom_ocr_strings,
            bottom_dbnet_data_serial,
            serial_numbers_coords,
            serial_letters_coords,
            filtered_bottom_dbnet,
        )

        (
            pin_num_x_serial,
            pin_num_y_serial,
            pin_1_location,
        ) = _pairs_module.find_pin_num_pin_1(
            serial_numbers_data,
            serial_letters_data,
            serial_numbers_coords,
            serial_letters_coords,
        )

        try:
            from packagefiles.PackageExtract.BGA_extract_old import Is_Loss_Pin  # type: ignore
        except ImportError:
            loss_pin = []
            loss_color = []
        else:
            try:
                loss_pin, loss_color = Is_Loss_Pin(pin_map, pin_1_location, color_map)
            except Exception as exc:
                print("Is_Loss_Pin 执行失败:", exc)
                loss_pin = []
                loss_color = []

        loss_pin_record = "None" if len(loss_pin) == 0 else loss_pin

        def _matrix_dimensions(matrix):
            if hasattr(matrix, "shape"):
                shape = getattr(matrix, "shape")
                if len(shape) >= 2:
                    return int(shape[0]), int(shape[1])
                if len(shape) == 1:
                    return int(shape[0]), 0
            rows = len(matrix)
            cols = len(matrix[0]) if rows else 0
            return rows, cols

        rows, cols = _matrix_dimensions(pin_map)

        pin_num_x_serial = int(pin_num_x_serial or 0)
        pin_num_y_serial = int(pin_num_y_serial or 0)

        if pin_num_x_serial == 0:
            pin_num_x_serial = int(cols)
        if pin_num_y_serial == 0:
            pin_num_y_serial = int(rows)

        recite_data(L3, "bottom_BGA_serial_num", serial_numbers_coords)
        recite_data(L3, "bottom_BGA_serial_letter", serial_letters_coords)
        recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
        recite_data(L3, "pin_num_x_serial", pin_num_x_serial)
        recite_data(L3, "pin_num_y_serial", pin_num_y_serial)
        recite_data(L3, "pin_1_location", pin_1_location)
        recite_data(L3, "loss_pin", loss_pin_record)
        recite_data(L3, "loss_color", loss_color)

    return L3


def match_pairs_with_text(L3, key: int):
    """F4.8：将尺寸线与 OCR 文本重新配对。"""

    top_yolox_pairs = find_list(L3, "top_yolox_pairs")
    bottom_yolox_pairs = find_list(L3, "bottom_yolox_pairs")
    side_yolox_pairs = find_list(L3, "side_yolox_pairs")
    detailed_yolox_pairs = find_list(L3, "detailed_yolox_pairs")
    side_angle_pairs = find_list(L3, "side_angle_pairs")
    detailed_angle_pairs = find_list(L3, "detailed_angle_pairs")
    top_border = find_list(L3, "top_border")
    bottom_border = find_list(L3, "bottom_border")
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    detailed_ocr_data = find_list(L3, "detailed_ocr_data")

    (
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
    ) = _pairs_module.MPD(
        key,
        top_yolox_pairs,
        bottom_yolox_pairs,
        side_yolox_pairs,
        detailed_yolox_pairs,
        side_angle_pairs,
        detailed_angle_pairs,
        top_border,
        bottom_border,
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
    )

    recite_data(L3, "top_ocr_data", top_ocr_data)
    recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    recite_data(L3, "side_ocr_data", side_ocr_data)
    recite_data(L3, "detailed_ocr_data", detailed_ocr_data)

    return L3


def finalize_pairs(L3):
    """F4.8：清理配对结果，输出最终可用的尺寸线集合。"""

    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    detailed_ocr_data = find_list(L3, "detailed_ocr_data")
    top_yolox_pairs_length = find_list(L3, "top_yolox_pairs_length")
    bottom_yolox_pairs_length = find_list(L3, "bottom_yolox_pairs_length")
    side_yolox_pairs_length = find_list(L3, "side_yolox_pairs_length")
    detailed_yolox_pairs_length = find_list(L3, "detailed_yolox_pairs_length")
    top_yolox_pairs_copy = find_list(L3, "top_yolox_pairs_copy")
    bottom_yolox_pairs_copy = find_list(L3, "bottom_yolox_pairs_copy")
    side_yolox_pairs_copy = find_list(L3, "side_yolox_pairs_copy")
    detailed_yolox_pairs_copy = find_list(L3, "detailed_yolox_pairs_copy")

    (
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        yolox_pairs_top,
        yolox_pairs_bottom,
        yolox_pairs_side,
        yolox_pairs_detailed,
    ) = _pairs_module.get_better_data_2(
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        top_yolox_pairs_length,
        bottom_yolox_pairs_length,
        side_yolox_pairs_length,
        detailed_yolox_pairs_length,
        top_yolox_pairs_copy,
        bottom_yolox_pairs_copy,
        side_yolox_pairs_copy,
        detailed_yolox_pairs_copy,
    )

    recite_data(L3, "top_ocr_data", top_ocr_data)
    recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    recite_data(L3, "side_ocr_data", side_ocr_data)
    recite_data(L3, "detailed_ocr_data", detailed_ocr_data)
    recite_data(L3, "yolox_pairs_top", yolox_pairs_top)
    recite_data(L3, "yolox_pairs_bottom", yolox_pairs_bottom)
    recite_data(L3, "yolox_pairs_side", yolox_pairs_side)
    recite_data(L3, "yolox_pairs_detailed", yolox_pairs_detailed)

    print("***/数据整理结果/***")
    print("top视图数据整理结果:\n", *top_ocr_data, sep="\n")
    print("bottom视图数据整理结果:\n", *bottom_ocr_data, sep="\n")
    print("side视图数据整理结果:\n", *side_ocr_data, sep="\n")
    print("detailed视图数据整理结果:\n", *detailed_ocr_data, sep="\n")

    return L3


def compute_qfp_parameters(L3):
    """F4.9：根据配对结果计算 QFP/BGA 参数列表。"""

    top_serial_numbers_data = find_list(L3, "top_serial_numbers_data")
    bottom_serial_numbers_data = find_list(L3, "bottom_serial_numbers_data")
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    detailed_ocr_data = find_list(L3, "detailed_ocr_data")
    yolox_pairs_top = find_list(L3, "yolox_pairs_top")
    yolox_pairs_bottom = find_list(L3, "yolox_pairs_bottom")
    top_yolox_pairs_length = find_list(L3, "top_yolox_pairs_length")
    bottom_yolox_pairs_length = find_list(L3, "bottom_yolox_pairs_length")
    top_border = find_list(L3, "top_border")
    bottom_border = find_list(L3, "bottom_border")

    nx, ny = _pairs_module.get_serial(top_serial_numbers_data, bottom_serial_numbers_data)
    body_x, body_y = _pairs_module.get_QFP_body(
        yolox_pairs_top,
        top_yolox_pairs_length,
        yolox_pairs_bottom,
        bottom_yolox_pairs_length,
        top_border,
        bottom_border,
        top_ocr_data,
        bottom_ocr_data,
    )
    _pairs_module.get_QFP_body(
        yolox_pairs_top,
        top_yolox_pairs_length,
        yolox_pairs_bottom,
        bottom_yolox_pairs_length,
        top_border,
        bottom_border,
        top_ocr_data,
        bottom_ocr_data,
    )

    QFP_parameter_list = _pairs_module.get_QFP_parameter_list(
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        body_x,
        body_y,
    )
    QFP_parameter_list = _pairs_module.resort_parameter_list_2(QFP_parameter_list)

    if len(QFP_parameter_list[4]["maybe_data"]) > 1:
        high = _pairs_module.get_QFP_high(QFP_parameter_list[4]["maybe_data"])
        if len(high) > 0:
            QFP_parameter_list[4]["maybe_data"] = high
            QFP_parameter_list[4]["maybe_data_num"] = len(high)

    if (
        len(QFP_parameter_list[5]["maybe_data"]) > 1
        or len(QFP_parameter_list[6]["maybe_data"]) > 1
    ):
        pitch_x, pitch_y = _pairs_module.get_QFP_pitch(
            QFP_parameter_list[5]["maybe_data"],
            body_x,
            body_y,
            nx,
            ny,
        )
        if len(pitch_x) > 0:
            QFP_parameter_list[5]["maybe_data"] = pitch_x
            QFP_parameter_list[5]["maybe_data_num"] = len(pitch_x)
        if len(pitch_y) > 0:
            QFP_parameter_list[6]["maybe_data"] = pitch_y
            QFP_parameter_list[6]["maybe_data_num"] = len(pitch_y)

    QFP_parameter_list = _pairs_module.resort_parameter_list_2(QFP_parameter_list)

    return QFP_parameter_list, nx, ny


if __name__ == "__main__":
    print("当前目录路径:", current_dir)
    print("上一级目录路径:", parent_dir)
    print("再上一级目录路径:", root_dir)
    print(os.path.exists("Result/Package_extract/data/top.jpg"))
    print(os.getcwd())

