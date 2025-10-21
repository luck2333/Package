import os
import queue
import threading

import numpy as np

from BGA_cal_pin import find_pin
from packagefiles.PackageExtract import get_pairs_data_present5_test as pairs_module
from packagefiles.PackageExtract.common_pipeline import (
    compute_qfp_parameters,
    enrich_pairs_with_lines,
    extract_pin_serials,
    finalize_pairs,
    get_data_location_by_yolo_dbnet,
    match_pairs_with_text,
    normalize_ocr_candidates,
    prepare_workspace,
    preprocess_pairs_and_text,
    remove_other_annotations,
    run_svtr_ocr,
)

# 全局路径
DATA = "Result/Package_extract/data"
DATA_BOTTOM_CROP = "Result/Package_extract/data_bottom_crop"
DATA_COPY = "Result/Package_extract/data_copy"
ONNX_OUTPUT = "Result/Package_extract/onnx_output"
OPENCV_OUTPUT = "Result/Package_extract/opencv_output"
OPENCV_OUTPUT_LINE = "Result/Package_extract/opencv_output_yinXian"


def extract_package(package_classes):
    """执行 BGA 封装参数提取的主流程。"""

    prepare_workspace(
        DATA,
        DATA_COPY,
        DATA_BOTTOM_CROP,
        ONNX_OUTPUT,
        OPENCV_OUTPUT,
    )
    test_mode = 0
    key = test_mode

    # BGA 独有流程：获取 pinmap，找到缺 PIN 位置信息
    time_save_find_pinmap()

    # (1) 在各视图中运行 YOLOX/DBNet 收集检测框
    L3 = get_data_location_by_yolo_dbnet(DATA, package_classes)

    # (2) 剔除 OTHER 类型干扰框
    L3 = remove_other_annotations(L3)

    # (3) 寻找尺寸线的配对边界
    L3 = enrich_pairs_with_lines(L3, DATA, key)

    # (4) 整理尺寸线与文本，生成初步候选
    L3 = preprocess_pairs_and_text(L3, key)

    # (5) 执行 SVTR OCR 识别
    L3 = run_svtr_ocr(L3)

    # (6) OCR 后处理，清洗文本候选
    L3 = normalize_ocr_candidates(L3, key)

    # (7) 提取序号/PIN 结构信息
    L3 = extract_pin_serials(L3, package_classes)

    # (8) 匹配尺寸线与文本
    L3 = match_pairs_with_text(L3, key)

    # (9) 整理配对结果
    L3 = finalize_pairs(L3)

    # (10) 语义对齐，生成参数候选
    QFP_parameter_list, nx, ny = compute_qfp_parameters(L3)
    parameter_list = pairs_module.get_QFP_parameter_data(QFP_parameter_list, nx, ny)
    parameter_list = pairs_module.alter_QFP_parameter_data(parameter_list)

    return parameter_list


def time_save_find_pinmap():
    """在独立线程中执行 pinmap 识别，避免长时间阻塞主流程。"""

    result_queue = queue.Queue()
    thread = threading.Thread(target=long_running_task, args=(result_queue,))
    thread.start()

    thread.join(timeout=6)
    if thread.is_alive():
        print("读取pinmap进程花费时间过长，跳过")
        pin_map = np.ones((10, 10))
        pin_num_x_y = np.array([0, 0]).astype(int)
        path = r"yolox_data\\pin_num.txt"
        np.savetxt(path, pin_num_x_y)
    else:
        try:
            pin_map = result_queue.get_nowait()
        except queue.Empty:
            print("Queue is empty, no result available.")
            pin_map = np.ones((10, 10))
            pin_num_x_y = np.array([0, 0]).astype(int)
            path = r"yolox_data\\pin_num.txt"
            np.savetxt(path, pin_num_x_y)

    pin_num_x_y = np.array([pin_map.shape[1], pin_map.shape[0]]).astype(int)
    path = r"yolox_data\\pin_num.txt"
    np.savetxt(path, pin_num_x_y)

    return pin_map


def long_running_task(result_queue):
    """后台线程执行 pinmap 检测并写入结果。"""

    print()
    print("***/开始检测pin/***")
    result = find_pin()
    result_queue.put(result)
    print("***/结束检测pin/***")
    print()
