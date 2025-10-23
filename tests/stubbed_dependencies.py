"""为 ``common_pipeline`` 相关单元测试提供依赖桩对象。"""

from __future__ import annotations

import shutil
import sys
import types
from typing import Any, Iterable, Tuple


def _ensure_numpy():
    """确保 ``numpy`` 可用；若未安装则注册一个简易替身。"""

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        fake_np = types.ModuleType("numpy")

        def _to_matrix(data: Iterable[Iterable[Any]], dtype=float):
            return [[dtype(value) for value in row] for row in data]

        def array(data, dtype=float):
            return _to_matrix(data, dtype=dtype)

        def empty(shape: Tuple[int, int]):
            rows, cols = shape
            return [[0.0 for _ in range(cols)] for _ in range(rows)]

        def around(data, decimals=0):
            factor = 10 ** decimals
            return [[round(value * factor) / factor for value in row] for row in data]

        def array_equal(left, right):
            return left == right

        fake_np.array = array  # type: ignore[attr-defined]
        fake_np.empty = empty  # type: ignore[attr-defined]
        fake_np.around = around  # type: ignore[attr-defined]
        fake_np.array_equal = array_equal  # type: ignore[attr-defined]
        sys.modules["numpy"] = fake_np

    return __import__("numpy")


def install_common_pipeline_stubs():
    """向 ``sys.modules`` 注入桩模块，复刻提取流程依赖的接口。"""

    np = _ensure_numpy()

    if "packagefiles.PackageExtract.get_pairs_data_present5_test" in sys.modules:
        return np

    fake_detr_module = types.ModuleType("packagefiles.PackageExtract.DETR_BGA")

    def _stub_detr(*_args, **_kwargs):
        return tuple(np.empty((0, 4)) for _ in range(10))

    fake_detr_module.DETR_BGA = _stub_detr  # type: ignore[attr-defined]
    sys.modules["packagefiles.PackageExtract.DETR_BGA"] = fake_detr_module

    fake_onnx_module = types.ModuleType("packagefiles.PackageExtract.onnx_use")
    fake_onnx_module.Run_onnx_det = lambda *_args, **_kwargs: []  # type: ignore[attr-defined]
    sys.modules["packagefiles.PackageExtract.onnx_use"] = fake_onnx_module

    fake_function_tool_module = types.ModuleType("packagefiles.PackageExtract.function_tool")

    def _stub_empty_folder(path):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass

    def _stub_set_image_size(_src, _dst):
        return None

    def _stub_find_list(dict_list, listname):
        for item in dict_list:
            if item.get("list_name") == listname:
                return item.get("list")
        return []

    def _stub_recite_data(dict_list, listname, value):
        for item in dict_list:
            if item.get("list_name") == listname:
                item["list"] = value
                break
        else:
            dict_list.append({"list_name": listname, "list": value})
        return dict_list

    fake_function_tool_module.empty_folder = _stub_empty_folder  # type: ignore[attr-defined]
    fake_function_tool_module.set_Image_size = _stub_set_image_size  # type: ignore[attr-defined]
    fake_function_tool_module.find_list = _stub_find_list  # type: ignore[attr-defined]
    fake_function_tool_module.recite_data = _stub_recite_data  # type: ignore[attr-defined]
    sys.modules["packagefiles.PackageExtract.function_tool"] = fake_function_tool_module

    fake_pairs_module = types.ModuleType("packagefiles.PackageExtract.get_pairs_data_present5_test")

    def _return_original(_other, data):
        return data

    def _stub_find_pairs_length(_img_path, _pairs, _test_mode):
        return np.empty((0, 13))

    def _stub_get_better_data_1(
        top_pairs,
        bottom_pairs,
        side_pairs,
        detailed_pairs,
        _key,
        top_dbnet,
        bottom_dbnet,
        side_dbnet,
        detailed_dbnet,
    ):
        return (
            top_pairs,
            bottom_pairs,
            side_pairs,
            detailed_pairs,
            top_pairs,
            bottom_pairs,
            side_pairs,
            detailed_pairs,
            top_dbnet,
            bottom_dbnet,
        )

    def _stub_svtr(top_dbnet_all, bottom_dbnet_all, side_dbnet, detailed_dbnet):
        return (0.0, 0.0, top_dbnet_all, bottom_dbnet_all, side_dbnet, detailed_dbnet)

    def _stub_data_wrangling(
        _key,
        top_dbnet,
        bottom_dbnet,
        side_dbnet,
        detailed_dbnet,
        top_ocr,
        bottom_ocr,
        side_ocr,
        detailed_ocr,
        _top_num,
        _bottom_num,
        _side_num,
        _detailed_num,
    ):
        return top_ocr, bottom_ocr, side_ocr, detailed_ocr

    def _stub_find_pin(top_serial, bottom_serial, top_ocr, bottom_ocr):
        empty = np.empty((0, 5))
        return empty, empty, top_ocr, bottom_ocr

    def _stub_find_bga_pin(bottom_serial_num, bottom_serial_letter, bottom_ocr):
        return bottom_serial_num, bottom_serial_letter, bottom_ocr

    def _stub_find_pin_num_pin_1(_serial_numbers_data, _serial_letters_data, _serial_numbers, _serial_letters):
        return 0, 0, np.array([0, 0])

    def _stub_mpd(
        _key,
        top_pairs,
        bottom_pairs,
        side_pairs,
        detailed_pairs,
        _side_angle,
        _detailed_angle,
        _top_border,
        _bottom_border,
        top_ocr,
        bottom_ocr,
        side_ocr,
        detailed_ocr,
    ):
        return top_ocr, bottom_ocr, side_ocr, detailed_ocr

    def _stub_get_better_data_2(
        top_ocr,
        bottom_ocr,
        side_ocr,
        detailed_ocr,
        _top_length,
        _bottom_length,
        _side_length,
        _detailed_length,
        top_pairs_copy,
        bottom_pairs_copy,
        side_pairs_copy,
        detailed_pairs_copy,
    ):
        return (
            top_ocr,
            bottom_ocr,
            side_ocr,
            detailed_ocr,
            top_pairs_copy,
            bottom_pairs_copy,
            side_pairs_copy,
            detailed_pairs_copy,
        )

    def _stub_get_serial(_top_serial, _bottom_serial):
        return 0, 0

    def _stub_get_qfp_body(
        _top_pairs,
        _top_length,
        _bottom_pairs,
        _bottom_length,
        _top_border,
        _bottom_border,
        _top_ocr,
        _bottom_ocr,
    ):
        return 0.0, 0.0

    def _stub_get_qfp_parameter_list(_top, _bottom, _side, _detailed, _body_x, _body_y):
        base = []
        for _ in range(19):
            base.append({"maybe_data": [], "maybe_data_num": 0})
        return base

    def _stub_resort_parameter_list(data):
        return data

    def _stub_get_qfp_high(values):
        return values

    def _stub_get_qfp_pitch(_values, _body_x, _body_y, _nx, _ny):
        return [], []

    def _stub_get_qfp_parameter_data(_parameter_list, _nx, _ny):
        return [["", "", "", ""] for _ in range(19)]

    def _stub_alter_qfp_parameter_data(parameter_list):
        return parameter_list

    fake_pairs_module.delete_other = _return_original  # type: ignore[attr-defined]
    fake_pairs_module.find_pairs_length = _stub_find_pairs_length  # type: ignore[attr-defined]
    fake_pairs_module.get_better_data_1 = _stub_get_better_data_1  # type: ignore[attr-defined]
    fake_pairs_module.SVTR = _stub_svtr  # type: ignore[attr-defined]
    fake_pairs_module.data_wrangling = _stub_data_wrangling  # type: ignore[attr-defined]
    fake_pairs_module.find_PIN = _stub_find_pin  # type: ignore[attr-defined]
    fake_pairs_module.find_BGA_PIN = _stub_find_bga_pin  # type: ignore[attr-defined]
    fake_pairs_module.find_pin_num_pin_1 = _stub_find_pin_num_pin_1  # type: ignore[attr-defined]
    fake_pairs_module.MPD = _stub_mpd  # type: ignore[attr-defined]
    fake_pairs_module.get_better_data_2 = _stub_get_better_data_2  # type: ignore[attr-defined]
    fake_pairs_module.get_serial = _stub_get_serial  # type: ignore[attr-defined]
    fake_pairs_module.get_QFP_body = _stub_get_qfp_body  # type: ignore[attr-defined]
    fake_pairs_module.get_QFP_parameter_list = _stub_get_qfp_parameter_list  # type: ignore[attr-defined]
    fake_pairs_module.resort_parameter_list_2 = _stub_resort_parameter_list  # type: ignore[attr-defined]
    fake_pairs_module.get_QFP_high = _stub_get_qfp_high  # type: ignore[attr-defined]
    fake_pairs_module.get_QFP_pitch = _stub_get_qfp_pitch  # type: ignore[attr-defined]
    fake_pairs_module.get_QFP_parameter_data = _stub_get_qfp_parameter_data  # type: ignore[attr-defined]
    fake_pairs_module.alter_QFP_parameter_data = _stub_alter_qfp_parameter_data  # type: ignore[attr-defined]
    sys.modules["packagefiles.PackageExtract.get_pairs_data_present5_test"] = fake_pairs_module

    fake_yolox_package = types.ModuleType("packagefiles.PackageExtract.yolox_onnx_py")
    fake_yolox_module = types.ModuleType(
        "packagefiles.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2"
    )

    def _stub_yolox(*_args, **_kwargs):
        return tuple(np.empty((0, 4)) for _ in range(10))

    fake_yolox_module.begain_output_pairs_data_location = _stub_yolox  # type: ignore[attr-defined]
    fake_yolox_package.onnx_QFP_pairs_data_location2 = fake_yolox_module  # type: ignore[attr-defined]
    sys.modules["packagefiles.PackageExtract.yolox_onnx_py"] = fake_yolox_package
    sys.modules[
        "packagefiles.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2"
    ] = fake_yolox_module

    return np
