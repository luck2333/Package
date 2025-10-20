"""针对 ``common_pipeline`` 公共函数的单元测试集合。"""

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import call, patch

try:  # pragma: no cover - 当环境未安装 numpy 时执行备用实现
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    fake_np = types.ModuleType("numpy")

    def _to_matrix(data, dtype=float):
        return [[dtype(value) for value in row] for row in data]

    def array(data, dtype=float):
        return _to_matrix(data, dtype=dtype)

    def empty(shape):
        rows, cols = shape
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    def around(data, decimals=0):
        factor = 10 ** decimals
        return [[round(value * factor) / factor for value in row] for row in data]

    def array_equal(left, right):
        return left == right

    fake_np.array = array  # 伪造 ``np.array``，返回嵌套列表
    fake_np.empty = empty  # 伪造 ``np.empty``，生成零矩阵
    fake_np.around = around  # 伪造 ``np.around``，实现四舍五入
    fake_np.array_equal = array_equal  # 伪造 ``np.array_equal``，用于断言
    sys.modules["numpy"] = fake_np
    import numpy as np

fake_detr_module = types.ModuleType("packagefiles.PackageExtract.DETR_BGA")


def _stub_detr(*_args, **_kwargs):
    """返回全为空数组的 DETR 结果，占位真实推理函数。"""
    return tuple(np.empty((0, 4)) for _ in range(10))


fake_detr_module.DETR_BGA = _stub_detr
sys.modules["packagefiles.PackageExtract.DETR_BGA"] = fake_detr_module

fake_onnx_module = types.ModuleType("packagefiles.PackageExtract.onnx_use")
fake_onnx_module.Run_onnx_det = lambda *_args, **_kwargs: []  # 伪造 DBNet 输出
sys.modules["packagefiles.PackageExtract.onnx_use"] = fake_onnx_module

fake_function_tool_module = types.ModuleType("packagefiles.PackageExtract.function_tool")


def _stub_empty_folder(path):
    """测试环境下复刻 ``empty_folder``，删除目录并忽略缺失异常。"""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _stub_set_image_size(_src, _dst):
    """测试环境的 ``set_Image_size`` 占位实现。"""
    return None


fake_function_tool_module.empty_folder = _stub_empty_folder
fake_function_tool_module.set_Image_size = _stub_set_image_size
sys.modules["packagefiles.PackageExtract.function_tool"] = fake_function_tool_module

fake_yolox_package = types.ModuleType("packagefiles.PackageExtract.yolox_onnx_py")
fake_yolox_module = types.ModuleType(
    "packagefiles.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2"
)


def _stub_yolox(*_args, **_kwargs):
    """返回空数组的 YOLOX 结果，避免依赖真实模型。"""
    return tuple(np.empty((0, 4)) for _ in range(10))


fake_yolox_module.begain_output_pairs_data_location = _stub_yolox
fake_yolox_package.onnx_QFP_pairs_data_location2 = fake_yolox_module
sys.modules["packagefiles.PackageExtract.yolox_onnx_py"] = fake_yolox_package
sys.modules[
    "packagefiles.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2"
] = fake_yolox_module

from packagefiles.PackageExtract import common_pipeline


class PrepareWorkspaceTest(unittest.TestCase):
    """验证准备工作区函数的副作用与历史实现一致。"""

    def test_prepare_workspace_recreates_directories_and_copies_files(self):
        """确认目录被清空重建且源图像完整复制。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            data_copy_dir = os.path.join(tmpdir, "data_copy")
            data_bottom_crop_dir = os.path.join(tmpdir, "data_bottom_crop")
            onnx_output_dir = os.path.join(tmpdir, "onnx_output")
            opencv_output_dir = os.path.join(tmpdir, "opencv_output")

            os.makedirs(data_dir, exist_ok=True)

            original_files = {}
            for view_name in ("top", "bottom", "side", "detailed"):
                file_path = os.path.join(data_dir, f"{view_name}.jpg")
                with open(file_path, "w", encoding="utf-8") as handle:
                    handle.write(f"content-{view_name}")
                original_files[file_path] = f"content-{view_name}"

            with patch("packagefiles.PackageExtract.common_pipeline.set_Image_size") as mock_set_image_size:
                mock_set_image_size.side_effect = lambda src, dst: None
                common_pipeline.prepare_workspace(
                    data_dir,
                    data_copy_dir,
                    data_bottom_crop_dir,
                    onnx_output_dir,
                    opencv_output_dir,
                )

            for directory in (
                data_copy_dir,
                data_bottom_crop_dir,
                onnx_output_dir,
                opencv_output_dir,
            ):
                self.assertTrue(os.path.isdir(directory))

            recreated_files = {}
            for filename in os.listdir(data_dir):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as handle:
                    recreated_files[filename] = handle.read()

            expected_files = {os.path.basename(path): content for path, content in original_files.items()}
            self.assertEqual(recreated_files, expected_files)

            mock_calls = [call.args[0] for call in mock_set_image_size.call_args_list]
            for view_name in ("top", "bottom", "side", "detailed"):
                self.assertIn(os.path.join(data_dir, f"{view_name}.jpg"), mock_calls)


class YoloClassifyTest(unittest.TestCase):
    """验证 YOLO 分类辅助函数的两条分支逻辑。"""

    def test_yolo_classify_bga_merges_detr_outputs(self):
        """BGA 模式下应融合 DETR 返回的 pin 与边框结果。"""
        yolox_pairs = np.array([[1, 2, 3, 4]], dtype=float)
        yolox_num = np.array([[5, 6, 7, 8]], dtype=float)
        yolox_serial_num = np.array([[9, 10, 11, 12]], dtype=float)
        yolox_pin = np.array([[13, 14, 15, 16]], dtype=float)
        yolox_other = np.array([[17, 18, 19, 20]], dtype=float)
        yolox_pad = np.array([[21, 22, 23, 24]], dtype=float)
        yolox_border = np.array([[25, 26, 27, 28]], dtype=float)
        yolox_angle_pairs = np.array([[29, 30, 31, 32]], dtype=float)
        yolox_serial_num_bga = np.array([[33, 34, 35, 36]], dtype=float)
        yolox_serial_letter_bga = np.array([[37, 38, 39, 40]], dtype=float)

        detr_pin = np.array([[100, 101, 102, 103]], dtype=float)
        detr_border = np.array([[104, 105, 106, 107]], dtype=float)
        detr_serial_num = np.array([[108, 109, 110, 111]], dtype=float)
        detr_serial_letter = np.array([[112, 113, 114, 115]], dtype=float)

        with patch(
            "packagefiles.PackageExtract.common_pipeline.begain_output_pairs_data_location",
            return_value=(
                yolox_pairs,
                yolox_num,
                yolox_serial_num,
                yolox_pin,
                yolox_other,
                yolox_pad,
                yolox_border,
                yolox_angle_pairs,
                yolox_serial_num_bga,
                yolox_serial_letter_bga,
            ),
        ) as mock_yolox, patch(
            "packagefiles.PackageExtract.common_pipeline.DETR_BGA",
            return_value=(
                np.empty((0, 4)),
                np.empty((0, 4)),
                np.empty((0, 4)),
                detr_pin,
                np.empty((0, 4)),
                np.empty((0, 4)),
                detr_border,
                np.empty((0, 4)),
                detr_serial_num,
                detr_serial_letter,
            ),
        ) as mock_detr:
            outputs = common_pipeline.yolo_classify("dummy.jpg", "BGA")

        self.assertTrue(np.array_equal(outputs[0], yolox_pairs))
        self.assertTrue(np.array_equal(outputs[1], yolox_num))
        self.assertTrue(np.array_equal(outputs[2], yolox_serial_num))
        self.assertTrue(np.array_equal(outputs[3], detr_pin))
        self.assertTrue(np.array_equal(outputs[4], yolox_other))
        self.assertTrue(np.array_equal(outputs[5], yolox_pad))
        self.assertTrue(np.array_equal(outputs[6], detr_border))
        self.assertTrue(np.array_equal(outputs[7], yolox_angle_pairs))
        self.assertTrue(np.array_equal(outputs[8], detr_serial_num))
        self.assertTrue(np.array_equal(outputs[9], detr_serial_letter))
        self.assertEqual(mock_yolox.call_count, 1)
        self.assertEqual(mock_detr.call_count, 1)

    def test_yolo_classify_qfp_rounds_outputs(self):
        """QFP 模式下应对检测结果进行保留两位小数的取整。"""
        floating = np.array([[0.12345, 0.54321, 0.99999, 0.00001]], dtype=float)
        zero = np.empty((0, 4))

        with patch(
            "packagefiles.PackageExtract.common_pipeline.begain_output_pairs_data_location",
            return_value=(floating, floating, floating, zero, zero, zero, zero, floating, zero, zero),
        ):
            outputs = common_pipeline.yolo_classify("dummy.jpg", "QFP")

        self.assertTrue(np.array_equal(outputs[0], np.around(floating, decimals=2)))
        self.assertTrue(np.array_equal(outputs[1], np.around(floating, decimals=2)))
        self.assertTrue(np.array_equal(outputs[7], np.around(floating, decimals=2)))


class GetDataLocationTest(unittest.TestCase):
    """检验 YOLO 与 DBNet 汇总函数的返回内容。"""

    def test_get_data_location_aggregates_per_view_results(self):
        """确保每个视图都追加对应的 L3 项并保留 BGA 序列输出。"""
        dbnet_top = np.array([[1, 2, 3, 4]], dtype=float)
        dbnet_bottom = np.array([[5, 6, 7, 8]], dtype=float)

        yolox_top = (
            np.array([[10]], dtype=float),
            np.array([[11]], dtype=float),
            np.array([[12]], dtype=float),
            np.array([[13]], dtype=float),
            np.array([[14]], dtype=float),
            np.array([[15]], dtype=float),
            np.array([[16]], dtype=float),
            np.array([[17]], dtype=float),
            np.array([[18]], dtype=float),
            np.array([[19]], dtype=float),
        )
        yolox_bottom = (
            np.array([[20]], dtype=float),
            np.array([[21]], dtype=float),
            np.array([[22]], dtype=float),
            np.array([[23]], dtype=float),
            np.array([[24]], dtype=float),
            np.array([[25]], dtype=float),
            np.array([[26]], dtype=float),
            np.array([[27]], dtype=float),
            np.array([[28]], dtype=float),
            np.array([[29]], dtype=float),
        )

        with patch("packagefiles.PackageExtract.common_pipeline.os.path.exists", return_value=True), patch(
            "packagefiles.PackageExtract.common_pipeline.dbnet_get_text_box",
            side_effect=[dbnet_top, dbnet_bottom],
        ), patch(
            "packagefiles.PackageExtract.common_pipeline.yolo_classify",
            side_effect=[yolox_top, yolox_bottom],
        ):
            result = common_pipeline.get_data_location_by_yolo_dbnet(
                "fake_path", "BGA", view_names=("top", "bottom")
            )

        expected_names = [
            "top_dbnet_data",
            "top_yolox_pairs",
            "top_yolox_num",
            "top_yolox_serial_num",
            "top_pin",
            "top_other",
            "top_pad",
            "top_border",
            "top_angle_pairs",
            "bottom_dbnet_data",
            "bottom_yolox_pairs",
            "bottom_yolox_num",
            "bottom_yolox_serial_num",
            "bottom_pin",
            "bottom_other",
            "bottom_pad",
            "bottom_border",
            "bottom_angle_pairs",
            "bottom_BGA_serial_letter",
            "bottom_BGA_serial_num",
        ]

        self.assertEqual([entry["list_name"] for entry in result], expected_names)
        self.assertTrue(np.array_equal(result[0]["list"], dbnet_top))
        self.assertTrue(np.array_equal(result[9]["list"], dbnet_bottom))
        self.assertTrue(np.array_equal(result[-2]["list"], yolox_bottom[9]))
        self.assertTrue(np.array_equal(result[-1]["list"], yolox_bottom[8]))


if __name__ == "__main__":
    unittest.main()
