"""针对 ``common_pipeline`` 公共函数的单元测试集合。"""

import os
import tempfile
import unittest
from unittest.mock import call, patch

from tests.stubbed_dependencies import install_common_pipeline_stubs

np = install_common_pipeline_stubs()

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


class PipelineStepFunctionTest(unittest.TestCase):
    """验证新抽取的流程步骤函数是否保持原始语义。"""

    def test_remove_other_annotations_updates_each_view(self):
        """delete_other 应被调用并写回四个视图的结果。"""

        L3 = [
            {"list_name": "top_yolox_num", "list": "top_num"},
            {"list_name": "top_dbnet_data", "list": "top_db"},
            {"list_name": "top_other", "list": "top_other"},
            {"list_name": "bottom_yolox_num", "list": "bottom_num"},
            {"list_name": "bottom_dbnet_data", "list": "bottom_db"},
            {"list_name": "bottom_other", "list": "bottom_other"},
            {"list_name": "side_yolox_num", "list": "side_num"},
            {"list_name": "side_dbnet_data", "list": "side_db"},
            {"list_name": "side_other", "list": "side_other"},
            {"list_name": "detailed_yolox_num", "list": "detail_num"},
            {"list_name": "detailed_dbnet_data", "list": "detail_db"},
            {"list_name": "detailed_other", "list": "detail_other"},
        ]

        with patch(
            "packagefiles.PackageExtract.common_pipeline._pairs_module.delete_other",
            side_effect=lambda other, data: f"{data}-filtered",
        ) as mock_delete_other:
            updated = common_pipeline.remove_other_annotations(L3)

        self.assertEqual(common_pipeline.find_list(updated, "top_yolox_num"), "top_num-filtered")
        self.assertEqual(common_pipeline.find_list(updated, "top_dbnet_data"), "top_db-filtered")
        self.assertEqual(common_pipeline.find_list(updated, "bottom_yolox_num"), "bottom_num-filtered")
        self.assertEqual(common_pipeline.find_list(updated, "side_dbnet_data"), "side_db-filtered")
        self.assertEqual(common_pipeline.find_list(updated, "detailed_yolox_num"), "detail_num-filtered")
        self.assertEqual(mock_delete_other.call_count, 8)

    def test_enrich_pairs_with_lines_reads_existing_images(self):
        """存在图片的视图应调用 find_pairs_length 并写入 L3。"""

        with tempfile.TemporaryDirectory() as tmpdir:
            for view in ("top", "bottom"):
                open(os.path.join(tmpdir, f"{view}.jpg"), "wb").close()

            L3 = [
                {"list_name": "top_yolox_pairs", "list": "top_pairs"},
                {"list_name": "bottom_yolox_pairs", "list": "bottom_pairs"},
                {"list_name": "side_yolox_pairs", "list": "side_pairs"},
                {"list_name": "detailed_yolox_pairs", "list": "detail_pairs"},
            ]

            fake_result = np.array([[1.0] * 13])

            with patch(
                "packagefiles.PackageExtract.common_pipeline._pairs_module.find_pairs_length",
                return_value=fake_result,
            ) as mock_find_pairs_length:
                updated = common_pipeline.enrich_pairs_with_lines(L3, tmpdir, test_mode=1)

        self.assertTrue(np.array_equal(common_pipeline.find_list(updated, "top_yolox_pairs_length"), fake_result))
        self.assertTrue(
            np.array_equal(
                common_pipeline.find_list(updated, "side_yolox_pairs_length"),
                np.empty((0, 13)),
            )
        )
        self.assertEqual(mock_find_pairs_length.call_count, 2)

    def test_extract_pin_serials_handles_bga_branch(self):
        """BGA 模式应使用原始检测框参与 PIN 推断。"""

        detected_serial_nums = [[1, 2, 3, 4]]
        detected_serial_letters = [[5, 6, 7, 8]]
        detected_dbnet = [[9, 10, 11, 12]]

        serial_matrix = [["1", "2", "3", "4", "1"]]
        letter_matrix = [["5", "6", "7", "8", "A"]]

        L3 = [
            {"list_name": "top_yolox_serial_num", "list": []},
            {"list_name": "bottom_yolox_serial_num", "list": []},
            {"list_name": "top_ocr_data", "list": []},
            {"list_name": "bottom_ocr_data", "list": ["raw"]},
            {"list_name": "bottom_BGA_serial_num", "list": detected_serial_nums},
            {"list_name": "bottom_BGA_serial_letter", "list": detected_serial_letters},
            {"list_name": "bottom_border", "list": []},
            {"list_name": "bottom_dbnet_data", "list": detected_dbnet},
        ]

        with patch(
            "packagefiles.PackageExtract.common_pipeline._pairs_module.time_save_find_pinmap",
            return_value=([[1, 1]], [[0, 0]]),
        ) as mock_pinmap, patch(
            "packagefiles.PackageExtract.common_pipeline._pairs_module.find_serial_number_letter",
            return_value=(detected_serial_nums, detected_serial_letters, detected_dbnet),
        ) as mock_find_serial_letter, patch(
            "packagefiles.PackageExtract.common_pipeline._pairs_module.ocr_data",
            return_value=["raw"],
        ) as mock_ocr_data, patch(
            "packagefiles.PackageExtract.common_pipeline._pairs_module.filter_bottom_ocr_data",
            return_value=(serial_matrix, letter_matrix, ["filtered"]),
        ) as mock_filter_bottom, patch(
            "packagefiles.PackageExtract.common_pipeline._pairs_module.find_pin_num_pin_1",
            return_value=(10, 12, [1, 2]),
        ) as mock_find_pin_num_pin_1:
            updated = common_pipeline.extract_pin_serials(L3, "BGA")

        self.assertEqual(
            common_pipeline.find_list(updated, "bottom_BGA_serial_num"),
            detected_serial_nums,
        )
        self.assertEqual(
            common_pipeline.find_list(updated, "bottom_BGA_serial_letter"),
            detected_serial_letters,
        )
        self.assertEqual(common_pipeline.find_list(updated, "bottom_ocr_data"), ["raw"])
        self.assertEqual(common_pipeline.find_list(updated, "pin_num_x_serial"), 10)
        self.assertEqual(common_pipeline.find_list(updated, "pin_num_y_serial"), 12)
        self.assertEqual(common_pipeline.find_list(updated, "pin_1_location"), [1, 2])
        self.assertEqual(common_pipeline.find_list(updated, "loss_pin"), "None")
        mock_pinmap.assert_called_once()
        mock_find_serial_letter.assert_called_once_with(
            detected_serial_nums,
            detected_serial_letters,
            np.array(detected_dbnet),
        )
        mock_ocr_data.assert_called_once()
        mock_filter_bottom.assert_called_once()
        serial_matrix_arg, letter_matrix_arg, *_ = mock_find_pin_num_pin_1.call_args[0]
        self.assertEqual(serial_matrix_arg, serial_matrix)
        self.assertEqual(letter_matrix_arg, letter_matrix)


if __name__ == "__main__":
    unittest.main()
