"""验证 ``run_f4_pipeline`` 串联步骤的脚本级单测。"""

import os
import tempfile
import unittest

from tests.stubbed_dependencies import install_common_pipeline_stubs

install_common_pipeline_stubs()

from packagefiles.PackageExtract import common_pipeline
from packagefiles.PackageExtract.f4_pipeline_runner import run_f4_pipeline


class F4PipelineRunnerTest(unittest.TestCase):
    """确保封装后的 F4 流程在桩环境中可以顺利执行。"""

    def test_run_f4_pipeline_returns_expected_structure(self):
        """运行结果应包含 L3、参数列表以及 nx/ny 数值。"""

        with tempfile.TemporaryDirectory() as tmpdir:
            for view in common_pipeline.DEFAULT_VIEWS:
                open(os.path.join(tmpdir, f"{view}.jpg"), "wb").close()

            result = run_f4_pipeline(tmpdir, package_class="QFP")

        self.assertIn("L3", result)
        self.assertIn("parameters", result)
        self.assertIn("nx", result)
        self.assertIn("ny", result)

        self.assertIsInstance(result["parameters"], list)
        self.assertEqual(len(result["parameters"]), 19)
        self.assertEqual(result["nx"], 0)
        self.assertEqual(result["ny"], 0)

        list_names = {entry["list_name"] for entry in result["L3"]}
        expected_keys = {"top_dbnet_data", "top_yolox_pairs", "top_yolox_num", "bottom_dbnet_data"}
        self.assertTrue(expected_keys.issubset(list_names))


if __name__ == "__main__":
    unittest.main()
