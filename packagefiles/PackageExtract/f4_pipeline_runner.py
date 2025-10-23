"""封装 F4.6-F4.9 流程的便捷调用入口。"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from packagefiles.PackageExtract import common_pipeline


def run_f4_pipeline(
    image_root: str,
    package_class: str,
    key: int = 0,
    test_mode: int = 0,
    view_names: Optional[Sequence[str]] = None,
):
    """串联执行 F4 阶段的主要函数，返回参数列表与中间结果。

    :param image_root: 存放 ``top/bottom/side/detailed`` 视图图片的目录。
    :param package_class: 封装类型，例如 ``"QFP"``、``"BGA"``。
    :param key: 与历史实现一致的流程参数，用于控制 OCR 清洗策略。
    :param test_mode: 传递给 ``find_pairs_length`` 的调试开关。
    :param view_names: 自定义视图顺序；默认为 ``common_pipeline.DEFAULT_VIEWS``。
    :returns: ``dict``，包含 ``L3`` 数据、参数候选列表以及 ``nx``/``ny``。
    """

    views: Iterable[str] = view_names or common_pipeline.DEFAULT_VIEWS

    L3 = common_pipeline.get_data_location_by_yolo_dbnet(image_root, package_class, view_names=views)
    L3 = common_pipeline.remove_other_annotations(L3)
    L3 = common_pipeline.enrich_pairs_with_lines(L3, image_root, test_mode)
    L3 = common_pipeline.preprocess_pairs_and_text(L3, key)
    L3 = common_pipeline.run_svtr_ocr(L3)
    L3 = common_pipeline.normalize_ocr_candidates(L3, key)
    L3 = common_pipeline.extract_pin_serials(L3, package_class)
    L3 = common_pipeline.match_pairs_with_text(L3, key)
    L3 = common_pipeline.finalize_pairs(L3)
    parameters, nx, ny = common_pipeline.compute_qfp_parameters(L3)

    return {
        "L3": L3,
        "parameters": parameters,
        "nx": nx,
        "ny": ny,
    }
