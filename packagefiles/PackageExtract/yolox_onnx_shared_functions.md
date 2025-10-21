# YOLOX ONNX 公共函数整理

在梳理 `packagefiles/PackageExtract/yolox_onnx_py/` 目录的脚本时，发现多份 YOLOX ONNX 推理代码重复维护相同的预处理、后处理与可视化逻辑。本次重构已将这些基础函数集中到 `yolox_onnx_py/yolox_onnx_shared.py`，方便统一维护，本文档同步记录相关函数职责与使用方式。

## 统计口径
- 遍历 `yolox_onnx_py` 文件夹下的全部 `.py` 脚本，收集顶层函数定义。
- 若函数在两个及以上脚本中复用，则归入“公共函数”列表，并说明它们在 F4.6-F4.9 流程中的定位。

## 公共模块 `yolox_onnx_shared.py`
`yolox_onnx_shared.py` 汇总了 YOLOX 推理的通用工具函数，脚本可按下面的方式复用（兼容包内导入与单脚本执行）：

```python
try:
    from packagefiles.PackageExtract.yolox_onnx_py.yolox_onnx_shared import (
        demo_postprocess,
        mkdir,
        multiclass_nms,
        preprocess,
        vis,
    )
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    from yolox_onnx_shared import (
        demo_postprocess,
        mkdir,
        multiclass_nms,
        preprocess,
        vis,
    )
```

| 函数名 | 主要职责 | F4.6-F4.9 相关说明 |
| --- | --- | --- |
| `mkdir` / `ensure_dir` | 保证输出目录存在，避免保存可视化/结果时出错。 | 推理前创建结果目录，常用于保存调试图片与中间输出。 |
| `preprocess` | 将输入图片缩放、填充并转换成 YOLOX 期望的张量格式。 | 所有 YOLOX ONNX 模型进入推理前的统一预处理。 |
| `nms` | 执行单类别非极大抑制，去除重叠检测框。 | 对标尺、引脚、外框等检测结果进行去重。 |
| `multiclass_nms` / `multiclass_nms_class_agnostic` / `multiclass_nms_class_aware` | 多类别 NMS 封装，根据 `class_agnostic` 标志切换类别无关/类别敏感模式。 | 确保不同检测脚本共享一致的多类别后处理策略。 |
| `demo_postprocess` | 将模型输出映射回原始图像尺度，生成真实坐标。 | 连接 ONNX 推理输出与实际几何数据的关键环节。 |
| `vis` | 将检测框绘制回原图，输出调试图像。 | 便于核对 F4.6-F4.9 阶段的检测结果与标注。 |

## 仍在各脚本内维护的共通逻辑
部分函数与业务模型高度耦合（如单脚本特有的 `onnx_inference`、`output_pairs_data_location`、`find_the_only_body` 等），暂保留在原文件中，后续重构可视具体需求再行抽取。下表列出其中复用频率较高的代表：

| 函数名 | 常见出现文件（节选） | 说明 |
| --- | --- | --- |
| `onnx_inference` | `onnx_QFP_pairs_data_location2.py`、`onnx_output_bottom_body_location.py` 等 | 建立 ONNXRuntime Session 并执行推理，是各检测脚本的入口封装。 |
| `output_pairs_data_location` | 与上同 | 结合推理输出、NMS、类别后处理，转换为业务字段。 |
| `get_img_info` | `onnx_QFP_pairs_data_location2.py`、`onnx_output_pin_num4.py` 等 | 读取图像宽高信息，为比例还原与边界判断提供支持。 |
| `get_rotate_crop_image` | `onnx_output_serial_number_letter_location.py`、`onnx_output_pin_num4.py` 等 | 根据四点坐标裁剪旋转区域，供 OCR 或精细化处理使用。 |
| `find_the_only_body` | `onnx_output_bottom_body_location.py`、`onnx_output_other_location.py` | 在多候选框中挑选最合理的封装本体。 |

## 后续建议
1. **统一参数约定**：在 `make_parser` 等入口函数中收敛默认模型路径与输入尺寸，避免脚本间默认值漂移。
2. **持续抽查复用点**：若后续新增脚本出现与 `yolox_onnx_shared.py` 重复的处理逻辑，优先回填到公共模块中，保持推理行为一致。
3. **配套文档同步**：与 `F4_onnx_method_notes.md` 等说明文件互相引用本表内容，确保重构后的依赖关系清晰可查。

