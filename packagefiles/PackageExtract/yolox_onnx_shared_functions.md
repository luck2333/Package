# YOLOX ONNX 公共函数整理

本文档梳理 `packagefiles/PackageExtract/yolox_onnx_py/` 目录下各推理脚本中反复出现的辅助函数，便于后续重构时识别可抽取的公共模块。统计基于当前仓库中的脚本内容。

## 统计口径
- 通过遍历 `yolox_onnx_py` 文件夹下的所有 `.py` 文件，记录顶层定义的函数名。
- 若同名函数出现在至少两个脚本内，则归入“公共函数”列表。
- 结合代码逐一说明函数职责及在 F4.6-F4.9 流程中的作用。 

## 公共函数清单

| 函数名 | 主要职责 | 常见出现文件（节选） | F4.6-F4.9 相关说明 |
| --- | --- | --- | --- |
| `make_parser` | 构建命令行参数解析器，约定 ONNX 模型、输入图片与输出目录路径。 | `onnx_QFP_pairs_data_location2.py`、`onnx_output_other_location.py`、`onnx_detect_pin.py` 等 | 统一不同检测脚本的启动方式，便于批量调用模型完成标尺、引脚、外框等推理。 |
| `mkdir` | 保证输出目录存在，避免保存可视化/结果时出错。 | 同上 | 运行 YOLO/ONNX 推理前准备输出目录，常用于保存 F4.6-F4.9 阶段的调试图片。 |
| `preprocess` | 将输入图片缩放、填充并转换成 YOLOX 期望的张量格式。 | 同上 | F4 阶段所有检测模型进入推理前均依赖该预处理逻辑。 |
| `nms` | 执行单类别非极大抑制，去除重叠检测框。 | 同上 | F4.6-F4.9 中的检测模型（引脚、标尺、外框等）都通过该函数筛选候选框。 |
| `multiclass_nms` | 根据 `class_agnostic` 标志选择类别无关或类别敏感的多类别 NMS。 | 同上 | 抽象封装多类别后处理，保证不同检测脚本对 NMS 的调用一致。 |
| `multiclass_nms_class_agnostic` | 类别无关的多类别 NMS 实现。 | 同上 | 主要用于多数只有单类别输出的模型，简化后处理。 |
| `multiclass_nms_class_aware` | 类别敏感的多类别 NMS 实现。 | 同上 | 针对存在多个标签（如文字、引脚类型）的模型，保留类别信息。 |
| `demo_postprocess` | 将输出张量映射回原始图像尺度，生成真实坐标。 | 同上 | 连接 ONNX 推理输出与实际几何信息的桥梁，是 F4 阶段获取坐标数据的关键。 |
| `onnx_inference` | 加载 ONNXRuntime Session 并执行模型推理，返回原始输出。 | `onnx_QFP_pairs_data_location2.py`、`onnx_output_bottom_body_location.py` 等 | 所有 YOLOX 模型统一的推理入口。 |
| `output_pairs_data_location` | 结合 `onnx_inference`、`demo_postprocess` 与 NMS，将检测框转换为结构化结果。 | 同上 | F4.6-F4.9 中多类几何元素（标尺、Pin、外框、侧视 standoff 等）的最终输出接口。 |
| `get_img_info` | 计算图像尺寸、缩放比例等信息，支撑后续坐标还原。 | `onnx_QFP_pairs_data_location2.py`、`onnx_output_pin_num4.py` 等 | 在根据缩放比例反推真实坐标时使用。 |
| `vis` | 将检测框绘制回原图，输出调试图像。 | 同上 | 便于验证 F4.6-F4.9 检测结果，常作为调试选项。 |
| `get_rotate_crop_image` | 根据四点坐标裁剪旋转区域，常用于序号/文字检测后的补裁。 | `onnx_output_serial_number_letter_location.py`、`onnx_output_pin_num4.py` 等 | 支撑 F4.7-F4.8 的 OCR 前裁剪及几何精修。 |
| `find_the_only_body` | 在多候选框中选出面积最大的“本体”框，避免重复。 | `onnx_output_bottom_body_location.py`、`onnx_output_other_location.py` 等 | 用于 body、外框等检测，保证流程只保留最合理的本体候选。 |
| `onnx_output_pairs_data_pin_5` | （仅 `onnx_detect_pin.py` 与 `onnx_yolox检测模板.py`）封装 PIN 检测输出逻辑。 | `onnx_detect_pin.py`、`onnx_yolox检测模板.py` | 供 F4.6/F4.7 的 PIN 位置识别脚本调用。 |

> **注**：以上列表覆盖所有在两个及以上脚本中复用的函数。若某些脚本另有特定后处理逻辑（如序号排序、特定字段写入），则未纳入“公共函数”范畴。

## 后续建议
1. **抽取公共模块**：可将上述函数移动到独立的工具模块（如 `yolox_utils.py`），减少每个推理脚本中的重复定义。
2. **统一参数约定**：整理 `make_parser` 默认参数，将模型路径、输入尺寸等配置集中化，方便批量部署与测试。
3. **完善文档**：结合 F4.6-F4.9 的业务流程，在 `F4_onnx_method_notes.md` 等文档中引用本表格，快速定位依赖的通用能力。

