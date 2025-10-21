# F4.6-F4.9 阶段所依赖的 ONNX 工具函数说明

本文汇总 `onnx_use.py` 与 `yolox_onnx_py/` 目录中，在流程图 F4.6-F4.9 阶段会被调用的关键函数及其职责，便于后续重构快速定位依赖关系。公共的 YOLOX 预处理、NMS 与可视化逻辑已统一沉淀在 `yolox_onnx_py/yolox_onnx_shared.py`，各检测脚本通过导入该模块共享相同实现。

## 1. `onnx_use.py` OCR 识别辅助
- `det_rec_functions.get_boxes(image_path, show)`：封装 DBNet 检测流程，返回按照行序排序后的文本框列表，为后续序号提取与标尺匹配提供基础几何信息。 【F:packagefiles/PackageExtract/onnx_use.py†L205-L243】
- `det_rec_functions.recognition_img(dt_boxes, Is_crop)`：批量裁剪检测框并执行识别模型，产出文字与置信度，用于 F4.6 阶段的序号拆分与 F4.9 阶段的参数计算。 【F:packagefiles/PackageExtract/onnx_use.py†L520-L560】
- `Run_onnx(image_path, dt_boxes)` / `Run_onnx_det(image_path)`：分别提供检测+识别一体化流程与仅检测流程，是 `QFP_extract`、`BGA_extract` 中触发 OCR 的入口。 【F:packagefiles/PackageExtract/onnx_use.py†L548-L606】

## 2. `yolox_onnx_py/` 检测脚本概览
- `onnx_QFP_pairs_data_location2.py`
  - `onnx_inference(img_path, package_classes, weight)`：执行 YOLOX 推理并分类整理 pairs、引脚、外框等检测结果，是 F4.7/F4.8 阶段匹配流程的核心数据来源。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_QFP_pairs_data_location2.py†L412-L508】
  - `output_pairs_data_location(cls, bboxes, package_classes)`：将模型输出拆分为标尺、引脚、角度等业务字段，供 `MPD`、`get_better_data_2` 使用。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_QFP_pairs_data_location2.py†L509-L860】
  - `begain_output_pairs_data_location(img_path, package_classes)`：封装完整检测流程，返回 F4.6-F4.8 需要的全部几何位置信息。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_QFP_pairs_data_location2.py†L862-L906】

- `onnx_output_serial_number_letter_location.py`
  - `onnx_inference(img_path)`：针对序号字母的检测推理，输出候选框集合，支撑 F4.6 的序号拆分。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_serial_number_letter_location.py†L327-L382】
  - `begain_output_serial_number_letter_location(img_path)`：入口函数，返回检测到的字母框列表，供 `find_serial_number_letter_QFP` 等函数消费。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_serial_number_letter_location.py†L418-L447】

- `onnx_output_pin_num4.py`
  - `onnx_inference(img_path, conf)`：执行引脚数量检测，并结合阈值生成 PIN、Pad 等候选框，覆盖 F4.6 中引脚行列数推断的输入。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_pin_num4.py†L489-L534】
  - `output_pairs_data_location(cls, bboxes)`：解析模型输出，提取 PIN 框坐标，为后续 pin map 计算提供基础。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_pin_num4.py†L1021-L1057】
  - `begain_output_pin_num(img_path, conf)` / `begain_output_pin_num_pin_map()`：分别封装整体检测流程与 pin map 生成逻辑，是 F4.6 阶段 `find_PIN`、`find_BGA_PIN` 所依赖的入口。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_pin_num4.py†L1058-L1264】【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_pin_num4.py†L1993-L2061】

- 其他检测脚本（`onnx_output_bottom_body_location.py`、`onnx_output_top_body_location.py`、`onnx_output_side_body_standoff_location.py`、`onnx_output_pin_yinXian_find_pitch.py`、`onnx_output_other_location.py` 等）通过各自的 `onnx_inference` + `begain_output_*` 组合产出不同视角的本体、引线、垫高与其它干扰目标坐标，为 F4.7-F4.9 阶段的匹配与参数求解提供输入。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_bottom_body_location.py†L325-L371】【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_pin_yinXian_find_pitch.py†L325-L451】【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_side_body_standoff_location.py†L326-L415】

## 3. F4.6-F4.9 流程对照
- **F4.6 序号/引脚定位**：调用 `onnx_output_serial_number_letter_location.py` 与 `onnx_output_pin_num4.py` 中的 `begain_output_*` 函数获取序号、PIN 框，再结合 `onnx_use.Run_onnx` 的 OCR 结果完成序号拆分。 【F:packagefiles/PackageExtract/onnx_use.py†L548-L569】【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_serial_number_letter_location.py†L418-L447】【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_pin_num4.py†L1058-L1264】
- **F4.7 标尺/文本匹配**：依赖 `onnx_QFP_pairs_data_location2.py` 的 `begain_output_pairs_data_location` 输出 pairs、角度、外框信息，为 `MPD_data`/`MPD` 提供几何约束。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_QFP_pairs_data_location2.py†L862-L906】
- **F4.8 数据整理**：同样使用 `output_pairs_data_location` 中整理后的标尺、引脚分类数据，为 `get_better_data_2`、`resize_data_2` 的最终筛洗提供输入。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_QFP_pairs_data_location2.py†L509-L860】
- **F4.9 参数汇总**：通过多视角检测脚本（顶/底/侧本体、引线间距等）的 `begain_output_*` 结果，配合 OCR 输出的文字内容，综合计算体尺寸、Pitch、Height 等参数。 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_top_body_location.py†L326-L461】【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_output_side_body_standoff_location.py†L326-L415】

以上函数均已补充中文注释，便于阅读与维护。
