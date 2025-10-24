# BGA PIN 处理函数速查

本文汇总当前代码库中与 **BGA 封装 PIN 识别、行列推断与缺 PIN 判断** 相关的核心函数，说明它们的职责、依赖关系以及在整条 F4.6–F4.9 流程中的调用顺序，方便后续挑选与复用。

## 1. 检测阶段：生成序号与 PIN 候选

| 入口 | 位置 | 作用 | 关键依赖 |
| --- | --- | --- | --- |
| `get_data_location_by_yolo_dbnet` | `packagefiles/PackageExtract/common_pipeline.py` | 对四个视图执行 DBNet + YOLO 推理；在 BGA 分支额外汇总数字/字母序号检测结果，写入 `bottom_BGA_serial_num/bottom_BGA_serial_letter`。 | `dbnet_get_text_box`、`yolo_classify` 【F:packagefiles/PackageExtract/common_pipeline.py†L196-L242】|
| `yolo_classify` | 同上 | 调用通用 YOLOX 推理；BGA 模式下再融合 DETR 推理，确保序号与外框检测更稳定。 | `begain_output_pairs_data_location`、`DETR_BGA` 【F:packagefiles/PackageExtract/common_pipeline.py†L104-L152】|
| `begain_output_pairs_data_location` | `packagefiles/PackageExtract/yolox_onnx_py/onnx_QFP_pairs_data_location2.py` | 载入 BGA 专用 YOLOX 权重，输出标尺、数字/字母序号、PIN、外框等检测框。 | `onnx_inference`、`yolo_model_path` 等公共推理工具 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_QFP_pairs_data_location2.py†L635-L682】|
| `DETR_BGA` | `packagefiles/PackageExtract/DETR_BGA.py` | 使用 RT-DETR ONNX 强化底视 PIN、外框检测，并将结果与 YOLOX 汇合。 | ONNXRuntime session、`yolo_model_path` 【F:packagefiles/PackageExtract/DETR_BGA.py†L1-L200】|
| `onnx_output_pairs_data_pin_5` | `packagefiles/PackageExtract/yolox_onnx_py/onnx_detect_pin.py` | 单独推理 PIN 目标，供 pinmap/缺 PIN 检测流程使用。 | `yolox_onnx_shared` 公共预处理、NMS、可视化工具 【F:packagefiles/PackageExtract/yolox_onnx_py/onnx_detect_pin.py†L1-L116】|

这些函数会把底视序号框、PIN 检测框、边框等信息写入 L3 列表，后续步骤均基于此数据进行清洗与组合。

## 2. OCR 与序号提取阶段

| 函数 | 位置 | 职责 | 关键依赖 |
| --- | --- | --- | --- |
| `find_serial_number_letter` | `packagefiles/PackageExtract/get_pairs_data_present5.py` | 根据 YOLOX 检出的序号框，在底视 DBNet 文本框中筛出对应的数字/字母候选，为后续 OCR 绑定做准备。 | 序号几何匹配、矩形重叠度阈值计算 【F:packagefiles/PackageExtract/get_pairs_data_present5.py†L4534-L4617】|
| `ocr_data` | 同上 | 统一调度 OCR 推理（SVTR/ONNX），给 DBNet 框填充文本与 `key_info`。 | `ocr_get_data_onnx` OCR 推理封装 【F:packagefiles/PackageExtract/get_pairs_data_present5.py†L2692-L2696】|
| `filter_bottom_ocr_data` | 同上 | 将 OCR 文本与序号框重新绑定，生成 `serial_numbers_data`/`serial_letters_data`（含坐标+字符串）。 | 序号矩阵拼接、底视文本筛选 【F:packagefiles/PackageExtract/get_pairs_data_present5.py†L4621-L4649】|
| `find_BGA_PIN` | `packagefiles/PackageExtract/get_pairs_data_present5_test.py`（被公共模块复用） | 在底视 OCR 结果中剔除非序号文本，统计合法的数字、字母列表，初步确定行列数量趋势。 | OCR 过滤、字母/数字合法性判断 【F:packagefiles/PackageExtract/get_pairs_data_present5_test.py†L658-L744】|

公共流程中的 `extract_pin_serials` 会调用 `find_BGA_PIN`，并将嵌套的 `key_info` 扁平化为序号字符串矩阵。【F:packagefiles/PackageExtract/common_pipeline.py†L406-L510】

## 3. 行列数与 Pin1 位置推断

| 函数 | 位置 | 说明 | 依赖 |
| --- | --- | --- | --- |
| `find_pin_num_pin_1` | `packagefiles/PackageExtract/get_pairs_data_present5.py` | 依据数字/字母序列的最大连续值推断 `pin_num_x_serial/pin_num_y_serial`，并通过序号排序判断 Pin1 角位置。 | 字符串→数字转换、序列排序、方向判断 【F:packagefiles/PackageExtract/get_pairs_data_present5.py†L7003-L7174】|
| `extract_pin_serials` | `packagefiles/PackageExtract/common_pipeline.py` | 汇总上一步结果并写回 L3；在 BGA 分支中若检测到完整数据，还尝试调用完整版 `extract_BGA_PIN` 获取缺 PIN 信息。 | `find_BGA_PIN`、`find_pin_num_pin_1`、`extract_BGA_PIN`（可选）【F:packagefiles/PackageExtract/common_pipeline.py†L406-L510】|

若需要缺 PIN 或颜色信息，需进一步走完整的 BGA PIN 子流程。

## 4. BGA 专用 Pinmap 流程（可选）

| 函数 | 位置 | 职能 | 依赖 |
| --- | --- | --- | --- |
| `extract_BGA_PIN` | `packagefiles/PackageExtract/get_pairs_data_present5.py` | 独立入口：重新拷贝底视图，执行 DBNet/YOLO、pinmap 识别、OCR 清洗，最终返回行列数与缺 PIN 列表。 | `dbnet_get_text_box`、`yolo_classify`、`time_save_find_pinmap`、`find_serial_number_letter`、`ocr_data`、`filter_bottom_ocr_data`、`find_pin_num_pin_1`、`Is_Loss_Pin` 等 【F:packagefiles/PackageExtract/get_pairs_data_present5.py†L8200-L8258】|
| `time_save_find_pinmap` / `long_running_task` | 同上 | 启动子线程运行 pinmap 识别，兼容超时降级；结果拆分为 pin 占位矩阵与颜色矩阵。 | `find_pin`、`queue.Queue`、`threading` 【F:packagefiles/PackageExtract/get_pairs_data_present5.py†L4905-L4961】|
| `find_pin` | `packagefiles/PackageExtract/BGA_cal_pin.py` | 结合 YOLOX pin 检测与外框定位生成 pinmap，并把行列数写入缓存。 | `yolox_find_waikuang`、`find_pin_core`、`onnx_output_pairs_data_pin_5` 等 【F:packagefiles/PackageExtract/BGA_cal_pin.py†L2011-L2029】|
| `Is_Loss_Pin` | `packagefiles/PackageExtract/BGA_extract_old.py` | 根据 pinmap、Pin1 位置和颜色矩阵生成缺 PIN 文本标签。 | 字母表映射、颜色矩阵遍历 【F:packagefiles/PackageExtract/BGA_extract_old.py†L527-L599】|

该分支通常在主流程外独立调用；公共模块会在捕获到返回值后，向 L3 记录 `loss_pin`、`loss_color` 等额外信息，供后续输出使用。【F:packagefiles/PackageExtract/common_pipeline.py†L486-L504】

## 5. 工作流概览

1. **检测汇总**：`get_data_location_by_yolo_dbnet` → `yolo_classify`/`DETR_BGA` 获得序号框、PIN 框等原始数据。
2. **OCR 清洗**：`find_serial_number_letter` + `ocr_data` + `filter_bottom_ocr_data` 生成带文本的序号矩阵。
3. **序号推断**：`find_BGA_PIN` 过滤无效标注；`find_pin_num_pin_1` 输出行列数与 Pin1 角信息；`extract_pin_serials` 写回共享缓存。
4. **可选 pinmap 扩展**：如需缺 PIN / pinmap，可调用 `extract_BGA_PIN`，内部串联 `time_save_find_pinmap` → `find_pin` → `Is_Loss_Pin` 等模块，得到更丰富的诊断结果。

通过以上分层，既可以在公共流程里快速得到 BGA 的行列数，也能在需要时跳转到完整版 pinmap 分析，全部函数互相独立又保持数据结构兼容，方便按需组合。
