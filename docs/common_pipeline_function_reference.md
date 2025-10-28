# `common_pipeline.py` 函数维护手册

本文档用于维护 `packagefiles/PackageExtract/common_pipeline.py` 中的公用函数，梳理它们在封装提取主流程（流程图 F4.6–F4.9）中的职责、输入输出含义以及对其他模块的依赖。阅读本手册可以快速了解各函数在项目中的定位，避免未来重构时破坏调用契约。

## 数据结构与约定

- **视图命名**：模块通过 `DEFAULT_VIEWS = ("top", "bottom", "side", "detailed")` 约定四个视图名称，并据此组织输入图像、检测结果与 OCR 结果。
- **路径常量**：文件顶部的 `DATA`、`DATA_COPY`、`ONNX_OUTPUT` 等常量保留了旧脚本的目录结构，确保在不同 IDE 或工作目录下也能定位到运行时产物。
- **L3 容器**：绝大部分函数围绕名为 `L3` 的列表进行读写。它是若干 `{"list_name": <str>, "list": <array/list>}` 字典的集合，用来存储不同阶段的检测/OCR/配对结果。诸如 `find_list`、`recite_data` 等工具函数负责从 L3 中取出或更新指定名称的数据。

## 函数索引

下表给出每个公开函数的概览，后续章节会详细描述其输入输出。

| 函数 | 流程阶段 | 主要职责 |
| --- | --- | --- |
| `prepare_workspace` | 预处理 | 清空工作目录、调整图片尺寸、重建数据副本 |
| `dbnet_get_text_box` | F4.6 | 调用 DBNet 推理并返回文本框坐标 |
| `yolo_classify` | F4.6 | 调用 YOLOX/DETR，汇总尺寸线、数字、PIN 等检测框 |
| `get_data_location_by_yolo_dbnet` | F4.6 | 汇集指定视图的 DBNet/YOLO 结果并写入 L3 |
| `remove_other_annotations` | F4.6 | 使用 `delete_other` 去除 OTHER 类框 |
| `enrich_pairs_with_lines` | F4.6 | 基于 LSD 结果计算尺寸线的标尺长度 |
| `preprocess_pairs_and_text` | F4.7 | 过滤并复制初始尺寸线/文本，准备 OCR 输入 |
| `run_svtr_ocr` | F4.7 | 调用 SVTR OCR 得到四视图的文本候选 |
| `normalize_ocr_candidates` | F4.7 | 对 OCR 结果做数字规整与场景分类 |
| `extract_pin_serials` | F4.8 | 处理数字/字母序号，补充 PIN 元数据（含 BGA 特例） |
| `match_pairs_with_text` | F4.8 | 将尺寸线与文本重新配对，产出最终绑定结果 |
| `finalize_pairs` | F4.8 | 清洗配对结果，输出最终可用的尺寸线集合 |
| `compute_qfp_parameters` | F4.9 | 计算封装参数列表，并推断行列数 (nx/ny) |

## 详细说明

### `prepare_workspace`
- **阶段**：预处理（流程图 F4 之前）。
- **参数**：
  - `data_dir`：原始图片所在目录，函数会在此目录中读取 `top.jpg` 等视图文件。
  - `data_copy_dir`：备份目录，保存原图副本以便后续还原。
  - `data_bottom_crop_dir`：底视裁剪结果目录，保持与旧版流程兼容。
  - `onnx_output_dir`：存放 ONNX 推理中间结果的目录。
  - `opencv_output_dir`：存放 OpenCV/LSD 等调试图的目录。
  - `image_views`：需要处理的视图名称迭代器，默认值为 `DEFAULT_VIEWS`。
- **逻辑**：逐一清空上述目录，调用 `set_Image_size` 调整图片尺寸，将原始图片复制到备份目录，再用备份数据回填 `data` 目录，确保后续步骤在干净的输入上运行。
- **返回值**：无。副作用是若干目录被清空重建，原始图片尺寸被标准化。

### `dbnet_get_text_box`
- **阶段**：F4.6 文本框检测。
- **参数**：`img_path` 表示当前视图的绝对路径。
- **逻辑**：调用 `Run_onnx_det(img_path)` 获得 DBNet 原始四点坐标，将其转换成左上、右下形式的矩形框，并保留两位小数。
- **返回值**：`np.ndarray`，形状为 `(N, 4)`，按顺序存放 `[x_min, y_min, x_max, y_max]`。

### `yolo_classify`
- **阶段**：F4.6 尺寸线/标注检测。
- **参数**：
  - `img_path`：视图图片路径。
  - `package_classes`：封装类型字符串（如 `"BGA"`、`"QFP"`）。
- **逻辑**：调用 `begain_output_pairs_data_location` 获取 YOLOX 输出，若封装类型为 BGA，再结合 `DETR_BGA` 的输出增强 PIN 与外框检测。非 BGA 分支会对尺寸线与角度框做四舍五入，兼容旧逻辑。
- **返回值**：长度为 10 的元组，依次为尺寸线、数字框、序号框、PIN、OTHER、PAD、外框、角度线、BGA 数字序号、BGA 字母序号，格式与旧脚本一致。

### `get_data_location_by_yolo_dbnet`
- **阶段**：F4.6 数据汇总。
- **参数**：
  - `package_path`：包含四视图图片的目录。
  - `package_classes`：封装类型。
  - `view_names`：需要处理的视图列表，默认使用 `DEFAULT_VIEWS`。
- **逻辑**：逐个视图执行 `dbnet_get_text_box` 与 `yolo_classify`，将返回结果整理成 `{"list_name": f"{view}_{key}", "list": array}` 的字典，并写入新的 L3 列表。如果底视数据存在，还会额外添加 `bottom_BGA_serial_num` 与 `bottom_BGA_serial_letter`。
- **返回值**：初始化好的 L3 列表，供后续阶段逐步补充信息。

### `remove_other_annotations`
- **阶段**：F4.6 数据清洗。
- **参数**：`L3`——上一阶段生成的列表容器。
- **逻辑**：对四个视图循环执行 `delete_other`，剔除 `OTHER` 类框带来的干扰，并通过 `recite_data` 回写 L3。
- **返回值**：更新后的 L3，仍指向原对象。

### `enrich_pairs_with_lines`
- **阶段**：F4.6 尺寸线补全。
- **参数**：
  - `L3`：当前数据容器。
  - `image_root`：视图图片所在目录。
  - `test_mode`：测试模式标志，直接传递给 `_pairs_module.find_pairs_length` 控制可视化等行为。
- **逻辑**：读取 `*_yolox_pairs`，调用 `find_pairs_length` 计算每条尺寸线对应的标尺长度，并将结果写入 `*_yolox_pairs_length`。
- **返回值**：更新后的 L3。

### `preprocess_pairs_and_text`
- **阶段**：F4.7 初步结构化。
- **参数**：
  - `L3`：当前数据容器。
  - `key`：测试/调试开关，透传至 `_pairs_module.get_better_data_1`，控制可视化与日志输出。
- **逻辑**：读取四视图的尺寸线与 DBNet 文本框，调用 `get_better_data_1` 完成第一次过滤与备份，函数会返回处理后的尺寸线、尺寸线副本以及扩展的 DBNet 数据。随后将这些结果回写到 L3，供 OCR 与后续绑定使用。
- **返回值**：更新后的 L3。

### `run_svtr_ocr`
- **阶段**：F4.7 OCR 推理。
- **参数**：`L3`。
- **逻辑**：从 L3 中取出 `top_dbnet_data_all`、`bottom_dbnet_data_all`、`side_dbnet_data`、`detailed_dbnet_data`，调用 `_pairs_module.SVTR` 执行 ONNX OCR，并将返回的四视图文本结果写回 L3。
- **返回值**：更新后的 L3。

### `normalize_ocr_candidates`
- **阶段**：F4.7 OCR 后处理。
- **参数**：
  - `L3`：当前数据容器。
  - `key`：调试开关，透传给 `_pairs_module.data_wrangling`。
- **逻辑**：对 OCR 原始结果进行格式化（数值补零、inch 转换、正负号识别等），并结合尺寸线数字框信息更新 `max_medium_min`、`Absolutely` 等字段。
- **返回值**：更新后的 L3。

### `extract_pin_serials`
- **阶段**：F4.8 PIN/序号处理。
- **参数**：
  - `L3`：当前数据容器。
  - `package_classes`：封装类型。
- **逻辑**：
  - **引脚式封装（QFP/QFN/SOP/SON）**：调用 `_pairs_module.find_PIN`，返回上下视图的序号矩阵并回写。
  - **BGA 封装**：
    1. 读取 `bottom_BGA_serial_num/letter`，通过 `_pairs_module.find_BGA_PIN` 过滤候选框，并同步更新底视 OCR 结果。
    2. 使用 `_flatten_key_info` 和 `_build_serial_matrix` 将嵌套 `key_info` 展平为 `np.ndarray`，作为 `find_pin_num_pin_1` 的输入以推断行列数和 Pin1 位置。
    3. 尝试导入 `extract_BGA_PIN`（位于 `get_pairs_data_present5.py`），若可用则使用其检测结果补充/覆盖行列数，并记录缺失 Pin 或颜色异常信息。
    4. 将处理后的底视序号、OCR 结果和 `pin_num_x_serial/pin_num_y_serial/pin_1_location` 回写到 L3。
- **返回值**：更新后的 L3。

### `match_pairs_with_text`
- **阶段**：F4.8 文本与尺寸线绑定。
- **参数**：
  - `L3`：当前数据容器。
  - `key`：调试开关，透传给 `_pairs_module.MPD`。
- **逻辑**：收集四视图的尺寸线、角度线、外框和 OCR 结果，调用 `MPD` 完成多轮匹配，得到绑定后的文本列表。
- **返回值**：更新后的 L3。

### `finalize_pairs`
- **阶段**：F4.8 终态整理。
- **参数**：`L3`。
- **逻辑**：调用 `_pairs_module.get_better_data_2` 对绑定结果做最后的筛选（包括同类候选合并、缺失补偿等），并生成 `yolox_pairs_top/bottom/side/detailed` 四个最终的尺寸线集合。函数还会打印调试信息帮助人工核查。
- **返回值**：更新后的 L3。

### `compute_qfp_parameters`
- **阶段**：F4.9 参数组装。
- **参数**：`L3`。
- **逻辑**：
  1. 通过 `_pairs_module.get_serial` 读取上下视图的序号信息，得到候选的 `nx/ny`。
  2. 调用 `_pairs_module.get_QFP_body` 结合外框和尺寸线计算 `body_x/body_y`。
  3. 使用 `_pairs_module.get_QFP_parameter_list` 汇总四视图的文本候选生成封装参数列表，并多次调用 `resort_parameter_list_2` 保证输出顺序与权重符合旧逻辑。
  4. 若高宽、pitch 等参数存在多候选，则进一步调用 `get_QFP_high`、`get_QFP_pitch` 做二次筛选。
- **返回值**：三元组 `(QFP_parameter_list, nx, ny)`，其中 `QFP_parameter_list` 与旧脚本一致，包含每个参数的候选值、来源视图、可信度等信息，`nx/ny` 是行列数候选。

## 内部辅助函数

`extract_pin_serials` 内部声明的 `_flatten_key_info` 与 `_build_serial_matrix` 仅在 BGA 分支使用，用于处理 `find_BGA_PIN` 返回的嵌套结构。它们不会在模块外部暴露，若未来将 PIN 逻辑独立成公共函数，可把这两个辅助方法迁移到专用的工具模块中。

## 依赖模块速查

- `_pairs_module`（`get_pairs_data_present5_test.py`）：提供尺寸线匹配、OCR 后处理、序号提取、参数组装等核心算法，实现了旧流程的绝大多数业务逻辑。
- `function_tool`：封装了 `find_list`、`recite_data`、`empty_folder`、`set_Image_size` 等基础工具，用于管理 L3 与临时目录。
- `Run_onnx_det`：DBNet 推理入口，位于 `onnx_use.py`。
- `begain_output_pairs_data_location` / `DETR_BGA`：YOLOX 与 DETR 推理入口，用于生成几何框和 PIN 序号。
- `extract_BGA_PIN`：原 BGA 流水线的 PIN 提取函数，当前仍保留在 `get_pairs_data_present5.py` 中，公共模块通过 try/except 方式兼容其存在与否。

## 维护建议

1. **保持输入输出契约**：任何修改都应确保函数返回值形态、L3 中的键名与旧流程一致，否则会影响多个封装脚本及单元测试。
2. **新增功能优先扩展 `_pairs_module`**：当需要调整尺寸线/文本匹配策略时，请在 `_pairs_module` 内新增或修改实现，再由公共模块调用，避免在 `common_pipeline.py` 中堆砌业务细节。
3. **引脚与参数逻辑慎重改动**：`extract_pin_serials` 与 `compute_qfp_parameters` 直接影响最终的封装参数输出。修改前请参考 `docs/BGA_pin_function_catalog.md` 与 `docs/BGA_shared_function_comparison.md`，确认不会破坏现有的 BGA/QFP 数据路径。

