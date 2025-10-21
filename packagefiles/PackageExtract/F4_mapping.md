# F4.6-F4.9 流程与现有函数对应关系

本文档梳理流程图 F4.6-F4.9 阶段在代码中的具体实现位置，便于后续重构时精准定位到现有函数。

## F4.6：提取并拆分行列序号
- **生产脚本入口**：`get_Pin_data`（`packagefiles/PackageExtract/QFP_extract.py`）
  - QFP/QFN/SOP/SON 分支使用 `find_PIN` 将 `yolox_serial_num` 中的序号框与 OCR 结果匹配，并把识别到的序号从文本集合中剔除。
  - BGA 分支调用 `find_BGA_PIN`、`find_pin_num_pin_1` 推导行列标记及 PIN1 位置。
- **测试脚本映射**：`packagefiles/PackageExtract/get_pairs_data_present5_test.py`
  - `find_PIN`：与生产代码一致，负责将顶视、底视的序号框与 OCR 结果匹配。
  - `find_BGA_PIN`、`find_pin_num_pin_1`：用于 BGA 流程推导 PIN 布局。
  - `find_serial_number_letter_QFP`：在测试脚本中拆分字母/数字序号并回写至 OCR 字典。

## F4.7：匹配尺寸标尺与文本
- **生产脚本入口**：`MPD_data`（`packagefiles/PackageExtract/QFP_extract.py`）
  - 调用 `MPD`，利用已经筛选后的 pairs、角度标尺、外框等信息，对 OCR 文本进行再次配对整理，输出更新后的 `top/bottom/side/detailed_ocr_data`。
- **测试脚本映射**：`get_pairs_data_present5_test.py`
  - `MPD`：核心匹配函数，负责将标尺、角度线、引脚等要素与 OCR 文本绑定。
  - `match_pairs_data`、`match_pairs_data_angle`、`match_pairs_data_table`：实现按方向、角度与表格多种匹配策略。
  - `Divide_regions_pairs`：将标尺线按空间区域划分，为匹配提供基础分组。

## F4.8：整理匹配结果
- **生产脚本入口**：`resize_data_2`（`packagefiles/PackageExtract/QFP_extract.py`）
  - 基于 `get_better_data_2`，结合 pairs 的长度信息，对 OCR 文本和 pairs 进行最终筛选与拷贝，得到 `yolox_pairs_*` 与整理后的 OCR 结果。
- **测试脚本映射**：`get_pairs_data_present5_test.py`
  - `get_better_data_2`：以匹配关系为基础，剔除噪声并同步更新 OCR 数据。
  - `copy_pairs_data`、`copy_ocr_data`：在测试流程中完成匹配结果的深拷贝与缓存。
  - `resize_data`、`resize_data_table`：对 pairs 与表格数据做最终格式统一。

## F4.9：汇总封装参数
- **生产脚本入口**：`find_QFP_parameter`（`packagefiles/PackageExtract/QFP_extract.py`）
  - `get_serial` 负责推算 nx/ny；`get_QFP_body` 计算 body_x/body_y。
  - `get_QFP_parameter_list` 生成初始参数候选，随后 `resort_parameter_list_2`、`get_QFP_high`、`get_QFP_pitch` 等函数完善 pitch、高度等参数，形成最终 `QFP_parameter_list`。
- **测试脚本映射**：`get_pairs_data_present5_test.py`
  - `get_QFP_body`：根据顶/底视标尺结果估算封装本体尺寸。
  - `get_serial`、`get_serial_QFP`：测试流程中推算 nx/ny 的辅助函数。
  - `get_QFP_parameter_list`、`resort_parameter_list_2`、`get_QFP_high`、`get_QFP_pitch`：复用生产逻辑整理参数候选。
  - `Completion_QFP_parameter_list`、`output_QFP_parameter`：补全缺失字段并输出最终参数表。

以上均为项目现存函数，可直接复用以支撑 F4.6-F4.9 的重构工作。
