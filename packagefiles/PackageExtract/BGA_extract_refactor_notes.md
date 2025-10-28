# `BGA_extract_test.py` 重构说明

本记录梳理 `BGA_extract_test.py` 在本次重构前后的差异，方便团队成员回溯旧实现并理解公共模块的复用方式。

## 1. 原始函数列表（重构前）

下表列出了旧版脚本内的全部方法及其对应的流程节点；详细源码已备份于 `docs/_bga_original_functions.md`，可供进一步查阅。

| 函数名 | 职责概述 | 流程图阶段 |
| --- | --- | --- |
| `extract_package` | 驱动 BGA 全流程，串联预处理、检测、OCR、配对与参数输出 | F4.6–F4.9 全部 |
| `time_save_find_pinmap` / `long_running_task` | 在后台线程调用 `find_pin` 并生成 `pin_num.txt` | 预处理准备 |
| `data_delete_other` | 从四个视图的 YOLO/DBNet 结果中剔除 OTHER 框 | F4.6 |
| `for_pairs_find_lines` | 基于 LSD 结果补齐尺寸线两端的标尺 | F4.6 |
| `resize_data_1` | 组合尺寸线与文本，形成初始候选 | F4.7 |
| `SVTR_get_data` | 调用 SVTR OCR 获取文本内容 | F4.7 |
| `get_max_medium_min` | OCR 候选清洗，保留最大/中值/最小组合 | F4.7 |
| `get_Pin_data` | 提取序号/PIN 信息（BGA/QFP 混合实现） | F4.8 |
| `MPD_data` | 匹配尺寸线与文本，完成语义绑定 | F4.8 |
| `resize_data_2` | 清洗配对结果并打印调试信息 | F4.8 |
| `find_QFP_parameter` | 依据配对结果计算 body/pitch/high 等参数 | F4.9 |

## 2. 公共模块复用与替换关系

为减少重复代码，本次重构将 F4.6–F4.9 各阶段的核心逻辑迁移至 `common_pipeline.py`，并在其中新增以下公共方法：

| 新公共函数 | 复用自旧版函数 | 作用说明 |
| --- | --- | --- |
| `remove_other_annotations` | `data_delete_other` | 统一剔除 OTHER 框 |
| `enrich_pairs_with_lines` | `for_pairs_find_lines` | 计算尺寸线与标尺边界 |
| `preprocess_pairs_and_text` | `resize_data_1` | 整理 YOLO/DBNet 初始候选 |
| `run_svtr_ocr` | `SVTR_get_data` | 执行 SVTR OCR |
| `normalize_ocr_candidates` | `get_max_medium_min` | OCR 文本清洗 |
| `extract_pin_serials` | `get_Pin_data` | 处理序号、PIN 及 BGA 特殊分支 |
| `match_pairs_with_text` | `MPD_data` | 重新配对尺寸线与文本 |
| `finalize_pairs` | `resize_data_2` | 输出最终配对结果 |
| `compute_qfp_parameters` | `find_QFP_parameter` | 计算 body/pitch/high 等参数 |

`BGA_extract_test.py` 现仅保留流程编排：

1. 调用 `prepare_workspace` 完成预处理。
2. 依次执行公共模块中的九个步骤，覆盖 F4.6–F4.9。
3. 通过 `get_QFP_parameter_data` 与 `alter_QFP_parameter_data` 得到最终参数列表。
4. 保留 `time_save_find_pinmap` 及 `long_running_task` 作为 BGA 专属的 PinMap 生成入口。

## 3. 当前脚本骨架

新版脚本核心仅包含三类逻辑：

- **导入公共函数**：复用 `common_pipeline` 的各个步骤函数。
- **流程驱动**：`extract_package` 中按流程图顺序执行公共函数。
- **PinMap 线程**：`time_save_find_pinmap`/`long_running_task` 负责异步生成 `pin_num.txt`。

这样一来，BGA 与 QFP 提取脚本可以共享同一套 F4.6–F4.9 实现，为后续进一步重构（例如让 QFP 同步复用公共模块）打下基础。

