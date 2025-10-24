# BGA 公共函数前后对比

本文对比了重构前 `packagefiles/PackageExtract/BGA_extract_test.py` 中的关键函数与当前 `common_pipeline.py` 中的共享实现，确认公共模块导入的逻辑与旧版本保持一致。旧代码引用自 `docs/_bga_original_functions.md`，新代码取自最新的共享模块。

## 映射概览

| F4 步骤 | 旧函数 (`BGA_extract_test.py`) | 新函数 (`common_pipeline.py`) | 说明 |
| --- | --- | --- | --- |
| F4.6 | `data_delete_other` | `remove_other_annotations` | 仍调用 `_pairs_module.delete_other`，按视图更新 `yolox_num` 与 `dbnet_data`。 |
| F4.6 | `for_pairs_find_lines` | `enrich_pairs_with_lines` | 根据视图拼接图像路径后调用 `find_pairs_length`，写回 `*_yolox_pairs_length`。 |
| F4.7 | `resize_data_1` | `preprocess_pairs_and_text` | 直接转调 `get_better_data_1` 并更新缓存拷贝/全量文本。 |
| F4.7 | `SVTR_get_data` | `run_svtr_ocr` | 调用 `SVTR` 返回四视图 OCR 结果，写回 `*_ocr_data`。 |
| F4.7 | `get_max_medium_min` | `normalize_ocr_candidates` | 调用 `data_wrangling` 对 OCR 结果归一化。 |
| F4.8 | `get_Pin_data` + `find_BGA_PIN` 等 | `extract_pin_serials` | 兼容 QFP/BGA 的序号提取逻辑未变，仍使用 `find_PIN`、`find_BGA_PIN`、`find_pin_num_pin_1`。 |
| F4.8 | `MPD_data` | `match_pairs_with_text` | 调用 `MPD` 重新绑定尺寸线和文本。 |
| F4.8 | `resize_data_2` | `finalize_pairs` | 复用 `get_better_data_2` 生成最终 `yolox_pairs_*`。 |
| F4.9 | `find_QFP_parameter` | `compute_qfp_parameters` | 沿用 `get_serial`、`get_QFP_body`、`get_QFP_parameter_list` 等方法计算参数列表。 |

## 关键函数对比

### `data_delete_other` → `remove_other_annotations`

旧实现：

```python
# 摘自 docs/_bga_original_functions.md
def data_delete_other(L3):
    top_yolox_num = find_list(L3, 'top_yolox_num')
    top_dbnet_data = find_list(L3, 'top_dbnet_data')
    top_other = find_list(L3, 'top_other')
    ...
    top_yolox_num = delete_other(top_other, top_yolox_num)
    top_dbnet_data = delete_other(top_other, top_dbnet_data)
    ...
    recite_data(L3, 'top_yolox_num', top_yolox_num)
    recite_data(L3, 'top_dbnet_data', top_dbnet_data)
    ...
    return L3
```

新实现：

```python
# 摘自 common_pipeline.py
for view in ("top", "bottom", "side", "detailed"):
    yolox_key = f"{view}_yolox_num"
    dbnet_key = f"{view}_dbnet_data"
    other_key = f"{view}_other"

    yolox_num = find_list(L3, yolox_key)
    dbnet_data = find_list(L3, dbnet_key)
    other_data = find_list(L3, other_key)

    filtered_yolox = _pairs_module.delete_other(other_data, yolox_num)
    filtered_dbnet = _pairs_module.delete_other(other_data, dbnet_data)

    recite_data(L3, yolox_key, filtered_yolox)
    recite_data(L3, dbnet_key, filtered_dbnet)
```

差异说明：循环代替逐视图代码，但仍依赖 `delete_other` 并回写同样的键值，不影响输出。

### `for_pairs_find_lines` → `enrich_pairs_with_lines`

旧实现按视图构造 `DATA/<view>.jpg` 并调用 `find_pairs_length`。新函数将根目录作为参数传入，逻辑仅将硬编码路径改为 `os.path.join(image_root, f"{view}.jpg")`，其余保持一致，仍使用 `np.empty((0, 13))` 处理缺图场景。

### `resize_data_1` → `preprocess_pairs_and_text`

两者均调用 `_pairs_module.get_better_data_1`，并把返回的多个数组写回 `L3`。新版本仅改名并明确参数顺序，输出键名不变（`*_yolox_pairs_copy`、`*_dbnet_data_all` 等）。

### `SVTR_get_data` → `run_svtr_ocr`

两者都调用 `_pairs_module.SVTR`，返回顺序一致，继续通过 `recite_data` 写回四个视图的 OCR 结果。

### `get_max_medium_min` → `normalize_ocr_candidates`

新函数依旧调用 `_pairs_module.data_wrangling`，传入/回写的键完全一致（`*_ocr_data`）。

### `get_Pin_data` + BGA 特殊序号处理 → `extract_pin_serials`

- QFP/QFN 等封装：仍调用 `_pairs_module.find_PIN`，保持 `top_serial_numbers_data`、`bottom_serial_numbers_data` 的写法。
- BGA：保留 `_pairs_module.find_BGA_PIN` 和 `_pairs_module.find_pin_num_pin_1`，并像旧代码一样生成 `pin_num_x_serial`、`pin_num_y_serial`、`pin_1_location`。

因此不同封装场景的输出结构与旧逻辑一致。

### `MPD_data` → `match_pairs_with_text`

继续从 `L3` 中读取 `*_yolox_pairs`、`*_ocr_data`、`*_border` 等键调用 `_pairs_module.MPD`，再将返回值写回。函数体与旧代码只有变量命名差异。

### `resize_data_2` → `finalize_pairs`

仍调用 `_pairs_module.get_better_data_2` 并写回 `yolox_pairs_top` 等键，同时保留原有的调试打印（`print("***/数据整理结果/***")`）。

### `find_QFP_parameter` → `compute_qfp_parameters`

两者使用同一套 `_pairs_module` 方法 (`get_serial`、`get_QFP_body`、`get_QFP_parameter_list`、`get_QFP_high`、`get_QFP_pitch`、`resort_parameter_list_2`)。唯一变化是函数名与返回变量命名更清晰，返回值仍为 `(QFP_parameter_list, nx, ny)`。

## 结论

公共模块的实现直接复用了旧函数内部调用的 `_pairs_module`/`pairs_module` 方法，并保持相同的 `L3` 字段读写顺序。除去循环与路径处理等微调外，逻辑与旧版一致。上述对比可作为确认替换等价性的依据。若需回溯具体实现，可同时参考 `common_pipeline.py` 与 `docs/_bga_original_functions.md` 中的源码片段。
