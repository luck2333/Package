# `common_pipeline.py` 公共函数说明

## 重构背景
在最初的实现中，`QFP_extract.py` 与 `BGA_extract_test.py` 内部分别维护了一套完全相同的**工作目录初始化逻辑**与**YOLO/DBNet 推理封装**。随着不同封装类型的脚本逐渐增多，复制粘贴的代码不仅难以维护，也增加了修改时遗漏的风险。因此本次重构把这些高频调用的逻辑统一迁移到 `common_pipeline.py`，供多个提取脚本共享。

## 原始重复代码示例
以下片段摘自 `7177589` 版本的仓库，可以看到 QFP/BGA 两个脚本中的实现几乎一致：

```python
# QFP_extract.py 中的 front_loading_work（节选）
def front_loading_work():
    empty_folder(ONNX_OUTPUT)
    os.makedirs(ONNX_OUTPUT)
    empty_folder(DATA_BOTTOM_CROP)
    os.makedirs(DATA_BOTTOM_CROP)
    for file_name in ("top", "bottom", "side", "detailed"):
        filein = f"{DATA}/{file_name}.jpg"
        fileout = filein
        try:
            set_Image_size(filein, fileout)
        except:
            print('文件', filein, '不存在')
    empty_folder(DATA_COPY)
    os.makedirs(DATA_COPY)
    for file_name in os.listdir(DATA):
        shutil.copy(f"{DATA}/{file_name}", f"{DATA_COPY}/{file_name}")
    empty_folder(OPENCV_OUTPUT)
    os.makedirs(OPENCV_OUTPUT)
    empty_folder(DATA)
    os.makedirs(DATA)
    for file_name in os.listdir(DATA_COPY):
        shutil.copy(f"{DATA_COPY}/{file_name}", f"{DATA}/{file_name}")
```

```python
# BGA_extract_test.py 中的 get_data_location_by_yolo_dbnet（节选）
def get_data_location_by_yolo_dbnet(package_path, package_classes):
    L3 = []
    empty_data = np.empty((0, 4))
    img_path = f'{package_path}/top.jpg'
    if not os.path.exists(img_path):
        top_dbnet_data = empty_data
        top_yolox_pairs = empty_data
        ...
    else:
        top_dbnet_data = dbnet_get_text_box(img_path)
        top_yolox_pairs, top_yolox_num, ... = yolo_classify(img_path, package_classes)
    # bottom/side/detailed 逻辑与上述完全相同，仅文件名不同
    L3.append({'list_name': 'top_dbnet_data', 'list': top_dbnet_data})
    L3.append({'list_name': 'top_yolox_pairs', 'list': top_yolox_pairs})
    ...
    return L3
```

当其他提取脚本需要同样的准备/检测流程时，只能继续复制粘贴上述代码，维护成本很高。

## 公共函数拆分与中文注释
现在所有共用逻辑收敛在 `common_pipeline.py` 中，并补充了中文注释：

- `prepare_workspace`：完整封装旧版 `front_loading_work` 的步骤，负责清理目录、统一视图尺寸并回写备份；
- `dbnet_get_text_box`：运行 DBNet 模型，返回文本框坐标；
- `yolo_classify`：统一的 YOLO 推理入口，BGA 会额外融合 DETR 输出；
- `get_data_location_by_yolo_dbnet`：遍历视图，组合 YOLO + DBNet 的检测结果，输出与旧版完全一致的 `L3` 数据结构。

上述函数内部均补充了中文说明，方便团队成员快速理解每一步的作用。

## 现有脚本的调用方式
`QFP_extract.py` 与 `BGA_extract_test.py` 已经切换为：

```python
from packagefiles.PackageExtract.common_pipeline import (
    get_data_location_by_yolo_dbnet,
    prepare_workspace,
)
```

对应地：

- 初始化阶段调用 `prepare_workspace(...)`；
- 获取检测结果时调用 `get_data_location_by_yolo_dbnet(...)`；
- 原脚本中其余业务逻辑保持不变，无需再维护各自的初始化/推理函数。

借助这次抽取，后续若需要调整目录结构或替换检测模型，只需在 `common_pipeline.py` 做一次修改即可被所有封装流程共享。
