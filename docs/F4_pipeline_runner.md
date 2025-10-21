# F4 流程测试脚本说明

为验证当前仓库在缺少 ONNX/YOLOX 模型与 OCR 运行环境时仍能跑通 F4.6-F4.9 的代码路径，新增了两个配套组件：

1. `packagefiles/PackageExtract/f4_pipeline_runner.py`
   - 将 `common_pipeline.py` 中拆分出的步骤函数重新串联，提供 `run_f4_pipeline` 入口。 
   - 输入为封装图片所在目录与封装类型，输出包含 `L3` 中间结构、参数候选列表以及 `nx`/`ny` 结果，便于主进程或脚本直接复用。

2. `tests/stubbed_dependencies.py` 与 `tests/test_f4_pipeline_runner.py`
   - 通过 `install_common_pipeline_stubs` 替换掉 DBNet、YOLOX、DETR 等推理依赖，模拟空结果并保留接口签名。
   - `tests/test_f4_pipeline_runner.py` 在临时目录生成空白视图图片，调用 `run_f4_pipeline`，确认整个流程不会抛出异常且返回结构满足预期。

若需要在本地进一步验证，可执行：

```bash
python -m unittest tests.test_f4_pipeline_runner
```

该测试无需真实模型文件，即可验证公共模块拆分后与主流程的接口兼容性。后续如需对接真实推理，只需移除或跳过桩模块注入步骤。 
