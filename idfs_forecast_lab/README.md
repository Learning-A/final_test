# 负载预测工程

请先阅读仓库根 [README](../README.md)、[双机协作手册](../docs/COLLABORATION.md) 和 [当前状态](../docs/STATUS.md)。

本目录包含现场已有工程的源码、配置示例和测试。`config/site*.json` 已替换认证入口，实际凭据不在仓库。历史vendor和本机资产清单不在本次上传范围。

```powershell
python -m unittest discover -s tests -v
python -m idfs_lab.cli --help
```

`collect`、`gpu-submit`、`stage-inference`、`relay-live` 默认显示计划；加 `--execute` 会产生实际操作。`collect_workloads` 是直接执行20轮负载的入口，只由实验端在现有现场工程使用；它目前读取固定配置，没有恢复模式，不要在研发端运行。

`runs/`、`evidence/`、`models/`、`prepared/` 的真实内容由本机保管，不进Git。数据交接见 [接口文档](../docs/DATA_CONTRACT.md)。

工程 `docs/` 中四份旧阶段文档保留用于理解实现，内容有过期段落；当前规范以根docs为准。
