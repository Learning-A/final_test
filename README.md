# 白泽 IDFS 负载预测协作项目

预测 8 个物理存储节点未来 1 秒的读写总 IOPS；每个节点由所属 12 个 OSS 聚合。96 个 OSS 和 16 个客户端每 100 ms 采样，客户端作为辅助观测。训练在专用 GPU 服务器进行，集群运行 CPU 推理。

本仓库供两台电脑上的 Codex 协作。代码可用于开发，当前模型未达到准确率指标，正式验收尚未开始。

## 从哪里开始

1. 阅读 [当前状态](docs/STATUS.md)，确认哪些工作已完成、哪些仍待验证。
2. 阅读 [双机协作手册](docs/COLLABORATION.md)，确定本机角色。
3. 按 [环境与信息获取](docs/SETUP.md) 配置本机；默认配置不能连接现场。
4. 数据交接看 [数据与模型接口](docs/DATA_CONTRACT.md)，更新看 [更新记录](docs/CHANGELOG.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `idfs_forecast_lab/idfs_lab/` | 采样、聚合、特征、训练、CPU 推理、复算 |
| `idfs_forecast_lab/tests/` | 本地功能和边界测试 |
| `idfs_forecast_lab/config/` | 协议、拓扑、示例任务；连接配置已去掉实际认证入口 |
| `docs/` | 当前协作规范、状态、交接约定 |
| `reports/` | 已复算的小型结果摘要，非原始数据集 |
| `data_catalog/` | 数据集位置和可用性；没有自动下载地址 |
| `scripts/` | 生成不含认证配置的数据交接包 |
| `docs/history/` | 早期大纲和能力描述，不能作为当前项目达标证据 |

## 本地检查

Python 3.10+ 为推荐开发版本。进入工程目录执行：

```powershell
cd idfs_forecast_lab
python -m unittest discover -s tests -v
python -m idfs_lab.cli --help
```

标准库可运行核心处理；SSH 需要 Paramiko，神经模型测试需要 PyTorch。没有 PyTorch 时部分测试会跳过，报告时需披露。导入或测试模型不等于获准在本机训练。

原始采集数据、权重、凭据、现场运行目录不进 Git。当前 20 轮开发采集没有完成；不要因为仓库中存在计划就报告数据已到位。完整限制见根目录 [AGENTS.md](AGENTS.md)。
