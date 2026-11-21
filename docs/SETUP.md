# 环境与信息获取

## 本地配置

```powershell
git clone https://github.com/Learning-A/final_test.git
cd final_test
New-Item -ItemType Directory -Force .local
```

在 `.local/ROLE.md` 写“实验端A”或“研发端B”、本地数据根目录和最近任务ID；此文件不提交。两台电脑可有不同绝对路径，交接以数据集ID、相对路径和SHA256为准。

推荐Python3.10/3.11隔离环境。核心处理只需标准库；SSH安装Paramiko；神经前向测试按本机环境安装匹配PyTorch。训练依赖由GPU任务使用已有环境，不自动升级或安装远端软件。用 `pip install -e ./idfs_forecast_lab` 可安装工程CLI，也可直接进入工程目录 `python -m idfs_lab.cli --help`。

## 认证与现场配置

仓库 `config/site*.json` 保留拓扑但将跳板和凭据路径替换为占位值。`config/training_host.json` 的代理地址为占位值。它们用于离线测试，不是生产连接配置。

A将本机已有现场配置放在仓库内 `.local/`（或继续使用原工程配置），CLI通过 `--site <私有文件>`、`--training-host <私有文件>` 传入；客户端密码通过 `IDFS_CLIENT_PASSWORD` 环境变量设置。不把值写进命令历史、提交、issue或日志。`collect_workloads` 仍读取工作目录下固定 `config/site_smoke.json`，因此目前只在A的原现场目录运行；移植配置参数前不要在仓库示例配置上启动。GPU训练仍检查用户名、主机和目录。

B无需现场凭据。若需要更改现场行为，通过候选配置和issue交给A；不要复制整个本机remote_idfs目录给B，也不要把远端凭据写入共享配置。

## 自动测试状态

当前仓库没有启用GitHub Actions，本机令牌缺少workflow权限。`docs/templates/github-actions-tests.yml` 只是待安装模板，不会自动执行。两台电脑提交前各自跑本地检查；不要将模板存在写成CI通过。

## 查找信息的顺序

| 想知道什么 | 先看哪里 |
|---|---|
| 当前完成了什么、是否能继续采集 | `docs/STATUS.md`；A再查现场 |
| 每台电脑负责什么 | `docs/COLLABORATION.md` |
| 数据是否已采、能否训练、去哪里取 | `data_catalog/datasets.json` |
| 特征/标签和模型输出格式 | `docs/DATA_CONTRACT.md` 与 joint.py/runtime.py |
| 最近谁改了什么 | `docs/CHANGELOG.md`、git log和对应PR |
| 哪次训练在运行 | 对应训练请求issue和A登记的job_id |
| 原始证据和结果 | 数据条目的保管端、相对路径与SHA256 |

`reports/` 是摘要，不包含全部原始证据；`docs/history/` 是早期设计和旧实验描述，不能证明当前集群精度。没有数据包或权重时应报告缺失，不生成替代数据伪装现场结果。
