# 模型端可立即使用的一轮带负载数据

2026-11-22用户明确授权：先用已有一轮带负载数据优化模型。该轮受OSS故障影响，允许探索、拟合方案设计和误差分析；原有故障标记保留，不计正常负载训练轮，也不能用于正式准确率验收。实际模型拟合仍由A在约定GPU服务器执行，B在本机分析数据、开发模型、回放推理并提交训练请求。

## 下载和校验

[数据发布页](https://github.com/Learning-A/final_test/releases/tag/diagnostic-ai-training-20261121-v2)提供ZIP及SHA256附件。无需集群登录，B可以直接下载。包约4.3MB，ZIP校验值：

```text
0e36d0a5a54e5a74f86a4327688ed09ad62c38bea48317a9da68672ac810ebd6
```

PowerShell示例（在仓库根运行）：

```powershell
New-Item -ItemType Directory -Force .local/data | Out-Null
Invoke-WebRequest 'https://github.com/Learning-A/final_test/releases/download/diagnostic-ai-training-20261121-v2/ai_training_fault_exploratory_v2.zip' -OutFile .local/data/diagnostic.zip
Get-FileHash .local/data/diagnostic.zip -Algorithm SHA256
# 与上方哈希一致后再解压；输出目录应为本次新建，不覆盖已有数据。
Expand-Archive .local/data/diagnostic.zip -DestinationPath .local/data/diagnostic
python scripts/export_handoff.py --verify .local/data/diagnostic/ai_training_fault_exploratory_v2
```

manifest本身的SHA256为 `4951b72711a72bd7cf66ecea9b9f700cea702345d2def0fa68475fc23e859d5d`，可与数据清单核对。

## 包内容与限制

- `processed/windows.jsonl`：客户端、OSS、集群和节点窗口，保留complete标记。
- `processed/quality.json`：来源哈希和完整性摘要。
- `features/idfs*.jsonl`：八节点415维输入和读写标签，共4532个有效因果特征样本。每行role=diagnostic、fault_affected=true，标签窗口限定在记录起点后600秒内。
- 记录起点、现场负载manifest、负载退出状态、节点质量报告、故障排除标记和使用说明。
- `handoff_manifest.json`：16个文件的大小和SHA256，不含自己；记录允许探索、不认证正常训练与正式验收。

原始采集739312条计数未随包上传；本包是从原始计数处理而来的窗口/特征，不是背景数据。没有重新训练模型，也没有对特征拟合归一化。

样本数：idfs44=583，idfs45=578，idfs46=578，idfs47=485，idfs64=564，idfs65=583，idfs66=583，idfs67=578。不同节点缺失程度不同，不能补零凑齐。

这里只有一轮，训练/验证若按时间分段，必须标注“同次采集时间切分”，中间清除跨边界历史与标签；不能宣称独立泛化。故障前后分段只能依据已记录健康证据，不得事后按误差挑窗口。离线可用时间来自源读取证据，不等同线上实际到达时间。

## B的第一批工作

1. 登记收到的数据集ID与哈希，复核节点范围、mask和实际分布。
2. 用相同可用历史比较last_value、五秒均值、已有MoE输出尺度和校准过冲。
3. 提出架构、目标缩放/损失、校准限幅/衰减的消融方案，记录理论假设和失败场景。
4. 通过训练请求明确此数据集为故障探索集，交A执行；现有正式训练读入器不接受diagnostic角色，不要将它悄悄改成正常train绕过检查。

## 负载来源

唯一明确来源为用户指定的 [ChrisLzt/idfs-workloads](https://github.com/ChrisLzt/idfs-workloads)。本次核对上游main与现场登记均为 `fb83711a1226f59899ed034b9195402156ef57d6`。未来若上游更新，必须重新记录提交，不能自动将新代码和旧manifest混用。现场其他prepare任务的来源和数据路径还需另行核对，本次没有启动或修改它们。
