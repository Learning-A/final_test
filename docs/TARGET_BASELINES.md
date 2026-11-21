# 八节点与集群双目标开发基线 v1

本候选只做确定性回放，没有拟合或现场操作。保持正式协议入口不变；两种目标按用户最新要求并行保留，最终等待正常负载独立开发验证。main基线提交：1a112eeb922a10690d0c38425398c6c0f113d7d2。

此处“双目标候选”的授权来自B本聊天用户任务原文：“预测目标保留‘8个存储节点分别预测IOPS’和‘整个集群IOPS’两种候选，根据开发验证结果最终确定”。这指研发比较，不代表已更改正式验收对象。当前正式目标仍为八节点host_oss总量，集群仅作开发诊断候选；本PR不修改正式目标/指标配置，最终冻结前需另行同步目标决定。

回放来源字段区分base_source_commit（研发基线）与candidate_commit（执行时Git HEAD），并记录candidate_code_dirty和两个代码文件的SHA256。无Git副本时commit为空，以代码哈希标识，不伪造候选提交；代码改动后未提交时dirty=true，不能把HEAD当作准确代码版本。报告更新提交与其执行代码提交可以不同，以报告中的执行提交和代码哈希为准。

## 输入与复现

在工程目录中导入 `idfs_lab.target_baselines.paired_baselines` 和 `metrics`。对每个独立run分别读取已经通过交接哈希验证的 `processed/windows.jsonl`；传入该run的完整窗口以及 `site.storage_nodes` 的八个hostname。禁止把多个run拼接后传入。仅消费 scope=host_oss，由现有12 OSS完整聚合得到的节点窗口；客户端计数不参与集群实际值。

```python
from idfs_lab.io import read_events
from idfs_lab.target_baselines import paired_baselines, metrics

result = paired_baselines(read_events(windows_path), hosts)
node_reports = {
    host: metrics((r['node_last_available'][i], r['node_actual'][i])
                  for r in result['pairs'])
    for i, host in enumerate(hosts)
}
cluster_report = metrics((r['cluster_last_available'], r['cluster_actual'])
                         for r in result['pairs'])
```

mean-five使用对应的 `node_mean_five` / `cluster_mean_five` 字段。基线代码不选择数据角色、不校验数据包、不执行正式验收；调用者须先验证数据清单/role/哈希并逐run隔离，保存dataset_id、run_id、workload、输入SHA256、候选提交SHA、协议哈希、schema和Python版本。故障/背景回放只能标诊断，不进入正常开发结果。

## 因果、覆盖及指标

每个目标秒使用开始前第2至第6秒的五个完整节点历史窗口；最新历史结束于目标前一秒，实际可用时间不得晚于目标前30ms。输出的是理论回放截止点，不是实测推理完成或发布证据。缺节点、缺历史、迟到均拒绝，不补零。标签必须八节点同时完整；仅历史可用性影响预测，未来标签不影响预测值。

`candidate_windows` 是输入中至少出现一个节点的秒数；`rejected` 互斥记录标签不完整、历史不足和历史迟到。所有节点都完全没有记录的秒无法由输入发现，A应另按采集清单预期时间轴核对，不能把该计数当全采集覆盖率。

同时保存逐秒预测、八节点指标及集群指标，按run/负载分别报告。MAPE只排除实际零值，accuracy=100−MAPE；MAE/WAPE作诊断，不能替换正式口径。空样本或全零分母返回None，不声称达标。

## 可检验假设和后续候选

对线性的last-available与mean-five，在完全相同历史上，直接集群基线等于八节点基线求和。因此本候选建立比较下限，不能单凭两者相等选择最终目标。独立训练的集群模型、非线性节点模型及校准后可出现差异。节点误差可抵消，集群MAPE较低不能推出八节点均达标。

已有冒烟摘要中，last-available MAE约338.724，raw-MoE约655.197，发布校准MoE约433.041；校准筛选最佳MAE约419.485，未胜过简单基线。这里只是同一冒烟选择集的描述，不是新实验。模型背景仓库的树模型/FourierGNN及R²结果不构成本项目正式证据。

正常数据到位后再提交唯一训练issue：固定完整轮次15训练/5开发验证，先跑基线，再消融树模型/GRU/TCN/MoE、log-Huber/训练尺度线性Huber、无校准/因果残差校准。直接集群训练适配尚未包含在本候选。训练请求必须补齐实际数据ID和哈希、种子、预算、依赖和输出路径；目前不创建缺数据的可执行训练请求。所有拟合仅由A在指定b507目录执行；正式三轮留出。

本地功能检查包括缺节点、迟到边界、未来标签不改预测、重复窗口拒绝、零值口径和误差抵消；检查通过不代表准确率或30ms发布通过。

## 已收到诊断数据

Issue #3指定的Release v2已下载，ZIP、manifest及16个成员文件哈希均通过。保留diagnostic/fault标记，可以立即开展探索。4532条已有联合特征不是独立验证样本；本次只用窗口回放，没有拟合。

报告reports/diagnostic_paired_baselines.json：600秒记录区间内451个八节点共同完整且历史及时的目标窗口。集群last-available WAPE=38.84%、MAPE=430.39%；mean-five WAPE=32.10%、MAPE=346.92%。这不是全程覆盖率或正常负载结论。WAPE改善未解决小正值相对误差，两种目标继续保留。未获得该数据上的MoE权重及逐窗口预测，不能把旧冒烟MoE与本轮比较。

复现：在仓库根执行 `python scripts/replay_diagnostic_baselines.py --archive <Release ZIP路径> --output <新报告路径>`。脚本校验ZIP、manifest及16个成员，只在内存读数据，不拟合、不连接远端。环境、输入哈希与候选代码哈希均在报告中记录。

下一步开发独立diagnostic读入/时间切分适配，保持原role及源run_id，清除跨边界历史，不绕过正式训练读入器。探索训练无需等待正常数据，但必须有显式诊断适配及固定配置；请求由A执行，本分支未触发训练。
