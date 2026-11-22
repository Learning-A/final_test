# 现有100ms和瞬停原始辅助字段离线导出

承接Issue17原有请求，不新建采集任务。A在已有采集电脑本地执行脚本，只读取已有文件；不SSH集群、不启动负载/探针、不修改源标签或环境。B的代码已完成，尚未拿到实际导出数据，不能把功能检查当成交接完成。

优先一个train角色采集：`dev_20261122_graph_zipf_train_03_a01` / run `graph_zipf_train_03`。内部开发中四个低总量秒start ns为1795343004000000000、1795343013000000000、1795343113000000000、1795343199000000000（实际集群8、14、59、53 IOPS）。验证轮的三个事件可后续只作独立诊断，不混作训练样本。

需有原run的raw目录、processed/buckets.jsonl，及当前交接包对应handoff_manifest、processed/quality、processed/windows；拓扑使用已交付telemetry/topology.json。示例为PowerShell一条命令，变量路径由A的实际源目录填入：

```powershell
python scripts/export_existing_fine_context.py --handoff-manifest "$package/handoff_manifest.json" --quality "$package/processed/quality.json" --windows "$package/processed/windows.jsonl" --raw "$capture/raw" --buckets "$capture/processed/buckets.jsonl" --topology "$telemetry/topology.json" --event-start-ns 1795343004000000000 --event-start-ns 1795343013000000000 --event-start-ns 1795343113000000000 --event-start-ns 1795343199000000000 --output "$new_export_directory"
```

输出：

- fine_context.jsonl.gz：全采集范围八节点（各12OSS）及client7414的100ms计数/可用时间；缺成员为无效，真实零保留。单位operations per nominal100ms bucket，不冒称按实际间隔除出的瞬时rate，保留有效成员实际采样间隔范围。
- raw_event_context.jsonl.gz：仅四事件前5秒至后6秒的原始allowlist字段：累计操作/字节、latency、op_wip、rpc_fields、采样/时钟/收到时间（存在才保留）、质量/身份。后续秒只作诊断，不能用于目标预测输入。
- export_manifest.json：dataset/run/role/fault及源manifest/quality/windows/raw/buckets/topology/输出哈希、行数、可用性证据。验证原始raw校验和，并用导出细粒度桶复算原完整节点/客户端秒标签，任何差异报错。没有密码/token/site配置复制。

请A一并提供现有workload.log/Vdbench interval输出和已有GC/暂停记录（不存在则明确不存在），及native RD开始时间对应的主机/控制端时钟证据。源码单阶段/iorate=max不解释这几个低秒；不要根据猜测清洗或重新启动采集。

该导出器已做真实零/缺失成员/重复桶测试；实际A端全量执行和标签复算尚待完成。返回包先按源角色/哈希接收，再构建近端100ms特征和辅助量时序，不从原有秒级窗口伪造细粒度数据，不直接混用未核验latency单位。
