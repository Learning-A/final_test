# R1完整训练范围交付格式 v1

request_id=B19-R1-TRAIN-CORE-20261122-01，owner=A，receiver=B。承接已verified的Graph R0样例，正式固定最小字段与容器，不新增采集，不重新导出已有Graph train03。用户已授权按计划开始；心跳与现场操作仍暂停。

## 交付范围和文件

沿development-all-20-rounds-20261122-v1原dataset_index的15个train轮，train01/02共10轮拟合、train03共5轮内部开发；保持原role与完整轮身份。Graph train03引用原R0包/SHA，新增只需剩余14轮。validation包不加入这次训练交付，事件切片不是完整训练范围。

索引模板：[r1_training_core_index_v1.example.json](templates/r1_training_core_index_v1.example.json)。其中state=template_not_delivery与null项是明确占位，不是实际交付或证据；A填真实值后发布。已verified的Graph train03指向现有Release，并保留source_read_end与时钟unknown。

每个新run目录建议为`<dataset_id>/`，至少包含：

- fine_context.jsonl.gz：与R0相同100ms字段，整个既有录制范围的八节点与client7414；有效零保留，缺成员mask，不强行归一速率。首卡模型只消费node fine。
- export_manifest.json：dataset/run/原role/fault，原handoff、quality、windows、topology、buckets和raw文件哈希；fine输出哈希/行数、原完整秒窗复算数量、可用性证据、实际导出脚本及依赖SHA。
- raw_event_context.jsonl.gz可为空或明确省略；新core-only入口产生空gzip并纳入manifest，兼容既有目录结构。首卡不需要新的事件raw/压力/遥测字段，不必重复上传原raw。

可将若干run目录压入一个ZIP分批交付；ZIP按原dataset_id保留目录，不拼接不同run的fine行。packages列表写每个实际包的name、download_url、bytes、sha256；capture.delivery.archive_url/archive_sha256与directory写实际位置。directory只能是包内相对目录，路径不得逃逸。Graph train03使用existing_release且directory为空，因为原包直接含文件。

顶层schema_version=r1_training_core_batch_v1、request_id及原数据索引/训练包/拓扑SHA固定。captures完整列15个原train身份，original_handoff_manifest_sha256不得变；development_partition单独记录，不把train03改成原validation。每个已交capture必须有export_manifest_sha256；status为delivered或此前previously_verified，未交仍pending_export。分批state=partial_delivery；全部文件可取且A已核验哈希后才state=delivered。delivered不是B已verified，更不是时间资格通过。

## 新增core-only离线入口

继续用`scripts/export_existing_fine_context.py`，只增加`--core-only`，不传`--event-start-ns`。其他参数仍指A的现有源文件：

```powershell
python scripts/export_existing_fine_context.py --handoff-manifest "$package/handoff_manifest.json" --quality "$package/processed/quality.json" --windows "$package/processed/windows.jsonl" --raw "$capture/raw" --buckets "$capture/processed/buckets.jsonl" --topology "$telemetry/topology.json" --core-only --output "$fresh_run_export"
```

它仍校验原raw文件哈希、原windows/quality/manifest以及完整秒窗的read/write/total与完整性，不为缩小包跳过来源核验。只省略原始事件行输出，生成export_scope=full_capture_core_only、raw_event_rows=0、diagnostic_event_start_ns=[]，自动保存导出脚本及fine_export/dataset/counters的实际文件SHA。输出目录必须新建，不能覆盖既有产物；不SSH、不采集、不训练、不改源标签。

旧事件模式仍接受`--event-start-ns`，core-only与事件参数互斥。Graph既有包不需要为了新字段重新打包；此前原scope/full label核验已保留，实际修改/脱敏来源补交说明即可。

## 时间状态不能靠填表补齐

availability_evidence如实沿原quality；online_receive_proof没有证据则false，clock_uncertainty_ns没有测量则null，不设为0。A已说明目前归档未发现逐节点同步误差测量或单独GC日志，该未知必须保留；不请求现场补采，不用下载时间充当接收时间。

R1正式卡需要可解释的事件/可用时间界和处理预算。核心字段及全量标签核验可以先通过，但时钟/线上条件不因数据齐全自动通过。若后续只剩源时钟信息，B先记录不能证明线上资格，按既定停止条件不启动声称合法近端因果输入的训练；若另议源时钟假设下的离线研究，应单独登记假设/用途和修订，不能偷换正式/现场证据。

## A/B回执

A在原request_id评论给索引链接/SHA、包链接/SHA、实际交付/缺项及来源代码，先确认此最小格式；若已有兼容格式，可给出明确映射，不要求丢弃已完成导出。B按批核验后回复verified/needs_fix，并分别记录核心数据、原标签、全范围、时间资格，不周期性重验已完成Graph样例。全部训练前仍需具体runner/代码SHA、数据清单、CPU4/workers0/1800秒及既有共享flock登记，A不重复训练。
