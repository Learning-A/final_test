# B回执：15轮100ms训练输入与时间覆盖

request_id=B19-R1-TRAIN-CORE-20261122-01；receiver=B；status=verified。范围为实际交付的数据身份、字段、哈希、原秒窗复算、固定样本集合，以及源时钟假设下的覆盖统计。线上接收、逐轮跨机时钟误差界与现场30ms资格不包含在本次verified内。

来源：[R1 Release](https://github.com/Learning-A/final_test/releases/tag/r1-train-core-100ms-20261122-v1)及[A覆盖审计](https://github.com/Learning-A/final_test/blob/fa335371064c03905db13648ee8ec618d032422a/docs/R1_TIMING_COVERAGE.md)。

## 哈希与范围

|文件|SHA256|
|---|---|
|r1-train-core.zip|d6fc9b0e6227d4ea67fbc53c914702c278cf1d4ec2b8d68ec189a538eb9bd0f9|
|r1_training_core_index_v1.json，15轮入口|b4efd63c33aa44dd124c3553233857a5d9d896e6f4c53e42b2ed8f59747381ce|
|dataset_index.json，14轮导出附加清单|975dc0e18d2fa4ee14c08d59a2269f83d602a2e16b5bfb35d389ea1b9257f447|
|export_r1.py，A实际导出脚本|962e906415f47a4bd3772f794cdc48db884f2f4968475e6b4ef3e00ab8ad6aaf|
|A r1_timing_coverage.json@fa335371|61de90975d76542b149b74dd7eadb8387be09c8eb529d5c9f28635d74779312d|

四个Release附件已实际下载并校验，ZIP内dataset_index与单独发布清单逐字节一致；逐run manifest、成员大小/SHA、原handoff/quality/windows/topology及原raw哈希清单对应一致。raw全文件未随核心包交付，B验证的是与原quality所记哈希列表的一致性，不冒充重新读取A本机raw全文件。

15轮身份恰好对应原20轮索引中的15个train，无重复、无validation混入；10轮train01/02拟合、5轮train03内部留出，原source_role=train保持。14轮为新包，Graph train03继续引用已verified的R0原包/原manifest，未要求重导出或重验其既有完整标签结果。

## 独立复算结果

- 新14轮：82,881个有效节点/client7414秒窗的read/write/total及available_epoch_ns与原train.zip逐项相等；508个不完整标记一致。
- 参考Graph train03：复用此前5,921个有效秒窗与36个不完整标记的验收结果。15轮合计88,802个已核验完整node/client秒窗、544个不完整标记；这是全录制范围的数据核验数，不是模型训练样本数。
- 每轮重新构造F0/F1配对，均与旧P2固定sample_ids一致。拟合5,328秒、内部开发2,661秒，近端缺失不额外删样本，不把内部留出改成拟合，不再消费外部validation模型评分。
- 15轮逐节点完整标签范围按recording_start后16至600秒计，共69,820个节点窗；按七档额外余量逐轮复算，可用桶总数、无桶窗口数、最近500ms有桶数均与A一致。

## 源时钟覆盖结果

|额外余量（假设）|平均可用桶/20|最近500ms有桶的节点窗|无任何桶的节点窗|
|---|---:|---:|---:|
|0ms|18.992867|69,820/69,820|0|
|25ms|18.992867|69,820/69,820|0|
|50ms|18.992280|69,820/69,820|0|
|75ms|17.993154|69,819/69,820|0|
|100ms|17.993154|69,819/69,820|0|
|200ms|16.993612|69,804/69,820|0|
|500ms|13.995016|0/69,820|0|

覆盖分母包括每个完整单节点标签，不要求同秒八节点全部完整或F0历史有效；因此69,820不能与5,328+2,661直接当相同样本量。下一阶段模型对照仍固定旧共同样本，不按覆盖结果重新挑窗口。

额外余量是离线敏感性参数，不是已测跨机偏差，不证明多少时间可用于处理或发布。原始Windows采集端的received_local_epoch_ns不等于推理服务实际收到时间；核心数据沿源端read_end谱系，逐轮跨主机时钟界仍unknown，不能把null设0或宣布online_ready。

## 复算入口与后续

[verify_r1_training_core_batch.py](../scripts/verify_r1_training_core_batch.py)只读执行，固定Release/旧train/index/topology/sample_ids/Graph参考包哈希。参数为--index、--delivery-index、--archive、--exporter、--original-index、--train、--topology、--sample-ids、--reference-archive、--core、--reference、--a-coverage、--output。报告为[r1_core_b_receipt_v1.json](../reports/r1_core_b_receipt_v1.json)，保存逐轮复算与覆盖计数；未调用模型拟合或真实新模型评分。

本request的数据交付依赖已解除。B在#18/#19回复verified并将#19恢复为running，接续准备明确时间假设与50/100/200ms敏感性范围的R1离线执行卡；不再以“缺14轮数据”为由等待A，不要求A现场补采。时间假设、选模范围及预算需在运行前登记，源时钟离线结果与线上/正式资格分开，冻结与发布仍守原证据门槛。
