# P2现有输入增量准备（不启动新采集）

五负载首轮同一固定ZIP已含client_nfs_rpc、oss、host_oss、cluster_oss四类秒窗口。各类都包括read_iops/write_iops/total_iops、complete、n_ticks及source available_epoch_ns。16客户端中仅client7414有非零序列，其余真实零与缺失必须区别；不把16零通道简单堆成有效信息。

现有可立即适配的两组候选：

1. client7414及完整客户端集合的历史读/写/总量、变化和活动mask，只使用预测截止前已结算且到达的秒。RPC只是跨层上下文，不能加到OSS标签或宣称已知请求到节点归因。
2. 96OSS历史的节点内读写分布、最大/最小/标准差、活跃数及集中度。oss实体是数字ID；必须核对现有site拓扑12OSS到host映射，不能从数字范围猜映射。标签仍要求完整12OSS/八节点，输入缺失占位必须mask。

先完成P1旧核心固定基线，再按只加client、只加OSS分布、两者均加的预先固定对照。共同原样本和各自覆盖都报告，不混入目标秒。当前只有source-read-end时间，实际接收/时钟边界pending，适配只用于有明确限制的开发探索。

包内缺100ms桶和queue/inflight/latency/CPU/network字节等字段；已有Issue17请求A导出原始采集/处理副本及证据，不重复开任务，不由B启动IDFS采集。不能因为100ms缺项而把已经存在的秒级客户端/OSS信息也搁置。

只读盘点数据见reports/existing_input_inventory.json，当前没有为此启动新增训练或选择特征权重。该盘点只是字段/覆盖/非零统计，不证明任何字段可以提高预测精度。
