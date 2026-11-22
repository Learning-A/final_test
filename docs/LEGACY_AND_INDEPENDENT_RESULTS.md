# 旧核心回收与首次独立开发轮冻结评估

用户最新要求：停止定时任务，继续在当前聊天手动执行计划。两处本机native automation.toml已设PAUSED并读取核验；本会话没有automation_update接口，未宣称调用了该接口。当前自动化不再作为后续执行入口。

## P1旧核心任务闭环

legacy_core_current1s_v1_a01在1800秒上限终止，原目录保留；GPU lease记录failed_or_interrupted并释放。两组GRU全九目标完成，TCN只有五节点完整导出，缺idfs65/66/67和集群，不虚报三组完成，不补零，不自动扩预算或重训。

CPU新job legacy_core_recovery_v1_a01用完整GRU导出恢复四checkpoint变体，200组指标独立复算；源模型GPU/CPU对拍最大相对误差<4.3e-14（含原进度中的已完成TCN导出），当前恢复评分只含完整GRU方案。原697开发秒宏MAPE：global按WAPE选70.65、按MAPE选66.73，window按WAPE选71.08、按MAPE选69.47%。现有MoE同697秒为60.09%。归一化没有整体改善，现有证据不支持继续单纯扩训练预算。

这只是原GRU/TCN核心的秒级适配，不是旧高分完整复现：旧逐负载训练/三seed/100ms长序列，与当前跨五负载pooled/单seed/15秒及预算不同；旧loss已从保存cfg确认为MAE。恢复报告见reports/legacy_core_v1.json，原训练和恢复execution分别保留，不能把恢复CPU推理说成追加训练。

## 新20轮接收

development-all-20-rounds-20261122-v1四资产SHA及149成员核验通过：15train（每负载3）、5validation（每负载1），旧首轮5个采集的windows和recording_start字节相同，去重不重复加入。遥测含24节点CPU/memory/net/block等5秒样本及拓扑；前两训练轮无系统遥测，AI训练train03无样本，AI推理train03部分覆盖。接收时间只做队列interval归属，跨机/逐秒对齐仍pending，尚不能直接作为每秒模型输入。

## 首次独立开发轮冻结评估

在消费新validation标签进行选模前，卡片固定last_available、mean_five、旧global-GRU-MAPE checkpoint及moe_combo。新job frozen_independent_rounds_v1_a01，源码48f7d52，CPU4推理23.65秒，无拟合、无checkpoint/归一化/校准更新。只使用5个原validation轮，分别490/396/486/511/517个共同有效秒，共2400秒。完整15秒因果历史与完整八节点标签一致。

|方案|五负载×九目标宏MAPE%|
|---|---:|
|last_available|175.28|
|mean_five|169.55|
|frozen_gru|141.42|
|frozen_moe|141.60|

直接集群MAPE%：

|负载|冻结GRU|冻结MoE|
|---|---:|---:|
|ai_training_ses|154.09|140.19|
|ai_inference_ses|70.09|70.88|
|bigdata_baleen|115.76|175.56|
|graph_zipf|1368.38|1615.76|
|hpc_wrf|217.77|198.32|

冻结GRU/MoE各自均0/40节点×负载组达到MAPE<10%，集群0/5；独立开发轮仍未达标。200组含节点/直接及求和集群指标独立复算，所有低正值继续保留，raw=final。它检验的是旧首轮模型迁移，不是利用15train重新训练后的结果；新开发验证消费计数1，正式轮仍未使用。时钟/边界及现场发布仍未认证。

## 已定位的新误差证据

图负载511有效秒中有3秒集群实际4、5、2 IOPS，发生在recording_start约194.45、280.45、498.45秒；全部96OSS标签完整，活跃客户端同秒RPC=0，前后秒客户端和OSS都在数千IOPS。不是只出现在采集结束尾部，也不是缺失补零。客户端/OSS同步下降支持“源负载或系统短暂停顿”候选，但不足以断言真实原因或排除采集时间问题。

固定上游fb83711版本graph_zipf.py配置600秒单zipf_all、target_iops=max；ZipfWorkload.java由native RD调度，所读适配器未见显式周期sleep。不能把这些低秒直接解释成已知计划空闲。需要A用现有Vdbench interval/stdout、Java GC/暂停记录（若有）、原始计数及对应时钟说明核对，不由B新开负载/探针，不删除窗口。

图验证最低约10%实际值窗口贡献99.41%APE；WAPE15.96%与MAPE1615.76%同时报告，不以WAPE替代MAPE。仅秒级、最晚结束于目标前1秒的输入很难描述这种突停的近端前兆，但能否改善必须由100ms/客户端/系统因果信息实测，不能宣称不可预测或保证更换损失即可达标。

下一步按既定计划进入P2：使用15train扩覆盖、完整轮内拆分checkpoint开发，5validation保持原角色；先客户端/OSS现有字段，再对齐系统遥测。后续训练卡应只改变一项输入因素、避免直接按本次验证最优结果拼模型。正式门槛不变，定时保持停止。
