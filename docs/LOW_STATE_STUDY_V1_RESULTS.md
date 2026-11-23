# 因果低IOPS状态及冻结回归修正结果

实验causal-low-state-v1，job causal_low_state_v1_a01，源码93c403df4f1592ce34cccccfd534d138fafcfa63。b507 CPU4/workers0、无GPU、55.05秒完成；同P2的5328拟合秒/2661内部开发秒，原source role=train。独立validation本轮新增消费0，定时保持停止。

辅助状态actual_cluster<=max(10,0.1*last_available_cluster)只作训练标签，不改变正式MAPE或丢弃低值；1092维输入不含未来值。一个固定ExtraTrees分类器（balanced），阈值0.5不调，quiet中位数只从79个训练quiet样本拟合。P2回归器冻结且精确复现，新增的是状态修正，不是活跃样本专属回归器。

## 固定阈值状态识别

内部开发共有38个辅助低状态窗口；命中4，误报98，precision=3.92%，recall=10.53%。各负载ROC-AUC约0.54–0.70，AP约0.012–0.078，阳性样本极少，不能以ROC看似高于0.5宣布可靠。

原始predict_proba来自class_weight=balanced模型，未校准，不能直接当真实事件发生概率。全体Brier=0.0983，训练阳性率常数基线=0.0141，概率质量明显不足。此常数只用于事后评分诊断，没有用它选择模型或校准参数。

## 同样本回归指标

|方案|宏MAPE%|
|---|---:|
|frozen_regression|134.79|
|hard_state|135.01|
|soft_state|105.44|

硬修正没有改善；软修正虽降低MAPE，但Baleen集群MAPE35.90→40.32%，且全部五类集群MAE/WAPE都恶化。图集群MAE1516.90→2719.09，WAPE18.42→33.01%；不能据宏MAPE降幅称为全局成功。原始分类分数偏大，软修正整体压低预测；尚未做训练先验常数收缩/概率校准匹配对照，不能把降幅全部归因动态状态识别。

150组节点/负载/直接及求和集群指标与分类Brier独立复算，预测SHA核验；分类器CPU序列化对拍通过，raw/final及每窗口原始分类分数/状态标签分开留存。没有发布/冻结，没有改正式指标；102项功能检查通过。

## 结论和下一步

当前条件下，这个固定状态分类器的可靠性不足；该单模型结果不证明瞬停不可预测。优先从原有采集导出100ms节点桶及低窗口原始bytes/latency/op_wip/RPC字段，结合现有Vdbench/GC日志核验；不重开负载，不把低值删除。

离线导出器scripts/export_existing_fine_context.py及EXISTING_FINE_EXPORT_REQUEST.md已准备，A只需指向现有源文件并打包。等待具体100ms/日志交接时可继续实现时间安全的近端/压力特征；不继续扫分类阈值或把未校准balanced分数作为真实概率部署。
