# 因果低IOPS状态可辨识性对照

experiment_id=causal-low-state-v1，job causal_low_state_v1_a01；手动执行，定时保持PAUSED。沿用P2同10train拟合/5train03内部开发（5328/2661秒），原role=train。validation.zip不读，本轮新增独立验证消费0。

问题：在现有秒级历史及client/OSS38上下文中，是否已有信号能预测下一秒整体突降？先检验状态可辨识性，不扩大回归模型、上下文或调参网格。

固定训练辅助状态：actual_cluster <= max(10 IOPS, 0.1×最近可用cluster)。这是训练标签，不是输入，不改正式零mask或准确率定义。特征严格使用既有1092维集群因果输入，无时间/阶段真值或目标值。一个ExtraTreesClassifier64/depth8/leaf3/class_weight=balanced/seed2026，4CPU/workers0，GPU不可见，1800秒上限；不选分类阈值、不校准概率。

先完整复现P2 client_oss182冻结回归预测，然后固定比较三组：原预测、p>=0.5时使用训练quiet样本各目标中位数、概率加权软修正。原回归器不重训；它并非只在活跃样本上拟合，因此本轮叫“冻结回归的状态修正”，不冒称完整hurdle模型。阈值0.5/10/0.1均预先固定，不从内部开发选择。报告所有三方案，不选最优阈值/逐负载拼接。

输出状态阳性率、PR-AUC/AP（与阳性率对照）、ROC-AUC、Brier、固定阈值precision/recall和触发数；回归仍逐节点/负载/直接及求和集群MAPE/MAE/WAPE/零值，raw/final分别保存。低样本precision/recall不作为稳定结论；若PR接近阳性率且修正退化，优先补近端100ms/压力/原始停顿证据，不继续扫阈值。

数据/拓扑/源码/原回归权重/分类器哈希、CPU序列化对拍、训练拟合的quiet中位数、逐窗口概率与真值留存并独立复算。模型只能在b507拟合，Windows仅功能检查/复算。

已核对采集代码remote_agent：原始100ms记录包含read/write_count、read/write_bytes、latency、op_wip，客户端含rpc_fields；dataset.process会产生processed/buckets.jsonl，但当前release只含windows。缺的是既有数据交接，不是必须新采集。原已有Issue17请求承接，不重复启动负载。当前秒标签处理是十个完整有效100ms计数和，身份/计数回退/抖动会拒绝；尚未找到代码直接把无效桶填零的证据，仍需原始事件复核真实瞬停。
