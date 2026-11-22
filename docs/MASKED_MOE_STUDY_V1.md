# masked144多组MoE与组合损失有界实验卡

experiment_id=masked-multigroup-moe-v1；job_id=masked_multigroup_moe_v1_a01。沿用Issue17/PR16，不重复先前两个已完成测试。原五负载首轮ZIP SHA256 f28fc45ba4e6d80c80da815b06d4e05573a7e9c788cbe199e0a1f01898b40b49，保留健康候选train角色及边界pending、非独立/非正式属性。固定原时间轴，masked144训练1514秒/开发762秒；同时报告旧严格15秒697开发秒共同交集。绝不补标签。

## 四组预先固定对照

- single_log：同144输入的共享MLP单专家，八节点共享骨干/节点bias，另有集群头。
- uniform_log：短35/长77/空间32各两个非线性专家，组内均匀融合、组间均匀融合。
- moe_log：相同组/专家容量，组内二路与上层三路可学习门控，平衡正则0.001。
- moe_combo：同moe_log，损失改为0.5 log-SmoothL1 +0.25训练中位数缩放原尺度SmoothL1 +0.25带训练分母floor相对SmoothL1 +0.05节点和/集群一致性 +0.001门控平衡。相对floor各目标训练值10%分位并下限1，指标分母不变。

原尺度八节点和独立集群头保留，输出均从可用最新历史开始学log空间残差，零初始化输出修正。训练拟合signed-log均值/尺度时排除masked占位；mask原样使用，不沿用旧415非负schema。预测非负，不进行校准。输入无负载真值类别/未来阶段/目标标签。

seed2026，Adam lr0.001，batch128，最多60epoch/组、patience12，梯度norm5；总任务1800秒，单空闲GPU，4线程、workers0。相同开发选模/早停指标为五类×九目标正值MAPE宏均值，记录所有epoch训练loss和开发loss，保留epoch0历史基线可被选中。开发复用带来选择偏差，四方案不作独立验收。没有数据/参数搜索或自动延长预算。

保存checkpoint及CPU TorchScript哈希、全部raw/final预测、source身份、输入/样本哈希、每类/每节点和直接/求和集群MAPE/MAE/WAPE/零值、共同交集/全覆盖、GPU/CPU对拍、全部九输出CPU延迟、门控均值/熵。独立复算。单专家的MoE门控值为未使用的诊断，不能解释成该模型使用了专家。门控均值/熵不能单独证明专家有效分工；后续需贡献/移除对照。

## 共享GPU协调协议（本次补齐，双方可见）

2026-11-22检查未发现既有全局锁/登记，A尚未回复先前惯例请求。B不再仅依空闲快照：在Issue17公布本实验及固定共享锁路径，由shared_gpu_lease执行锁定和占用双重检查。两端所有新GPU任务应使用同一路径，旧worker不自动采用新协议；A应在启动前查看Issue17及登记，避免并发。

- 锁：`/home/lmj/idfs_forecast_lab/model_b_20261122T074416Z/coordination/gpu_training.lock`，全任务持有非阻塞flock；锁忙直接退出，不动其他PID。
- 逐任务登记：同目录`<job_id>.json`，新建不覆盖，含GPU UUID/PID/状态/起止时间。
- 选择无compute PID、0利用率、显存<=64MiB的GPU，并在锁内启动前复查。CUDA_VISIBLE_DEVICES固定该UUID；只操作B本次child PID。失败保留输出、状态，不能覆盖重试。
- 此协议已在GitHub发布才执行。它不声称A已确认采用，也不能保护绕过协议的其他用户任务；不抢占任何进程，发现选中GPU已有任务则拒绝启动。

执行目录为B专属新job，复用tsfm环境，不改环境/集群、不采集。正式发布提前量和A现场验证仍未认证，离线CPU延迟不能代替现场结论。
