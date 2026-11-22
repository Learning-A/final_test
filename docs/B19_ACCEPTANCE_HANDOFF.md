# B任务19：认领及现有实验核对

owner=B，status=accepted。用户要求当前先分工与核对，不启动新训练。任务[19](https://github.com/Learning-A/final_test/issues/19)，数据端[18](https://github.com/Learning-A/final_test/issues/18)，总协调[17](https://github.com/Learning-A/final_test/issues/17)，继续使用PR16及model/causal-multiscale-inputs，不另建实验分支或复制任务。

已读当前协作分支3521d3b273a2c4bb92444b5679f1f3a3b4a8ce6c的AGENTS、POST_CAPTURE_COORDINATION、TASK_B_MODEL_OPTIMIZATION、TASK_A_DATA_EVIDENCE、STATUS、COLLABORATION和数据目录。旧STATUS/目录尚有采集初期记录；本次以最新用户要求、当前分工和已校验20轮Release为准，不据旧记录重启采集或恢复心跳。A维护共享数据状态，B不覆盖A正文。

## 当前职责与停止边界

A负责原始计数、质量/时钟/遥测/池证据、辅助输入交付和独立复核。B负责因果处理、输入、架构、多MoE、损失、校准设计和有限实验；需要训练时执行已授权b507目录内的资源协议。集群已交付，双方心跳暂停；本次没有训练、拟合、采集、部署或现场操作。CPU训练也纳入当前共享锁规则，不以历史CPU任务无GPU为由绕过锁。

正式对象仍为八节点各12 OSS的未来一秒总IOPS；集群另作研发对照。严格准确率>90%、100×(1−MAPE)、仅排除实际零值不变，不删除低正值或缺失补零。

## 只读核对结果

核对时间2026-11-22 21:50:39北京时间。命令为本地SSH只读脚本，远端仅枚举B专属目录、读取报告/模型哈希、查询本用户进程及GPU计算进程；共享锁只用已有文件的非阻塞共享查询并立即释放，不获取训练独占租约、不写锁文件。

共享路径：`/home/lmj/idfs_forecast_lab/model_b_20261122T074416Z/coordination/gpu_training.lock`。快照未被独占，GPU无计算进程，本用户无正在运行的B训练；这不是后续启动时资源检查的替代。历史神经任务租约为released或failed_or_interrupted，没有重置历史状态、删除锁或操作其他PID。

|job_id|核对状态|处理|
|---|---|---|
|all_workloads_first_round_e2e_v1_a01|completed|保留唯一首次全流程结果，不重跑|
|first_round_input_loss_v1_a01|completed|保留六组输入/损失结果|
|masked_multigroup_moe_v1_a01|无完成报告，失败租约|已知CUDA确定性问题，失败保留|
|masked_multigroup_moe_v1_a02|completed|四组MoE/组合损失结果保留|
|frozen_moe_group_diagnostics_v1_a01|completed|冻结模型诊断，不重新拟合|
|legacy_core_current1s_v1_a01|无完成报告，失败租约|1800秒预算超时，不自动续跑|
|legacy_core_recovery_v1_a01|partial_results_recovered|仅已完成GRU部分，TCN不完整不冒充完成|
|frozen_independent_rounds_v1_a01|completed|五轮开发验证已消费一次，不宣称未见|
|workload_causal_input_v1_a01|completed|源e7f1db6，27树模型保留|
|causal_low_state_v1_a01|completed|源93c403d，分类器未采用|

精确清单和哈希见[核对清单](../reports/b19_artifact_inventory.json)。最新两份远端报告与本地字节哈希一致；逐窗口预测哈希均匹配报告；27个回归器、一个状态分类器及样本清单匹配已发布manifest。这里是B的产物一致性检查，不冒充A独立复核或新的推理对拍。大产物仍在上述B工作根的各job/results，报告和执行命令已在PR16，现阶段未冻结新候选。

P2固定5328拟合秒/2661内部开发秒；train01/02拟合、train03内部留出，保留原train角色。本轮新增外部验证消费0，总外部验证消费仍1。低状态38个实际阳性只命中4、误报98；硬修正宏MAPE135.01%，冻结基线134.79%；软修正105.44%但五类集群MAE/WAPE全部恶化，因此不采用、不继续扫阈值。

数据版本development-all-20-rounds-20261122-v1：15train、5validation，首五轮去重；train.zip SHA256=889d303141802a9c826497eb3cf5f31ddac7c7a5d4db63c48698c5894d1d4425，dataset_index SHA256=52e0bf86ff62079e6104a78b58126f59d0f63210ecb405d86b1614d98e92fd2d。正常认证/跨机到达时间证据仍待A，不将开发探索写成正式达标。

## 唯一交接请求与下一步

request_id=`B19-FINE-SCHEMA-SAMPLE-20261122-01`，owner=A，status=queued，见[#18请求评论](https://github.com/Learning-A/final_test/issues/18#issuecomment-7080142124)。该请求承接#17旧100ms请求，先对齐A已有Graph验证轮三个低值事件审计与最小schema，不重复启动审计或整批导出。导出代码沿用67dedd7；原训练轮四事件需求保留为样例verified后的批量步骤，未取消也不并发再提一份。

顺序为：A提供schema/小样例/哈希及来源→B核对身份、整数时间、单位、缺失、标签复算和因果可用性→同request_id回复verified或needs_fix→确定既有train03原始数据批量交付→预登记近端100ms与压力输入的有限对照→仅在数据证据与预算具备后启动新job。验证样例只作事后诊断，不混入拟合，未来邻域不能作为预测输入。

后续所有拟合都使用既有共享flock覆盖完整子进程生命周期，并登记job/owner/PID/代码数据SHA/预算/状态；4CPU、workers0、默认1800秒，神经训练仅启动时空闲单GPU。锁忙、数据无法证明因果或预算耗尽则停止对应任务，不扩大预算或自动重跑。接收者未实际检查前不写verified；GitHub评论不自动唤醒暂停端。
