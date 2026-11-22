# B工作流v2：新数据接收、时间证据和输入复核

本次完成此前未提交的AI训练候选初审，并接收AI推理候选；不再只在旧故障集上迭代GRU。原故障实验的固定数据/角色不替换，本次没有拟合或启动任何现场负载。

|候选|ZIP SHA256|manifest SHA256|成员校验|共同因果目标秒|
|---|---|---|---:|---:|
|ai_training_20261122_train01_attempt02|4505b0240625bfa16ae840ec66f7d82fc4cf67a8a821805e84e50db52cf59314|1b8196d7a0da8ed218936224ff5370af294a5dd04e5ad1fcedfe31e2004611a3|6|562|
|dev_20261122_ai_inference_ses_train_01_a01|5b825034e3d2807bab862e67f220c85b028561c26b0f148be39f7e001fde44e6|b3c20ff8750f8ac8c206d46eca5d782fe3c4c3b7c4ec9738dc2b888b7c2d94b2|6|546|

两包role=train、fault_affected=false，但normal_training_certified=false，review_status仍为candidate_pending_time_boundary_review，不能擅自改成独立验证或认证正常轮。第二包登记尚在A的PR13分支，以对应Release和manifest为来源，不能声称已合入main。

当前统计区间仅为recording_start后[0,600)秒内的完整秒标签。由于t0不是整秒，区间内有599个绝对网格完整目标秒；AI训练各节点完整597—599秒，AI推理595—599秒。562/546是共同标签及五历史均完整且及时的交集，不是整段采集覆盖率。零值保留，缺失不补零。

两个来源的source_read_end_only_not_online_reception_proof不能作为线上接收证据。workload_exit.end_local_ns减源recording_start分别约623.587/621.207秒；该差值涉及不同时间来源，既不能直接证明时钟错位，也不能认证600秒负载边界。需要A提供去认证信息的负载启动/结束事件、各clock_host、偏移/不确定度和采集到达时间证据后再确定训练区间。不得按当前误差挑选起点。

在同一因果交集上复算确定性基线：AI训练集群last/mean-five的MAE=883.972/736.858，MAPE=310.426%/286.807%；AI推理=1994.489/1626.328，MAPE=446.964%/388.821%。这些是未拟合、非独立验证的初步描述；两包负载/尺度不同，不能直接以数值大小判断模型跨负载优劣或达标。

原故障轮与新健康候选可能共用source_run_id=ai_training_ses_train_01。后续以dataset_id+source_batch+attempt+manifest哈希标识完整采集身份，同时保留原run_id，不能按短名称误合并或伪造两个独立轮。

## 输入/处理下一步

1. 在一秒标签保持不变的前提下，先做历史read/write/total、差分、滚动分布及跨节点mask的因果特征组；对同时间戳单组增量比较，拟合放在GPU服务器用户空间。
2. 当前交接只有一秒windows，无法恢复秒内突发顺序。请A检查能否从已有处理产物导出对应100ms有效桶（read/write counts、scope/entity、bucket时间、源可用时间/实际接收时间、质量标记）。不要求重采或启动探针，先做现有信息可用性审计。
3. 请A列明已采集的客户端请求/并发、读写字节、延迟/队列、CPU/网络/健康与已知负载阶段的字段、单位、采样率、clock_host和到达证据；不存在的字段明确缺失，不凭空归因。新增采样需另行评估开销和授权。
4. 在足够完整轮次和边界确认后，建立同样本last/mean/岭回归/树基线，比较基础历史、多尺度、空间和辅助输入。AI推理train轮不能直接冒充AI训练validation轮。没有独立验证时，探索报告须显式声明局限。

用户已在2026-11-22明确授权B后续直接在b507专属目录训练，覆盖旧B仅交A提交训练的默认规则。工作根/home/lmj/idfs_forecast_lab/model_b_20261122T074416Z；A继续独占IDFS负载、现场采集和正式验证。B首个GPU作业前需要双方可见的任务登记/共同锁协议，不能只依赖瞬时空闲；本PR没有启动GPU任务。

脚本scripts/review_train01_candidate.py支持显式archive/manifest SHA参数。默认检查训练包；AI推理包需传上述两个哈希。逐文件验证后在内存处理，不解压敏感配置、不加载权重、不拟合。报告保留输入windows及代码哈希，数据/原始日志不提交Git。
