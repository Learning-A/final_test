# 原生负载机制驱动的P2输入对照

用户提供原生仓库ChrisLzt/idfs-workloads。2026-11-22核验main与采集源均为fb83711a1226f59899ed034b9195402156ef57d6；20轮handoff的model_sha256逐类全部一致，亦与仓库rendered/manifest.json一致。运行用户/路径不同不影响model SHA；不把仓库默认runtime路径当现场配置。

|负载|已核实生成机制|优先输入/建模改进|
|---|---|---|
|ai_training_ses|70%读/30%写、多请求大小、类别内Zipf0.99；600秒固定分布/随机访问/iorate=max，没有epoch或checkpoint事件回放|读写量及大小/字节吞吐、OSS集中度、队列与服务压力；不臆造训练阶段|
|ai_inference_ses|80%读/20%写、64/128KiB各半、类别内Zipf；同一随机访问阶段，没有prefill/decode|客户端历史、存储服务状态/吞吐；不依据名称构造token或decode阶段|
|bigdata_baleen|真实选定键的原联合访问频率，每1秒切换对应原源区间；600秒max-rate无请求配额，大请求可跨多个对象|大小分布/字节吞吐、客户端到OSS放大、短时变点；不把max当固定IOPS|
|graph_zipf|32B随机纯读、固定全局热点/Zipf0.99、600秒单阶段、max-rate/O_DIRECT|客户端活动、OSS负载集中/不平衡、系统停顿/恢复；没有“图迭代阶段”证据|
|hpc_wrf|明确四段：0–60启动读、60–150初始history、150–510checkpoint、510–600最终history；活动对象随角色切换|历史读写切换和热点迁移；仅真实启动时钟/预先可用计划被核验后才加入计划阶段特征|

Zipf描述空间访问分布，不直接规定时间IOPS曲线。iorate=max表示饱和尝试，不是每秒固定配额。业务请求比例不硬约束OSS完成计数：内部放大/缓存/背景I/O仍存在。热点/带宽与IOPS应分清，使用静态模型信息不能伪造未采集字节指标。

来源为各workloads/hot_cold/<id>/README.md及rendered/model.json/manifest.json。AI模型字段明确no native AI event replay；WRF阶段是合成安排而非真实WRF trace。Baleen每秒联合分布虽预先生成，若现场推理未被提供计划及真实启动时钟，就不能读取目标秒时间表作为输入。recording_start不直接充当native RD开始。

## 当前立即执行的有界卡：workload-causal-input-v1

新job workload_causal_input_v1_a01。只用新train.zip的15个原train轮，index1/2共10轮拟合，index3共5轮作内部开发；不读validation.zip标签，本轮新增验证集消费0。source role仍train，完整轮隔离，不伪装正式validation角色。

三组固定ExtraTrees64/depth8/leaf3/seed2026/raw squared loss，只改变输入：base144、base+client20、base+client20+ownOSS18。每节点144/164/182维；直接集群连接八节点自身112维及一份空间32维，再加一份client20和各节点OSS18。相同样本，额外输入缺失只配mask，不删难窗口，不填标签。

38维新增输入已实现causal_client_oss38_v1：五历史client7414读写/总量及observed（20维）；按拓扑中12OSS计算各读写/总量分布、活动比例、观测覆盖率、集中度及读占比（18维）。最近源秒从目标前2秒开始；实际记录可用时间必须早于目标前30ms。客户端只是上下文，不与OSS标签相加。静态拓扑来自新telemetry.zip/topology.json，完整映射SHA核验，不用未来系统遥测回填。

本轮不用系统遥测：5秒采样且覆盖不齐，逐秒对齐/跨机时钟待复核；也不用计划阶段：真实workload开始时钟pending。先检验已经具备的因果上下文，再推进启动/阶段和系统压力特征。

只在b507用户专属新目录CPU4、workers0、GPU不可见，1800秒上限，不改环境/集群、不启动负载。保存样本/代码/数据/拓扑/权重哈希、训练loss、逐窗口raw/final、按节点/负载/直接和求和集群MAPE/MAE/WAPE/零值、覆盖及CPU序列化对拍。树无早停；一次固定拟合，不按内部开发扫参数。原5validation被冻结保留角色，后续仅固定候选后消费并登记重复选择。

若三组无可靠收益，不能继续堆同类特征：优先核对图负载3个客户端RPC=0/OSS总量2–5的中段秒及A现有Vdbench/GC/原始计数记录；再判断是否需要近端100ms、压力信号或状态预测头。低值分类+活跃回归为后续候选，门控只能看历史，不能用目标低值状态真值。
