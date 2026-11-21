# 数据与模型接口

## 数据集标识与划分

每个数据集使用唯一dataset_id；每次负载run_id不可复用。目录名字相同也不说明两台电脑上的内容相同，须核对manifest里的SHA256。每轮清单记录负载ID/提交、role、目标、采样协议哈希、实际运行状态、记录起点、环境故障和质量结果。

role取train、validation、formal或diagnostic。故障轮/背景轮用diagnostic，不计正常负载训练轮。当前计划五类 `ai_training_ses`、`ai_inference_ses`、`bigdata_baleen`、`graph_zipf`、`hpc_wrf` 各3训练+1验证；FAST不在本批次。验证按完整轮次分开；训练窗口的历史和标签不能跨轮。归一化、早停与超参数只用相应开发数据，正式轮不进入选择。

## 原始与处理数据

原始每条计数的scope、entity、boot/socket/挂载身份、源时间、累计读写数及质量标记须保留。身份变化、计数回退和缺桶标无效；不补零、不插值。采样100ms；一秒标签由十个有效桶计数相加。节点实际IOPS需要完整12 OSS，不混客户端NFS/RPC。

`processed/windows.jsonl` 的关键字段：scope、entity、window_start_ns、window_end_ns、complete、read_iops、write_iops、total_iops、available_epoch_ns。标签对应未来完整一秒，单位ops/s。当前八节点：idfs44/45/46/47/64/65/66/67，OSS连续分组0—11、12—23、…、84—95。

`recording_start.json` 至少包含event=recording_start、epoch_ns、clock_host；这只是开始记录，不声称实际首IO。旧代码为兼容仍可能将该证据存为first_io.json。

## 联合模型输入（joint_context_v1，415维）

特征顺序由 `idfs_forecast_lab/idfs_lab/joint.py` 定义，不能自行调换：

| 下标（从0起） | 维数 | 含义 |
|---|---:|---|
| 0—14 | 15 | 该节点五个历史秒，每秒read/write/total，按旧到新排序 |
| 15—398 | 384 | 96 OSS按ID排列，每个read/write/total/mask |
| 399—406 | 8 | 跨层辅助值 |
| 407—414 | 8 | 跨层mask |

当前使用目标窗口开始前留一秒安全间隔的历史。精确时间和可用性检查见 `causal_features`，不可用目标秒观测回填输入。mask=0代表缺失，输入张量占位零不代表原始IOPS为零。拓扑图是同物理节点OSS分组，不代表客户端请求归属。

特征导出须带run_id、role、workload、target、feature_end_ns、features_available_ns和标签。归一化参数只在训练轮拟合，随模型一起导出。修改维数、顺序、mask、时间间隔或拓扑必须升级schema并同步线上与离线实现。

## 模型与结果交付

候选交付包含代码SHA、schema、目标、依赖、seed、训练配置、训练/验证dataset_id和run_id。GPU任务生成model_manifest，保存artifact SHA256、归一化、训练曲线和验证预测；导出CPU后必须GPU/CPU对拍，现场再核对CPU输出。

CPU模型输出非负有限总IOPS；MoE还输出校准前read/write和专家权重。校准修正总量时不能伪造读写分摊。预测日志需保存raw与final、特征时间、计算完成时间及落盘确认；两者必须早于目标窗口起点。校准只能消费已经可用的历史结算值。

开发结果至少报告每节点/负载的MAE、WAPE、MAPE、实际零值数量、样本数及完整性；与最近可用历史值和五秒均值比较。分清离线回放、背景冒烟、现场带负载验证和正式验收。

## 数据包怎么拿

A从原现场工程生成仅包含训练所需文件的包：

```powershell
python scripts/export_handoff.py --run "F:/sessions/26.9.8_测试大纲/idfs_forecast_lab/runs/workload_development_20261121_v1/ai_training_ses_train_01" --output .local/handoffs/ai_training_fault_v1
```

脚本默认拒绝故障轮。只有明确分析故障时增加 `--allow-diagnostic`；生成包的manifest继续标记为diagnostic，不能因此转成正常训练集。脚本不包含原始collection_manifest/site连接配置。默认只复制processed/windows、quality和选定轮次元数据；权重、原始计数不自动转交。

A把目录通过双方实际可用的共享盘或文件服务传输，核验 `handoff_manifest.json` 中每个文件的SHA256，再更新数据清单的transfer_location、manifest_sha256。GitHub源码仓库不是当前大数据传输通道；没有配置可用传输位置前，B不能声称已拿到数据。

B收包后在仓库根执行：

```powershell
python scripts/export_handoff.py --verify .local/handoffs/ai_training_fault_v1
```

确认dataset_id、role和排除标记后再分析。路径可以不同，内容哈希必须相同。
