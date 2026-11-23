# 首次五负载端到端开发测试

唯一test_id：all-workloads-first-round-e2e-v1。

用户已明确告知五类负载首轮采集完成。按此确认登记采集阶段完成；当前不能把GitHub交接缺项误写成现场尚未采集。B已核对main、所有候选分支、issue及Release：AI训练与AI推理已有可校验包，Baleen/Graph/HPC暂未发布到可见清单/附件。无需等待第二轮或20轮，但需要取到这三类数据才能实施五负载测试。

## 预先固定的测试范围

- 覆盖ai_training_ses、ai_inference_ses、bigdata_baleen、graph_zipf、hpc_wrf五类各一个有效首轮，保留dataset/batch/attempt/run身份、角色、哈希和缺失/故障信息。
- 任务为开发试运行，formal_acceptance=false，independent_validation=false。若每类仅有train轮，按各自recording_start预先固定训练[0,360)、开发验证[420,600)，丢弃跨边界历史及间隔；角色不重标为独立validation轮。
- 基线使用last-available/mean-five；有界拟合候选优先固定岭回归与ExtraTrees，用相同因果输入/相同切分定位新增信息价值，不做广泛参数搜索。全八节点分别评估，同时比较节点预测求和和直接集群预测。
- 输入先比五历史总量与读写/5、15秒统计/空间mask；60秒上下文作为覆盖审计而非默认强制要求，避免已发现的交集收缩。具体schema/特征顺序和共同样本清单先冻结，不能通过换交集伪造提升。
- 预处理/特征构造/数据检查允许本机执行；所有学习参数拟合在b507新任务目录、4 CPU线程、workers0执行，GPU任务需先协调单空闲GPU。树模型可使用服务器CPU，不强制占用GPU或安装新环境。
- CPU导出/预测对拍、单样本延迟、时间因果回放及独立指标复算完成后报告全流程状态。现场发布/30ms证据若未执行，明确为未验证；不能把数据集回放当现场通过。
- 每负载/节点/集群输出MAPE、MAE、WAPE、实际零数、样本数、完整性/可用性及预测日志，正式MAPE仅排零。报告最差组，保留失败项，不把平均分当达标。

## 数据交接及运行前检查

AI训练：ai_training_20261122_train01_attempt02，ZIP4505b0240625bfa16ae840ec66f7d82fc4cf67a8a821805e84e50db52cf59314，已核验。

AI推理：dev_20261122_ai_inference_ses_train_01_a01，ZIP5b825034e3d2807bab862e67f220c85b028561c26b0f148be39f7e001fde44e6，已核验。

另三类需要A同步实际batch/attempt/run、可读取路径或附件、manifest及文件哈希、退出/健康/时序质量。现有两包的边界审查请求保留；如证据仅够recording_start范围的开发回放，明确限制，不擅自认证正式可用。A的负载采集控制器不因本任务重复启动。

共享GPU协调规则尚待A回复；B已获用户直接训练授权，这不需要再次请求用户许可。任务登记后先准备数据/代码，执行前使用双方可见锁和作业记录，避免与A运行中的任务冲突。

检查同test_id的issue、远端目录与本地状态；已有running/completed不重复。数据到位并核验后才将状态从awaiting_handoff改为ready/running，不提前声称训练或完整测试已启动。配置、代码SHA、数据哈希、环境、预算及seed2026均在实际实验卡固定。

## 可执行的只读交接门槛

`python scripts/check_first_round_gate.py --plan <条目列表JSON> --output <新报告路径>`仅核验数据，不启动全流程或拟合。每条含workload、dataset_id、archive、archive_sha256、manifest_sha256、time_boundary_reviewed、quality_approved_for_development、review_evidence_reference。后3项必须来自实际质检/时间证据，不能猜测或为启动测试擅自填true。

模块核验五负载唯一性、ZIP及manifest/成员哈希、数据身份/源角色/非故障标记、必需证据文件、负载manifest.id、recording_start事件与成功退出；报告保留batch/attempt/run身份。它核验交接及已记录的审核状态，不替代原始时序/健康质检，也不认证正式数据。

状态区分awaiting_handoff、pending_quality_review、ready，缺包不表示现场未采集。2026-11-22 16:47检查中两包各6成员再次核验通过，仍待边界/开发审核；三包尚未交接，ready=false。该状态不同于训练running或completed。
