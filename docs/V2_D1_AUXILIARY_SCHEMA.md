# V2-D1 辅助字段样例契约

状态：sample_delivered_pending_B_review。当前只导出3轮固定400ms样例；全15轮尚未导出。来源为既有raw，不连接现场。样例在`.local/v2_d1_sample_a01`，脚本`scripts/audit_v2_d1_sample.py`。

拟议全量包每行：dataset_id、scope=host_oss、entity=node_id、bucket_start_ns/end_ns、sample_ok、available_epoch_ns、expected_oss=12、observed_oss、read_bytes/write_bytes、read_count/write_count、op_wip_sum/op_wip_max、字段各自valid/mask。不能只用sample_ok替代各辅助字段有效性。

时间/核心操作差分身份规则复用counter_delta；bytes另检查非负整数、累计回退及同源同区间。核心有效不意味着bytes也有效；仅12OSS对应bytes全部有效才聚合该方向节点bytes，缺失不得填零。比例在聚合后用sum(bytes)/sum(ops)，ops=0为null并带mask。原核心标签不改变。

op_wip为本桶末端快照的瞬时量，12OSS均为有限非负数才聚合sum/max，availability为参与快照源读取结束最大值；不把gauge做差分。不记录虚构队列长度/吞吐p95。

latency count/sum尚缺源端单位证明：样例只保留差分原单位，现阶段全量训练字段默认不包含latency。单调样例和collector字段名不足以证明秒/毫秒。若后续已有源文档证明单位，再另发schema修订并只用delta_sum/delta_count，禁止差分avgtime/使用全程maxtime作桶p95。

客户端先保持原16实体来源和活动审计，不将RPC加到OSS目标，不把长期恒零通道当业务先兆。原core的client7414保留，无需在此包再导出同一核心计数。

字段归一化仅拟合角色有效观测；缺失附mask/age。源读取可用不等于推理服务收到，包维持offline_source_clock_only、online_time_eligibility=false、clock_uncertainty_ns=null。新字段输入截止沿用S−80ms，不能用目标后压力解释当预测输入。
