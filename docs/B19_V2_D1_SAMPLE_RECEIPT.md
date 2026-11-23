# V2-D1 样例限定范围回执

2026-11-23。接收 `A18-V2-D1-SAMPLE-20261123-01`，状态为 `verified_limited_sample_schema`。B 独立读取 Graph train01、WRF train01、AI inference train03 原 raw 的固定 400 ms 切片，重算 112 实体的相邻样本，不复用 A 的差分函数。全量 raw 哈希沿用此前已验收清单；本次另记了实际重读切片 SHA，不能声称重新哈希了整轮。

已核对 1,344 对有效核心差分、源读取结束与实际单调时间间隔、2,304 个读写 bytes 差分、1,152 个 op_wip 末端快照、3,456 个诊断 latency 差分，并构造 96 个完整12 OSS节点桶。样例文件 SHA 为 `d9396c7923dc715d7c581a5ea3086ca3cc24db9bdb4d51f8bf8e7e9d7decf550`。latency 单位未知，仅核对数值诊断，不进入模型输入。

核验支持 A 继续导出原15train的 bytes/op_wip、各字段独立 mask 与源时间。先验收节点 pilot，再验全量 manifest/成员和 core 的 count/available 一致性；缺值保留 null+mask，不缩减共同样本，ops=0比值缺失，gauge不差分。此回执不代表全15轮已完成或在线时间资格通过。

[核验报告](../reports/v2_d1_sample_B_verification.json)；入口 `scripts/verify_v2_d1_sample.py`，用时约0.192秒，已登记180秒硬上限。无拟合、无现场工作。交接 `B19-V2-D1-SAMPLE-VERIFIED-20261123-01` 已直接发送 A，不重复催发。
