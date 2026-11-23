# V2-M3 短因果卷积结果

M3 的节点宏 MAPE 为 115.4762%，比复用的 R2 L0 基线 112.3163% 相对高 2.81%。最差组从 341.4612% 升至 347.6438%，两臂均为 0/40 组低于 10% MAPE。普通阶段的宏 MAPE 贡献为 32.8384 个百分点，基线为 32.8268。下降首秒的贡献从 79.0361 升至 82.1619 个百分点。M3 未通过预设筛选。 [执行登记](https://github.com/Learning-A/final_test/issues/19#issuecomment-7091177275)

同一批已构造输入上，单线程 CPU 对八节点执行一次模型前向，Graph 的 100 个样本中，P95 从 0.177 ms 升至 0.968 ms。测量范围为模型前向，实际服务接收、特征构造和发布耗时仍待验证。 [CPU 计时](../reports/v2_m3_cpu_forward_timing.json)

本卡新增一次拟合，B 专属 GPU 租约 946.968 秒。第 4 轮为最佳检查点，训练 12 轮，含第 0 轮共计 13 次内部评分。B 按冻结预测复算全部 2,661 秒、21,288 个原始节点标签和 80 组指标。外部验证新增评估为零。 [训练报告](../reports/v2_m3_short_causal_tcn.json) · [复算记录](../reports/v2_m3_result_verification.json) · [交付包](https://github.com/Learning-A/final_test/releases/tag/b19-v2-m3-short-causal-tcn-20261123-v1)

A 独立核对了 2,661 秒、21,288 个节点标签、80 组指标和阶段误差贡献，确认 M3 未达到晋级门槛。I1、M1、M2、M3 的晋级结论均已复核。S1 缺少可靠的低状态先兆和匹配的训练轮外预测，G1 和 F1 没有晋级候选，本轮按计划收口。 [A 复核记录](https://github.com/Learning-A/final_test/blob/6c2e8b2/reports/v2_m3_A_independent_review.json)
