# V2-M2 轻量负载适配头结果

加入轻量负载适配头后，节点宏 MAPE 从 112.3163% 降至 109.8460%，相对改善 2.20%，低于预设的 5% 门槛。最差组 MAPE 从 341.4612% 降至 335.0066%，五类负载平均节点 MAE 均下降，降幅为 2.18%–6.33%。两臂均为 0/40 组达到 MAPE<10%。[执行登记](https://github.com/Learning-A/final_test/issues/19#issuecomment-7091091776)

普通阶段对 40 组宏 MAPE 的贡献从 32.8268 降至 31.5375 个百分点，下降首秒的贡献从 79.0361 降至 77.8874 个百分点。记录中的业务类型作为离线条件输入。实际预测前能否取得该信息仍待独立核实。

本卡在 B 的专属 GPU 拟合一个新网络，资源租约 35.461 秒。基线直接复用 R2 L0。第 13 轮为最佳检查点，共训练 21 轮，含第 0 轮共 22 次内部评分。B 从冻结预测独立复算 2,661 秒、21,288 个原标签、两臂 80 组指标及阶段贡献，确认未达晋级门槛。[A已独立复核一致](https://github.com/Learning-A/final_test/blob/c5e984d/reports/v2_m2_A_independent_review.json)。外部验证新增评估次数为零。

本卡已完成并标记不重跑。保留原 R2 L0 骨干作为 M3 对照。M3 将在相同的 20 个名义槽及输入截止下检验短因果卷积分支。

[训练报告](../reports/v2_m2_workload_adapter.json) · [独立复算](../reports/v2_m2_result_verification.json) · [交付包](https://github.com/Learning-A/final_test/releases/tag/b19-v2-m2-workload-adapter-20261123-v1)
