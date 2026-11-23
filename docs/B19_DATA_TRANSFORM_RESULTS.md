# T0–T3 数据变换对照实际结果

2026-11-23，北京时间。T1/T2已完成，T0精确复用R2 L0，没有旧实验重训。T3因训练分布准入条件不满足跳过；没有尝试其它替代变换。两张新卡均未通过原研发筛选，待A独立复核，不声称正式达标。

| 同2661内部开发秒、40组等权 | T0，冻结R2 L0 | T1，log输入+robust | T2，固定T1输入+显式log目标 |
| --- | ---: | ---: | ---: |
| 原节点宏MAPE% | 112.3163 | 112.1806 | 105.9579 |
| 相对T0改善 | — | 0.1208% | 5.6612% |
| 最差组MAPE% | 341.4612 | 336.4734 | 335.2889 |
| 宏logMAE | 0.28529 | 0.29638 | 0.29394 |
| 宏RMSLE | 0.61613 | 0.62416 | 0.60949 |
| 宏WAPE% | 22.1503 | 23.6339 | 22.9231 |
| MAPE<10%的节点×负载组 | 0/40 | 0/40 | 0/40 |

T1五类平均节点MAE相对T0增加1.63%–9.43%；宏MAPE改善很小，原尺度误差与辅助log指标均未改善。T2相对T0的五类平均节点MAE变化为：AI推理+3.99%、AI训练+8.40%、Baleen+1.57%、Graph+4.69%、WRF+0.58%。T2虽然通过5%宏MAPE改善和最差组约束，却因AI训练MAE超过5%退化限额落选。相对T1的改善属于显式log输出+logMAE损失包，不能单独归因输入log或其中某个组件。T2的宏RMSLE较好，logMAE/WAPE反而比T0更差，进一步说明不同指标表达不同偏好。

## 精确执行

输入为固定原C258。178非负连续列log1p，signed差分30/31/32只robust，77个mask/fraction/age保留。统计只在b507拟合训练有效观测，真实0保留、缺失占位排除；T1/T2的transform.json字节SHA完全相同。没有使用留出拟合统计。训练有效列中满足预登记T3长尾条件的列数为0，低于18，因此不拟合Yeo–Johnson。

两臂均使用原5328拟合秒、2661内部开发秒、原train角色、21089参数MLP、seed2026、批次序列和run/node等权。T1保持原L0输出/损失；T2采用z=log1p(s)*softplus(h)、expm1恢复、logMAE，不补反变换偏差。T1 epoch0恢复原s完全相同，T2最大浮点绝对误差0.0000916 IOPS，满足预登记相对容差。两臂按原MAPE选checkpoint，T1最佳epoch4、训练12轮、评分13次；T2最佳epoch6、训练14轮、评分15次。

新job为b19_data_transform_t1_v1_a01、b19_data_transform_t2_v1_a01。工作代码25b5a3c8908eebef033e6148026cefe241f0974b，b507既有Python3.11/torch2.4.1/numpy1.26.4。各1800秒预算，完整租约分别21.300秒和22.543秒，正常退出、释放锁。CPU4/workers0，B原UUID；A卡未占用。没有安装远端环境、启动现场工作或新增外部validation模型评估。

## 复核与解释边界

B另写冻结输出核验代码，每臂核对2661个唯一键、21288个原始节点标签、40组MAPE、逐负载MAE、宏logMAE/RMSLE、选中epoch、逆变换输出、mask/age上下文、原截止及产物SHA，确认未通过筛选。没有在Windows加载模型、拟合统计或新增推理。原始标签来源是此前已独立核对的冻结R2 L0预测记录。

这支持本卡处理包不足以取得整体改善；没有证据表明单调变换已经补上突降先兆。T2存在指标权衡，不能用其MAPE改善或RMSLE下降就宣布所有误差变好。用户授权研究指标修订，当前结果仍按原口径与辅助口径并列；尚未另定正式阈值。

[执行卡](B19_DATA_TRANSFORM_EXECUTION_CARD.md) · [T1报告](../reports/data_transform_T1.json) · [T2报告](../reports/data_transform_T2.json) · [B独立复算](../reports/data_transform_B_verification.json)。完整checkpoint、输入状态、逐窗原/log预测、epoch、mask/age上下文和租约在本机.local/data_transform/T1、T2以及b507原job目录保留，交付包不重复放入原始大数据。
