# V2-I1 字节与压力输入实际结果

2026-11-23。状态：`completed_no_gain / do_not_repeat`。两组新输入都没有达到预登记的研发筛选门槛，后续保持258维C，不继续扩B/P或补扫C+P。结论只覆盖本卡特征和固定树，不等于原始字节或压力信息在所有模型下均无价值。

## 固定范围与结果

执行前登记见 [#19](https://github.com/Learning-A/final_test/issues/19#issuecomment-7090644625)。唯一作业 `b19_v2_i1_aux_inputs_v1_a01`，worker源码 `adae75ab0d0895f4e08852d4812bc1efb38fd545`。同原5,328拟合秒、2,661内部train03秒、八节点和50ms源时钟假设。C复用R1冻结预测，没有再拟合；CB与CBP各拟合八个森林，共16个，均采用原ExtraTrees64/depth8/leaf3/seed2026/raw MSE及相同库设置。

| 输入 | 节点维数 | 节点宏 MAPE | 最差组 MAPE | 对 C 相对改善 | 达到MAPE<10%的组 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C，原 F0+F1 | 258 | 108.0662% | 322.5187% | — | 0/40 |
| C+B，bytes/ops代理 | 438 | 107.7635% | 324.1433% | 0.2802% | 0/40 |
| C+B+P，增加op_wip快照 | 518 | 107.9214% | 320.7244% | 0.1340% | 0/40 |

CBP 相对 CB 宏 MAPE 退化0.1466%。CB 各负载平均节点 MAE 比 C 下降约0.16%–1.59%；CBP 比 C 下降约0.07%–1.65%。这些小幅变化未达到至少5%宏 MAPE改善的门槛；最差组/各负载MAE约束不能替代主门槛。全部结果保留，没有选另一档余量、seed、阈值或指标救分。

D0阶段贡献中，C、CB、CBP 的 ordinary 宏 MAPE贡献分别为33.0881、32.8106、32.8236个百分点；下降首秒分别为74.5285、74.5036、74.6387个百分点。新增输入对本卡困难下降首秒几乎没有帮助，也未明显解决 ordinary 误差。各阶段贡献按原40组正值分母计算，不是层内分数。

## 核验与资源

B独立从原R1包核对C预测精确一致及21,288个原节点标签，逐行检查三臂全部7,983条预测（63,864个节点预测标签）、无重复/缺键/改标签/后处理，并用另一套NumPy矩阵聚合核对每臂40组宏/最差MAPE、逐负载MAE、量级/阶段贡献与三个晋级判定。正式MAPE继续只排真实零，无新增外部validation评估。功能检查158项，153通过、5个Linux专用跳过、0失败；实际诊断不能由功能测试替代。A已完成独立复核，记录见 [A的I1报告](https://github.com/Learning-A/final_test/blob/experiment/postcapture-collaboration/reports/v2_i1_A_independent_review.json)。

拟合仅在b507授权新目录执行，CPU4、workers0、CUDA不可见；原门锁共享+B UUID/owner独占锁覆盖整个worker。worker用时40.211秒，租约41.479秒。输入纯特征构造90.747秒按91秒记账，剩余worker硬预算1,709秒；按记账加租约约132.479秒，未超原全卡1,800秒。两个GPU没有用于本卡，A/B不同卡可并行的规则保持；没有现场操作或环境安装。

## 交付与下一步

[完整包](https://github.com/Learning-A/final_test/releases/tag/b19-v2-i1-aux-inputs-20261123-v1) SHA256：`77d96a5c8bf033f0237ecaac0de095794a6d05aa9dcb0485e90a9e57e86b3131`。包含16个权重、所有预测、精确预构造输入数组、样本清单、源码、配置和已完成租约；29个manifest成员均已核对。包内报告 SHA为 `42b8a9f0173af5badfd64023d238193e1763e2d910159c1a67f88389fdfaffe4`。权重没有在Windows反序列化或再拟合。

[训练报告](../reports/v2_i1_input_study.json)、[独立指标复核及分层](../reports/v2_i1_result_verification.json)、[输入前检](../reports/v2_i1_input_preflight.json)、[执行卡](B19_V2_I1_EXECUTION_CARD_DRAFT.md)。未知逐轮跨主机时钟误差、推理服务接收和真实发布仍未知，不能因源输入满足截止便宣称线上资格。

后续按V2进入固定C的M1历史局部尺度/残差对照，M0完全匹配时复用R2 L0，损失仍固定缩放MAE。先冻结历史尺度、基线、非负恢复、零历史fallback、输入mask与匹配比较范围，再登记新配置；不能把M1整套变换的效果单独归因于尺度。全局/动态状态路线仍须满足独立校准与误报约束，不据三个低值例或一次输入卡自动启动。原R1/R2及I1均标记不重跑。
