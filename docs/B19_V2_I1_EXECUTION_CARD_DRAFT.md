# V2-I1：字节与压力输入对照草案

2026-11-23。状态：`completed_no_gain / do_not_repeat`。[实际结果](B19_V2_I1_RESULTS.md)已交付，以下保留执行前规格。C复用原模型预测，CB/CBP共16个新增森林已完成，不再次拟合。两组新增输入均未达到预登记门槛，后续固定C。

## 问题与准入

D0 显示下降首秒与 ordinary 阶段都有较大误差。末桶下降的简单规则误报较高，本卡只检验原始字节与快照压力是否有增量，不引入状态修正。B 已独立核对 D1 的三轮固定 400 ms raw 切片：1,344 对有效差分，2,304 个方向 bytes、1,152 个 op_wip 末值和 96 个完整节点桶。范围限于样例；延迟单位未知，latency 不进入 P。

执行前须接收 A 的节点级 pilot 并复算 12 OSS 聚合、核心 count/available 与既有 core 的一致性；再验收全 15 train manifest、每成员 SHA、拓扑顺序、字段 mask 和单位。所有原 5,328/2,661 秒键保持，辅助缺失不删样本。单位为服务端采集器的 bytes 与 operations，每操作 bytes 是代理；不硬乘副本或 EC 系数，不当应用请求大小真值。op_wip 为末端快照，不差分。

## 冻结特征设计

C=原 R1/R2 的 258 维 F0+F1，沿相同字段顺序、原节点顺序和输入构造源码。源时钟主假设为输入截止 S−80 ms、计划发布 S−30 ms，20 个绝对 100 ms 名义槽与原 fine_lag 一致。lag1 桶结束于 S，应被截止谓词屏蔽；其余槽也要实际源读取结束不晚于截止，不回填未来。每桶必须有完整 12 OSS，辅助方向有效性各自判断。缺失用零占位加 mask；真实零保留 observed=1。

| 组 | 每槽固定顺序 | 新增维数 | 节点输入总维数 |
| --- | --- | ---: | ---: |
| B | read_bytes、write_bytes、read_bytes_per_operation、write_bytes_per_operation 各一个值和一个 observed，再一个 byte_age_seconds；20槽 | 180 | C+B=438 |
| P | op_wip_sum、op_wip_max、op_wip_observed、op_wip_age_seconds；20槽 | 80 | C+B+P=518 |

方向比值由节点 sum(bytes)/sum(ops)重新计算，ops=0时比值占位0、observed=0，bytes本身可为真实0且有效。byte_age 只在至少一个方向有效时写 (S−30 ms−bucket_end)/1s；压力 age 仅在快照有效时写，其他情况为0并伴随对应 mask。两组使用同一源读取可用上界。原始桶和属性值均不在 Windows 上拟合归一化；树直接用原始数值。当前实现 `idfs_forecast_lab/idfs_lab/v2_auxiliary_inputs.py`，功能检查覆盖真实零、零 ops 比值缺失、晚到 1 ns、目标同秒桶、缺字段、重复键和非法 mask。

## 三臂和预算

| 臂 | 比较 | 需要新拟合 |
| --- | --- | --- |
| C | 原 R1 50 ms 的八个节点森林及冻结预测 | 0；完全一致时直接复用。第九个直接集群头不加入本卡拟合。 |
| C+B | 对 C 检验 bytes 结构增量 | 八个节点森林 |
| C+B+P | 对 C+B 检验已有 bytes 时的压力增量 | 八个节点森林 |

固定 ExtraTreesRegressor 64棵、max_depth=8、min_samples_leaf=3、seed2026、raw squared_error；其余构造参数逐项沿原 R1 实例并在实际配置中列出，不把库默认变化当成增量。所有节点和臂同 10 个拟合轮、5 个 train03 内部开发轮，C 与新增臂的样本、标签、列顺序和核心数值须复核。C+B+P 只回答条件增量；若字段组缺乏资格，在看分数前登记替换/跳过，最多三个配置，不追加 C+P 或另一 mask 变体。集群只报告八节点之和。

I1 全卡硬预算1,800秒，至多16个新增节点拟合；不把每节点预算相加到超出全卡。b507 授权新目录，CPU4、workers0；使用原门锁共享及 B UUID/owner 独占槽，CPU树不改 GPU 算法。唯一 job_id、实际源码/数据/schema/样本/模型哈希、原 C 模型清单和输出目录在启动前登记。输入前检、拟合、预测及产物写入受监督器总时限约束；失败保留，不自动扩大或重跑。

输入构造已用90.747秒，按91秒计；训练监督器固定余量1,709秒。唯一拟合任务为 `b19_v2_i1_aux_inputs_v1_a01`，入口 `scripts/run_v2_i1_study.py`，实际配置沿原R1所有默认值：criterion=squared_error、max_features=1.0、min_samples_split=2、bootstrap=False、min_weight_fraction_leaf=0、max_leaf_nodes=None、min_impurity_decrease=0、oob_score=False、warm_start=False、ccp_alpha=0、max_samples=None、monotonic_cst=None；模型与参数规格沿用 sklearn1.9.0。两臂均无归一化、损失搜索、校准或按负载再加权，保持与旧R1的 pooled 树拟合一致。全部拟合受继承至worker退出的既有资源锁约束，CUDA_VISIBLE_DEVICES为空。

## 报告与进入下一步

各臂报告原始预测、权重、40组 MAPE/MAE/WAPE/偏差/零标签/覆盖，逐负载与最差组，以及 D0 固定阶段和量级贡献。相对匹配基线宏 MAPE 改善至少5%、最差组和每负载节点平均 MAE 退化不超过5%、共同键覆盖不下降。保留所有分数，不据 train03 调字段窗口、seed或压力阈值。外部 development-validation 不读取，既有消费记录不重置；不宣称线上资格或正式达标。

有增量的最小输入才交给 M0–M3；原258维与优化器/选模规则完全相同的 M0 直接复用 R2 L0。若树输入卡未见贡献，仍可按 D0 对 ordinary 偏差的证据检验固定 C 的历史尺度/残差，须另行登记 M1，不自动扩维或搜索损失。
