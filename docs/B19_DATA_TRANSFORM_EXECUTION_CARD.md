# B19 数据变换有限对照执行卡

2026-11-23，评分前登记。授权为 A18-DATA-TRANSFORM-EXECUTE-20261123-01、A18-DATA-TRANSFORM-CONTRACT-20261123-01。旧R1/R2/V2全部保持do_not_repeat，新卡不改变原标签和已有判定。

## 冻结范围

固定C258原始物理特征、源截止S−80ms、5328/2661原sample_ids、train01/02拟合与train03内部留出、八节点顺序和seed2026。T0精确读取原R2 L0冻结预测及预处理，不拟合。T1/T2/T3每个唯一新job上限1800秒，含参数拟合/构造/训练/导出；至多3配置5400秒，不借用其他卡余额，不自动重试或续训。无新增外部validation模型评估或现场操作。

输入数组、样本、metadata、R2报告、原预处理和基线从已核验M3输入目录复制，以每文件SHA固定。数值环境匹配R2：Python3.11、torch2.4.1、numpy1.26.4，复用b507既有环境。

## 输入与训练统计

精确列名/索引/有效观测映射由源码 `transform_contract()` 和配置输出固化，并与A的data_transform_A_column_contract.json逐项比较：181连续列中，178个非负物理连续列变换，signed差分30/31/32保留原符号原值后只robust；77个mask、fraction、age列保持原值。IOPS/操作计数沿用各列原单位reference=1，当前C258没有bytes列。

T1对178列用ln(1+x)，随后每连续列仅以10个拟合轮的有效观测估计median、IQR，quantile method=linear。真实零参与，masked占位排出统计；变换后masked缺失恢复0。IQR=0固定scale=1；全缺失固定center=0/scale=1，非负变换仍按其固定函数应用于后来有效观测。signed列不log、不clip。统计代码首先拒绝非b507主机。

T2精确复用T1的transform.json，不再拟合输入统计。T3条件只根据T1训练分布诊断，不读取T1/T2开发分数：178列中至少18列同时满足有效观测>=1000、不同原值>=100、log1p后绝对三阶标准矩偏度>2，才准入。满足时Yeo–Johnson替换T1的全部178列路径，其余设置匹配T1；非恒定列lambda由scipy.stats.yeojohnson_normmax在拟合观测上一次估计，恒定/全缺失列lambda=1；再以相同训练有效值median/IQR缩放。门槛未满足则记录skip，不换其它变换、不扫lambda。

## 输出、初始化与选择

三臂网络均为R2的21089参数节点共享MLP（258+4→64→64→1），相同隐藏初始化、batch排列、训练run/node等权、AdamW lr=.001/decay=.0001、batch128、clip5、最多60epoch/patience8。

T1/T3输出s*softplus(h)，固定原R2训练s，loss原L0缩放MAE。T2输出z=log1p(s)*softplus(h)，原IOPS pred=expm1(z)，loss=训练run/node等权的abs(z−log1p(y))；无反变换bias修正。各臂head零权重、bias=log(expm1(1))，epoch0恢复原s（浮点容差1e−6相对）。T2是输出参数化+log损失包，不把结果单独归因某个变化。z>80或非有限值直接失败，不clip预测或修改标签。

所有臂按原40组节点宏MAPE选checkpoint，严格较小才更新，同分保留更早轮（含epoch0）。保留每次内部评分。原研究5%相对改善/最差组<=1.05/每负载平均节点MAE<=1.05门槛保持。logMAE/RMSLE作为辅助，各40组先计算再宏平均，含实际零；原MAPE只排真实零。

## 输出与资源

保存原始/最终IOPS预测、target_z/prediction_z、全部epoch、checkpoint、transform.json、有效观测数/lambda/center/scale、训练分布诊断及哈希。每个开发窗的77 preserved列（含mask/age）与源可用时间另存input_context.npz，与sample_ids及C原数组共同恢复输入。

仅b507授权新目录，CPU4/workers0，B卡GPU-ac63ac15-888b-c3fb-c42f-eb1cb26487c2，A卡保持GPU-8923f552-5691-b49b-d2c8-87cb9292ae92。原共享门锁→UUID→owner全生命周期锁，启动前核查作业/锁/空闲。job_id分别b19_data_transform_t1_v1_a01、t2、t3；不创建已存在目录。

真实推理接收、跨机时钟界和实际发布仍未知，formal_acceptance/online_time_eligibility维持false。A独立复核，不重复拟合。
