# 冻结MoE组贡献诊断卡

唯一experiment_id=frozen-moe-group-diagnostics-v1。沿用Issue17/PR16及B专属新job frozen_moe_group_diagnostics_v1_a01，不启动训练、不增加候选选模。仅b507四线程CPU推理、workers0、1800秒上限，不占GPU。

固定现有masked_multigroup_moe_v1_a02中的moe_log/moe_combo checkpoint（按原报告哈希校验），不改变权重、训练归一化、样本、输出或校准。对各模型比较full、without_short、without_long、without_spatial：删除该组的专家输出通路，并将剩余组原门控重新归一化；输入/其他门控保持原值。该扰动不是重训消融，也不是因果特征重要性，不能据此换模型宣布提高。

helper full必须与原CPU TorchScript逐窗口对拍<=1e-10。762可用开发秒及697共同窗口均报告八节点/直接和求和集群MAPE/MAE/WAPE/零值和逐窗口预测；记录所有变体，禁止取开发集最好删除策略发布。本次只定位HPC退化和专家协作问题，帮助设计下一独立有界实验。

单项新测试用非零输出头核验无删除回归一致、被删除门控为零/剩余归一及有限输出。数据仍为首轮同轮分区、边界pending，无独立/正式/现场通过声明。源采集不扩增；独立轮/时钟到达证据请求不重复创建。
