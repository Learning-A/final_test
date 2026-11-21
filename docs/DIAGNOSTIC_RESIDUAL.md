# 显式历史残差输出对照 v3

v1预测接近全局常数，v2输入归一化扩大范围却使两个目标MAE变差。下一候选检验：用最近可用历史作为结构先验，再学习log修正，能否比全局均值中心的绝对输出更有效保留峰谷。

训练目标归一化、原log-SmoothL1损失和验证早停口径完全保持v1；只换输出参数化及其对应的零修正初始化：

```text
z = head(GRU(log1p(history))) + (log1p(newest_available_IOPS) - train_target_log_mean) / train_target_log_std
prediction = max(0, expm1(z * train_target_log_std + train_target_log_mean))
```

head权重/偏置初始化为0，训练前预测精确为last-available，包含零值。后续学习log修正，不是拟合后再混合基线。这个先验不保证预测突发下降，可能放大历史滞后；12 IOPS异常窗口前历史均为3300—3500，仍需保留并单独诊断。

不同时启用输入归一化、校准或新损失；当前main尚未合入PR7，本分支从origin/main c1ccc3f建立，保持v1输入。PR7为负结果实验，合并前需A注意两个候选会修改同一训练器签名，不能整文件覆盖后丢失任一已保留开关。请求执行固定提交，不依赖哪个PR先合入。原residual_output=False调用不变，正式协议、八节点正式目标及零排除口径不改。

新schema diagnostic_total_lags_v3，外部仍是六维原始IOPS输入；模型内部封装历史skip，不能部署到415维线上服务。manifest记录output_structure与head_initialization。不能用v2输入尺度权重或额外预处理替代v3。

## A端复现

配置config/diagnostic_sequence_residual_v3.json。数据、ZIP/manifest哈希及源role/run/fault完全同Issue #6/#8。固定recording_start训练[0,360)、验证[420,600)，60秒间隔并清除跨边界历史；每目标271/125共同秒，单次故障采集，不是独立验证。第三次预先登记的神经设置比较，不因本次误差再改切分。

只做idfs44和cluster两个独立新任务，对照已有v1 a03，不重跑v1/v2，不扩展七节点。GRU32、seed2026、40epoch、早停12、原目标训练尺度/损失/优化器不变，无校准。A选启动时空闲单GPU，4CPU线程、workers0、1800秒/任务，无依赖安装，不操作其他PID。在lmj@b507的项目新任务目录设置IDFS_GPU_JOB_ROOT与CUDA_VISIBLE_DEVICES，OMP/MKL/OPENBLAS线程均4，从固定候选工程目录运行：

```text
python -m idfs_lab.diagnostic_training --archive <same Release v2 ZIP> --output <new_job_root>/idfs44 --target idfs44 --model gru --epochs 40 --residual-output
```

另起新进程/目录改为cluster。仅A执行；B不拟合模型或尺度。A登记唯一job_id、实际代码SHA及环境，失败保留原因。

返回权重/数据/配置哈希、manifest、逐窗口预测、曲线、同样本last/mean指标、输出范围、分实际值段误差及GPU/CPU对拍。关键判断：相较原v1能否改善预测峰谷且不恶化MAPE；若只改善某一项仍保留多指标差异。CPU对拍门槛1e-5不代表30ms现场发布，正式三轮不接触。固定常数的零修正与trace测试仅证明结构算式和序列化，不证明训练成功。
