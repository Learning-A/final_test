# 诊断GRU输入尺度单因素对照 v2

依据已复核a03结果（Issue #6，DIAGNOSTIC_PILOT_REVIEW.md），idfs44/cluster预测动态范围过窄。训练输入只取log1p而未中心化，可能造成GRU门对高正输入不敏感；这是待实验检验的假设。

仅开启训练分区输入log历史的全局均值/标准差归一化，标准差下限0.25。不改架构、seed2026、log目标归一化、SmoothL1损失、40 epoch/早停12、切分或无校准设置。均值/标准差只由A在训练时从271个训练样本的五个历史值计算，完全不读取验证值；随TorchScript推理变换封装。旧调用默认normalize_inputs=False，保持现有行为。

源joint_context_v1不变，新候选输入schema标记diagnostic_total_lags_v2，仍接收原始IOPS六维输入；模型内部处理尺度，不要求推理端另拟合或重复归一化。不能混用旧模型或415维线上入口。manifest记录normalize_inputs、input_transform、input_log_mean/input_log_std和训练分区来源。正式目标、协议和指标不变。

## A的执行请求

固定配置config/diagnostic_sequence_inputnorm_v2.json；同一Release v2 ZIP和manifest哈希、原run_id/diagnostic/fault标记全部保留；训练[0,360)、开发验证[420,600)、60秒间隔，共同时间戳每目标271训练/125验证。首批只执行idfs44、cluster两个独立新任务，不重跑v1，也不扩展其他七节点。相同开发切分上的第二次预先规定的神经设置比较；最终泛化仍需独立轮次。

在A既有隔离提交流程中，核对启动时空闲的一张GPU，限制4 CPU线程、workers0、1800秒/任务，无依赖安装，不操作其他PID。在新的lmj所属/home/lmj/idfs_forecast_lab子任务目录设置IDFS_GPU_JOB_ROOT及单GPU CUDA_VISIBLE_DEVICES；OMP/MKL/OPENBLAS线程均4。在固定候选提交的工程目录运行：

```text
python -m idfs_lab.diagnostic_training --archive <Release v2 ZIP> --output <fresh_job_root>/idfs44 --target idfs44 --model gru --epochs 40 --normalize-inputs
```

另起新进程、新job root执行cluster。A登记job_id和提交SHA，失败保留目录。所有实际拟合只由A执行；B的本机检查仅固定参数的输入变换、CPU trace和主机拒绝路径，未拟合归一化或模型。

交付同v1的逐秒预测、曲线、权重/配置/数据哈希、环境/GPU/耗时及CPU/GPU对拍，额外复核input scaler来源与预测动态范围。按固定实际值段复算MAPE贡献，比较同样本MAE/MAPE及峰谷响应。单独一个指标改善不足以证明输入门饱和假设；无改善也保留失败证据，不能删掉低IOPS窗口。GPU/CPU对拍仍用1e-5门槛，不代表30ms线上发布通过。

此请求不做损失或校准扫参。下一步由结果决定是否需要残差结构/目标对齐损失；不从125验证窗口拟合校准系数。原Issue #6已完成，新请求以唯一v2标识登记，避免重复执行。
