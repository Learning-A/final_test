# 诊断数据独立读入适配 v1

本候选针对Issue #3指定Release v2；不修改正常训练/正式验收读入器，不改源role和run_id。`diagnostic_split.read_release`校验已固定的ZIP、manifest及全部成员哈希，再调用纯函数`partition_rows`划分。无拟合、无归一化、无远端操作。

时间区间由recording_start确定，未依据误差或故障时间优化：训练[0,360)秒，开发验证[420,600)秒。每个分区保留历史全部位于自身区间的样本（五秒历史从目标前6秒开始），中间60秒不使用；标签须完整落入区间。源role仍为diagnostic，新增diagnostic_partition=train/validation仅描述本次探索划分。它不是独立实验轮验证。

实际验证Release结果：4532行联合特征中2693行训练、1301行验证，538行因边界/间隔被排除。先校验原始诊断标记、源run和窗口哈希、目标、有限非负张量、掩码/覆盖、因果时间和读写总量，再做划分；异常不静默丢弃。数值统计不表示训练已经执行。

接口：在工程目录执行以下代码，archive_path指向下载且保留的Release ZIP。

```python
from idfs_lab.diagnostic_split import read_release
splits, provenance = read_release(archive_path)
```

`diagnostic_training`已提供显式诊断入口，复用现有b507主机限制和序列训练器。节点与集群均用八节点特征同时完整的共同时间戳：各节点取历史总量，集群对八节点历史和标签分别求和。新输入schema为diagnostic_total_lags_v1，五个总IOPS按新到旧排序再加均值，共6维，不能把权重交给415维线上入口。不得将diagnostic_partition覆盖源role后传入正常训练器，也不得把两分区伪造为两个独立run。

## A端可复现请求

固定配置：config/diagnostic_sequence_v1.json；初次诊断选择1次，不按本轮结果更改切分。先执行idfs44和cluster两个pilot，再由B复核曲线/尺度决定是否执行剩余七节点。初次GRU，40 epoch，早停12，seed2026，训练log1p目标均值/标准差只来自训练分区，SmoothL1损失，不校准。TCN入口可用但未纳入首批执行。

仅A在b507上选定启动时空闲的一张GPU，登记唯一job_id、当前代码提交和1800秒/任务预算。设置IDFS_GPU_JOB_ROOT为/home/lmj/idfs_forecast_lab下新的lmj所属任务目录、CUDA_VISIBLE_DEVICES为单GPU UUID，OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS均4，DataLoader workers=0。每个目标使用独立新进程、新任务目录；目录已存在时拒绝，不操作其他PID。A的现有隔离提交流程负责空闲检查和超时，本入口自身不连接或调度服务器。

在候选提交的工程目录执行（archive为同一Release ZIP，output须位于对应新任务目录之下）：

```text
python -m idfs_lab.diagnostic_training --archive <archive> --output <job_root>/idfs44 --target idfs44 --model gru --epochs 40
python -m idfs_lab.diagnostic_training --archive <archive> --output <another_job_root>/cluster --target cluster --model gru --epochs 40
```

依赖：已有Python3.10+与PyTorch/CUDA；不自动安装或更改既有环境。输出model.ts、diagnostic_model_manifest.json、validation_predictions.json，含每epoch训练/验证损失、归一化尺度、逐秒预测、同样本last/mean基线、正式零排除口径MAPE及MAE/WAPE、CPU/GPU导出对拍、数据/权重/代码标识。A补充GPU、运行环境、job_id、墙钟耗时、文件SHA256；CPU延迟和线上30ms仍需单独测，导出对拍通过不代表发布达标。

功能检查只运行纯数据适配和主机拒绝路径，本机未拟合；训练请求与实际执行状态在唯一GitHub issue中维护。模型输出尺度失败、近零预测和低IOPS误差都是首批要检验的失败场景；本版本先取得可解释神经基线，后续再按曲线设计损失/校准消融，不盲目扩大架构搜索。

该固定划分是第一次探索设计，后续若调参需登记重复选择次数。故障数据不能转为正常训练认证或正式准确率证据。源读取可用时间不构成在线到达及30ms发布证据。

实际共同时间戳为每个目标271训练、125验证；不同于PR4全600秒回放451秒，禁止直接比较两份报告指标。正常训练/验证轮仍为0。78项本地工程检查全部通过，无本机拟合。
