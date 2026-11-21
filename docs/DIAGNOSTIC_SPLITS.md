# 诊断数据独立读入适配 v1

本候选针对Issue #3指定Release v2；不修改正常训练/正式验收读入器，不改源role和run_id。`diagnostic_split.read_release`校验已固定的ZIP、manifest及全部成员哈希，再调用纯函数`partition_rows`划分。无拟合、无归一化、无远端操作。

时间区间由recording_start确定，未依据误差或故障时间优化：训练[0,360)秒，开发验证[420,600)秒。每个分区保留历史全部位于自身区间的样本（五秒历史从目标前6秒开始），中间60秒不使用；标签须完整落入区间。源role仍为diagnostic，新增diagnostic_partition=train/validation仅描述本次探索划分。它不是独立实验轮验证。

实际验证Release结果：4532行联合特征中2693行训练、1301行验证，538行因边界/间隔被排除。先校验原始诊断标记、源run和窗口哈希、目标、有限非负张量、掩码/覆盖、因果时间和读写总量，再做划分；异常不静默丢弃。数值统计不表示训练已经执行。

接口：在工程目录执行以下代码，archive_path指向下载且保留的Release ZIP。

```python
from idfs_lab.diagnostic_split import read_release
splits, provenance = read_release(archive_path)
```

下一步为A准备显式诊断训练入口，复用现有训练主机限制，记录固定种子/预算和数据来源。节点与集群两种目标均保留；此模块先准备现有节点联合特征，尚未实现集群训练输入。不得将diagnostic_partition覆盖源role后传入正常训练器，也不得把两分区伪造为两个独立run。

该固定划分是第一次探索设计，后续若调参需登记重复选择次数。故障数据不能转为正常训练认证或正式准确率证据。源读取可用时间不构成在线到达及30ms发布证据。
