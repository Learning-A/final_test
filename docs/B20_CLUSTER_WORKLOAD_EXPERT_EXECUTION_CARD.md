# B20 集群总 IOPS 逐类负载专家对照执行卡

## 目的

在同一冻结 fit/internal-dev 样本上比较共享模型与逐类负载专家模型，预测集群总 IOPS 的未来 1、5、10 秒窗口平均值。神经 direct/fine 结果直接复用 B19，不重复训练。

## 固定输入与目标

- fit：4615；internal dev：2297；不读取外部 validation。
- 五类负载按已知 workload 路由：ai_inference_ses、ai_training_ses、bigdata_baleen、graph_zipf、hpc_wrf。
- 目标来自 `.local/cluster_horizon/prepared.npz` 的未来 10 个 1 秒集群轨迹；1/5/10 秒均为前 h 秒的均值。
- I1 C/B/P 特征按 `available_ns` 精确连接；C 为 2064 维，C+B+P 为 4144 维。
- 评价以每负载组为主，分别计算 MAPE、WAPE、MAE、RMSE、R²、Pearson；另列五负载等权宏均值和 pooled 值。pooled 值不替代逐负载结果。

## 对照

- C_shared_64：单共享 ExtraTrees，64 棵。
- C_workload_experts_64：五个专家，每个 64 棵。
- C_shared_capacity_320：单共享模型 320 棵，作为总树数容量对照。
- 对 C+B+P 复现上述三臂。

参数固定为 `max_depth=8`、`min_samples_leaf=3`、`random_state=2026`、原尺度 squared error、`n_jobs=4`。专家只使用训练集内同类样本，内部开发样本按其 workload 进入对应专家。

## 当前状态

本地已完成不拟合的输入/样本/哈希核验，见 `reports/cluster_workload_expert_verification.json`。Windows 没有 sklearn，且正式规则禁止 Windows 拟合，因此模型拟合必须在 b507 既有 Python 3.11/sklearn 环境中以 CPU4、workers0、新 job 目录执行；不得把本地环境错误当作实验失败或成功结果。

正式完成条件：保存每个配置逐样本 10 秒预测、模型/输入/样本 SHA、逐负载与宏指标、运行租约和正常退出状态；不消费外部 validation，不改变正式 MAPE 口径。
