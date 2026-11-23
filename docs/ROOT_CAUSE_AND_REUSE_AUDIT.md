# 当前误差与历史高分复用审计

2026-11-22，响应用户追问。当前结果确实不足，不能仅用指标不同解释。已经做了低值/突变、输入覆盖、损失权衡和专家组诊断，但未完成历史强流水线在当前数据上的完整复现。

## 已核对的旧高分

GPU /home/lmj/dl_forecast/final_results.csv（528条历史结果）及common.py只读核验：AI推理gru_t2_norevin/ens3的acc_wape=0.9329189，acc_dlt=0.8281193，masked_frac=0.0737798；HPC WRF/IOR gru_t2/ens3的acc_wape=0.9151171，acc_dlt=0.6770904，masked_frac=0.2184532。acc_wape定义为1-WAPE；acc_dlt为过滤低值后1-相对RMSE，不是当前零值之外全部保留的100-MAPE。旧主协议0.1秒网格，arrival/concurrency目标、暖机后70/30同序列划分。低值过滤只用于acc_dlt/pred20等字段，不能误写成acc_wape也过滤同一批低值。

Learning-A/models README中graph_osd FourierGNN=0.988200及SYSU OSD ExtraTrees+校准=0.904108是R²。其evaluation/model_inventory还记录active-only OSS和固定历史轮划分；不能把所有旧工作都说成同一种切分或存在泄漏。旧结果说明模型资产值得复用，尚不是当前八节点未来完整一秒、MAPE<10%、独立轮/发布时限的直接通过证据。

来源：[README](https://github.com/Learning-A/models/blob/main/README.md)、[旧评测说明](https://github.com/Learning-A/models/blob/main/reports/evaluation_README.md)、[模型清单](https://github.com/Learning-A/models/blob/main/reports/model_inventory.md)、[特征定义](https://github.com/Learning-A/models/blob/main/data_specs/feature_definitions.md)、[校准实现](https://github.com/Learning-A/models/blob/main/src/storage_forecasting/calibration/online_residual.py)。GPU只读核对文本另存B工作区outputs/legacy_best_reaudit.txt。

## 当前误差证据

用最新moe_combo的762开发秒重算，按每负载实际集群IOPS最低floor(10%样本)做事后分层，AI推理15/157秒贡献90.63%总APE，最小集群12 IOPS。按目标实际值相对最近可用历史值变化超过50%定义事后跳变组（不是模型输入）：65秒MAPE723.95%，其余92秒29.84%。即使非跳变组也未达标，不能只归咎MAPE分母。

其他负载跳变/其余集群MAPE分别：AI训练106.96/29.21，大数据123.41/20.37，图125.16/12.73，HPC99.81/27.21%。这些分层仅做诊断，不删正式样本、不改变指标。原始分层结果见reports/current_error_diagnosis.json。

当前只用1秒处理窗口，最新输入区间是目标前2秒至前1秒，未利用更靠近发布时刻的100ms内变化。旧工程存在高频、EWMA、长lag和序列窗口资产，但当前交接包只有秒级窗口，不能重构同一次采集100ms内部变化；需A导出现有原始数据的有效桶，不要求B重新启动采集。这是明确的信息差，是否足以改善突变预测仍须实测。

质量方面618次拒绝中571是历史不完整，已以短历史+可选长mask扩展覆盖；但源available只是read-end，不能证明真实接收或时钟同步。边界证据未齐是待核风险，不是已经证明采集错误。每负载只有一轮，无法证明独立泛化；也不能据此断言差分数全由样本少造成。

模型方面最新多MoE是小型MLP专家处理短/长/空间特征，不是完整GRU/TCN/GAT/Fourier序列模型复现。组合损失只小幅改善宏MAPE并造成负载退化，推理组移除亦显示相互依赖，尚未证明复杂度是主要瓶颈。

## 已借鉴与尚未迁移

已使用：现有tsfm环境/工程规则、历史值/均值/Ridge/ExtraTrees基线、读写/多尺度/差分/空间思路、历史残差输出，并补因果时限和mask。历史故障集上的GRU/尺度/残差实验独立留痕，不能算新五类数据已完成GRU复现。

尚未在当前五类数据上完整复现：旧GRU/TCN强基线及多seed集成、按窗口RevIN归一化、LightGBM/HGB强表格模型、FourierGNN/GAT-GRU、因果事件驱动的在线残差校准、100ms及客户端/OSS内部上下文。旧权重只盘点，未证明schema/时距适配，不能声称已迁移。

## 工作优先级修正

先做历史强流水线复现和同指标审计，再继续增加新MoE/损失组合。固定现有数据/共同样本、只排零MAPE，先将GRU/TCN单模型及窗口归一化适配到当前144输入/原始序列，并保留旧协议旁路诊断，分清数据变化、信息时效、模型迁移和指标变化的贡献。复现通过后再考虑多seed集成及按实际历史结算到达更新的校准；不能照搬按行索引校准或低值过滤。

复现卡应预先写源码/权重/数据哈希、输入映射、时距、训练预算、选模指标和旧/新双口径诊断。缺少当前100ms数据时明确仅秒级适配，不等待所有辅助字段才能先完成代码与基线。独立开发轮、时钟/边界/接收证据继续使用现有Issue17请求，不重复开任务或采集。当前不追加无界调参，不宣称找到全部根因或保证达标。
