# 后续优化执行计划（2026-11-22用户确认后）

恢复来源work/idfs-b-state.json；沿用Issue17、候选分支/PR16。目标仍是八节点未来完整一秒及集群候选，正式100-MAPE/只排零/严格>90%不变。业务容限仅辅助曲线，未经用户另行确认不改验收。

|阶段|执行内容|证据与进入下一阶段条件|
|---|---|---|
|P0 固定证据|保留首轮来源/哈希/切分/共同样本；低值与突变诊断；MAPE/MAE/WAPE、容限曲线分节点分负载|不给容限预设“通过”含义；源边界/独立轮pending不掩盖|
|P1 旧强基线适配（当前执行）|读取旧高分npz实际cfg，复用原models.py核心；GRU训练全局归一、GRU窗口归一、TCN窗口归一，先固定单seed有界pilot|源码/配置快照和哈希、全部逐窗口预测、旧WAPE与新MAPE选checkpoint双记录、CPU/GPU对拍；失败保留，不能自动扩预算|
|P2 输入补强|A导出现有100ms有效桶/到达证据，审计现有客户端/OSS/系统辅助字段；比较近端细粒度与秒级、长历史mask|完整标签不变、共同交集和全覆盖分别评估；每组字段有单位/身份/可用时间，不由B启动采集|
|P3 单因素损失|固定P1/P2合理骨干、输入、seed与样本，MAE/原尺度Huber/log-Huber/相对Huber有界对照；组合项逐个引入|训练floor/尺度只训练拟合；保留低正值正式MAPE；若只换指标变好、或跨负载明显退化，不晋级|
|P4 集成及因果校准|仅合理候选再做固定多seed集成，校准按已到达历史结算更新、每run重置|raw/final、迟到回放、有无校准对照；不照搬旧按数组索引更新|
|P5 独立验证/冻结|A交接独立开发轮后验证五负载/八节点；冻结哈希后A现场发布与正式轮|缺独立轮不能宣布正式通过；现场30ms不能以CPU单模型耗时替代|

先完成P1再决定扩展模型，暂不追加新MoE结构或大范围权重搜索。无新数据可继续准备P2/P3代码；已有completed测试不周期性重跑。每阶段小任务新目录、单空闲GPU、CPU4/workers0、1800秒，GPU共享lease/Issue登记不变。仅在实质结果/失败/需处理时通知。

## P1固定实验卡：legacy-core-current1s-pilot-v1

新job legacy_core_current1s_v1_a01，b507 B专属根，源码提交后登记。五类首轮ZIP f28fc45ba4e6d80c80da815b06d4e05573a7e9c788cbe199e0a1f01898b40b49，严格15秒共同样本1215train/697development，原同轮分区，不伪称独立轮。按时间先后排列15个可用总量秒，末秒为目标前2秒；八节点分别与独立集群共九目标，不补缺失。

旧core来源/home/lmj/dl_forecast/models.py，SHA9510dfb820d770394a91b4d18d7ac29974a6db7303f64cdc9b17d9111eec2685；任务内复制原文件并核验，旧工程不改。已核对旧preds_final的gru_t2(_norevin)/tcn_t2实际cfg：hidden128/layers2，TCN channels32/levels6，dropout0.1，lr0.001，loss=mae（不是默认common训练器的SmoothL1）。

三组：gru_global_mae、gru_window_mae、tcn_window_mae。直接复用原GRUNet/TCNNet；全局mean/std仅当前训练标签拟合，窗口归一mean/var只输入窗口。AdamW wd1e-4，batch128，grad5，seed2026，最多40epoch/目标，旧式WAPE早停patience10、scheduler factor0.5/patience4。同一轨迹同时记录/保存WAPE最佳和正值MAPE最佳checkpoint并评估，不额外重训；两者都不替代正式MAPE。

这是旧核心/归一/优化器的秒级适配pilot，不是完整历史高分复现：旧L256×100ms=25.6s，当前L15×1s=15s；时距、采集对象、切分、预算均不同，旧三seed集成/校准尚未复现。报告须逐条保留差异；当前缺100ms不阻止先验证旧核心，但不能以pilot失败否定旧模型能力。

预期产物：源文件/旧cfg/环境/命令/数据/模型哈希、每epoch loss/WAPE/MAPE、双checkpoint、分节点/负载/集群正值MAPE/MAE/WAPE/零值、逐窗口raw/final、CPU/GPU对拍与CPU延迟、独立复算。记录新增3训练配置及双checkpoint重复选择，累计从13更新，不混同173个旧epoch评价。若超1800秒失败保留已完成进度，不自动加时。
