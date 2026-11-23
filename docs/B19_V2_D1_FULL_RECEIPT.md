# V2-D1 全量节点辅助输入回执

2026-11-23。状态：`verified_all15_input_preflight / I1_ready_to_register`。A 交付 [D1 全15轮包](https://github.com/Learning-A/final_test/releases/tag/v2-d1-auxiliary-train15-20261123-v1)，manifest SHA 为 `6a0847f7bb03a2740a15d028c382845dfd98b5167e6beef20caa3592d985472b`。本机按同一完成版成员验收，不等待重复下载 Release。

B 核对了 15 个成员 SHA、原15train身份、八节点/12 OSS与逐字段 mask，并与旧固定 core 对齐 **791,680 个原有效节点100ms桶**的 read_count、write_count 和 available_epoch_ns；无新增或丢失有效核心桶。原 fine 中为录制空档列出的 invalid 网格，与新辅助包缺行在取特征时同为不可用；允许这两种无效表示差异，禁止把任何有效桶丢掉或变成有效。Graph pilot 曾在严格要求无效网格也完全同表时中止，诊断出149个 core-only invalid和5个aux-only invalid；该差异和修正检查范围保留于 pilot 报告，没有放宽有效桶/标签规则。

Graph train01另核对52,808个节点辅助行和32个已验证 raw 样例节点聚合，所有 bytes、ops后求比及op_wip sum/max一致。三轮样例原raw复算范围见[样例回执](B19_V2_D1_SAMPLE_RECEIPT.md)。A 声明导出器已重读核验全部120份OSS raw；B 本次自行核对的是15个辅助成员及旧core，不冒称又哈希了120份完整raw。latency单位未证实，继续排除；无在线接收资格声明。

随后构建原5,328拟合秒/2,661内部开发秒的固定 C、B、P 张量，尺寸分别为258、180、80。节点 C+B=438维、C+B+P=518维。字段顺序、原 core 源码/样本/标签/掩码与50ms源时钟假设全部复核；缺辅助字段不删键、不拟合统计。数组压缩包SHA为 `ffc01763d54e6185eb62772769e489eed63b6eef817bf79171adfbbb24d41170`。耗时90.747秒，原拟议I1总1,800秒据实按91秒记账，后续拟合监督硬上限为1,709秒，保持总上限。

[输入前检及schema](../reports/v2_i1_input_preflight.json)、[Graph pilot复核](../reports/v2_d1_graph_node_pilot_B_verification.json)。纯特征构造在本机完成，没有归一化拟合、模型拟合或模型推理；树拟合仅在 b507 授权新任务下进行。C 复用原R1预测，无新增拟合；C+B和C+B+P各八个森林，合计16次。参数、源码和数据哈希在 I1 启动前登记。
