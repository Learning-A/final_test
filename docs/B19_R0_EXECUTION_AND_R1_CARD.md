# B19 R0执行记录与R1首张输入卡

用户已授权按方案开始，任务#19=running；方案6e2a40a继续有效，旧research_only停止点由本次授权替代。现场操作与定时心跳继续暂停。A #18负责原始审计/交付与独立复核，B负责实现、核验和有界训练；双方通过#18/#19评论交互，不重复训练或创建重复请求。

## 已执行R0部分

- 已在#18/#19同步恢复授权，coordination_id=B19-RESUME-APPROVED-PLAN-20261122-01。
- 2026-11-22 22:25:52北京时间只读核对b507：既有10个B job目录未新增，无运行中的B训练，共享训练锁未被独占，GPU无计算进程。历史完成/失败记录保留，不获取训练租约。
- 已阅读A方案WORKLOAD_OPTIMIZATION_ROADMAP.md@75ccd0d9b770e9b0276b322f1d9ae34ce5b144c8，与B方案在证据优先、近端输入、损失/容量/校准分开、常数收缩对照及正式门槛上相容。采纳其“首卡先两输入臂”建议：先F0/F0+F1，F2及压力组留后续登记卡，保持总预算上限。
- 当前尚未收到request_id=B19-FINE-SCHEMA-SAMPLE-20261122-01的实际可下载样例/manifest SHA。不能写verified，也不把“另一边准备好”写成已交付。

## 已实现，不拟合

`idfs_lab.fine_inputs`提供两个固定schema：

- `causal_fine20_mask_age_v1`：目标前2秒的20个名义100ms槽，每槽read/write/total count、observed mask、age，加3/5/10/20槽的有效计数和/覆盖/有效零比例，总114维。缺失/迟到有mask，不补标签；不将不完整槽总量乘比例伪造一秒IOPS。节点严格12 OSS完整，客户端作为独立scope使用。
- `causal_long60_mask_age_v1`：可选60个完整秒，最近仍为目标前2秒开始的窗口，每秒读写总量/mask/age，5/15/60摘要，总321维。不新增60秒必须完整条件，不跨run/分区历史；此组件先准备，不加入首卡。

截止分两层：发布D=T−30ms；输入截止C=D−processing_budget。仅消费事件结束及available+时钟不确定性均≤C的桶。默认processing_budget=0、时钟未知、source_read_end证据只支持源时钟离线候选；`require_online=True`还要求明确线上接收、已知时钟界及非零后续处理预算。预算是调用者提供的测量依据，不因非零数字就宣称现场通过。输入函数不证明到达来源真实、不认证现场时延。

`scripts/verify_existing_fine_delivery.py`只读核对已解包目录，不下载/解压、不采集/训练。核验外部固定manifest/windows SHA、原dataset/run/role、文件哈希/行数、8节点身份、事件10有效桶的读/写/总量与原一秒标签精确一致；复算事件集群总量。验证返回`status=verified`的scope严格为哈希/身份/指定事件标签，不代表线上接收/时钟/辅助字段单位认证，不能凭返回值直接授权训练。辅助raw邻域包含目标后部分时永远diagnostic_only。

调用形式（实际交付后填入A确认的值，不从待验包自生成预期SHA）：

```powershell
python scripts/verify_existing_fine_delivery.py --directory $sample --manifest-sha256 $expected_manifest_sha --windows $original_windows --windows-sha256 $expected_windows_sha --dataset-id $dataset --run-id $run --source-role validation --hosts idfs44 idfs45 idfs46 idfs47 idfs64 idfs65 idfs66 idfs67 --event-start-ns 1795346566000000000 --event-start-ns 1795346652000000000 --event-start-ns 1795346870000000000
```

接收器适配既有export_manifest格式；如果A的小样例采用不同schema，先对齐显式字段映射和完整性范围，不能静默放宽角色/哈希/时间或把局部样例冒充全范围。

## R1执行卡：b19-causal-input-v2 / 第一步两臂

状态=waiting_data；本次没有创建job或模型结果。精确训练脚本/源码SHA、接收包SHA、运行命令及资源登记需数据通过后补齐，再由共享锁启动，不能直接运行本文。

|项目|预先固定内容|
|---|---|
|假设|相较现有秒级上下文，截止前可用100ms节点活动能改善下一秒预测，而非因删难样本或增加模型容量|
|数据|既有development-all-20-rounds-20261122-v1，15个train；01/02拟合、03内部留出，首5轮去重；不读取validation标签训练/选模|
|角色|Graph验证小样例只诊断；训练实验需要完整拟合/内部留出范围的细粒度数据，只有事件切片不能启动|
|标签|八节点各12OSS完整，完整8节点才有集群；read+write=total，无缺失补零/插值|
|两臂|F0现有base144，F0+F1增加各节点114维；scope分开，首卡不加client fine/压力/60秒，以免混杂|
|共同样本|先复现P2既有base144样本/预测；新F1全缺失也保留mask回退，同一5328/2661旧样本清单若原标签一致仍沿用，不因F1缺失删样本|
|基线复用|与源码e7f1db6、数据/样本/参数完全相同时复用已有base144模型和预测，服务器推理验证；不把复用算作新训练|
|模型|每目标ExtraTreesRegressor，64棵、max_depth8、min_samples_leaf3、random_state2026、n_jobs4；原尺度MSE，无早停/校准|
|集群|八节点各自输入，直接集群可拼接八节点F1并与原共享空间组组合；另报节点预测求和，不以集群替代节点|
|时间|分别记录源时间候选与线上证据；接收/时钟未知时不声称因果部署资格；证据无法建立所用时间界则停止该臂并回报A，不回填|
|预算|B授权b507新目录，4CPU/workers0；最多1800秒/任务；CPU拟合也持有既有共享flock完整生命周期，无GPU强制占用；不自动扩预算|
|选模|无参数搜索；报告40个节点×负载组、节点宏MAPE与最差组，MAE/WAPE/零/覆盖/低量和恢复窗口；直接/求和集群单列|
|晋级|按主方案预设研发门槛：节点宏MAPE相对改善至少5%，最差组与各负载节点宏MAE退化不超过相对5%；不满足则保留权衡，不立即调损失或加专家救分|
|停止|样例/全范围不齐、身份/标签不一致、单位/时间不明而无可核验假设、资源冲突/预算耗尽；不删有效低正值、不新增现场采集|
|产物|固定schema/样本/模型/数据/源码SHA、训练loss、逐窗口raw/final、CPU序列化对拍、全部9输出推理及特征构造耗时；A独立复核，非现场30ms证明|

## A接下来需要的动作

在原request_id下交小样例+来源/manifest及包SHA、Graph事件证据/单位/时钟限制；若缺现有日志，明确未知或不存在。B收到后回复verified/needs_fix及限定范围，再确认训练角色全范围批量；不要求A立刻导出全15轮、不重复审计。首卡代码准备可独立推进，但实际拟合等待门槛满足。
