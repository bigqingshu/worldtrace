# UI 锚点支持率与方向生长离线实验

## 解决什么问题

现有实时实验会先在每一帧上判断边缘强度、方向和稳定性。未充能的半透明
终结技图标只有很弱的固定轮廓，证据可能在进入累计器前就被硬阈值删除。

第一轮实验保留每一帧的连续 RGB 梯度，先跨多帧累计二维结构张量，最后才
生成热力图和 Otsu 掩码。第二轮在此基础上增加：

1. 按 `motion_episode_count` 分组，每个独立运动阶段对同一像素最多投一票；
2. 从 `support_count / eligible_count` 得到支持率，不再把热力图亮度解释为
   概率；
3. 将高支持率像素记为强核心，将约 30% 重复出现的直接弱证据记为弱支持；
4. 只沿空间邻近且梯度方向相容的弱支持生长；
5. 分开输出只包含直接证据的紧框，以及带明确
   `HYPOTHESIS_ONLY` 来源的宽框。

它只验证“不同动态背景上是否出现相同的屏幕固定局部形状，并能否提出有界
框候选”，不确认 UI 状态、图标身份或可执行控件。

## 边界

- 独立离线实验，不改 WGC、实时 GUI、Trace Core、OCR、SAM 或候选存储
  契约。
- 视频按解码器报告的 PTS 毫秒采样；解码器没有有效时间戳时才退回
  `frame_index / average_fps`，使用情况写入 `summary.json`。
- 复用 `minimal_trace_gui` 的变化率、LK 光流和 RANSAC 世界运动门控。已知干净
  视频窗口只把“运动必须覆盖画布边缘”从实时默认的 `4/4` 显式降为 `2/4`：
  第一个窗口的相干运动最多只覆盖三边，原门禁会得到零样本；其余阈值不变，
  实时 GUI 默认值也不修改。
- 必须提供两个互不重叠、属于同一 UI 状态的干净时间窗。
- 技能开启、关闭等快速过渡不能混入干净窗口。
- 所有结果保持 `UNKNOWN`，需要人工查看两个窗口及其交集。
- 当前像素可观测性的最小定义是“有效运动阶段内整张分析画布可观测”；以后
  若引入遮挡或转场门禁，必须改变逐像素 `eligible_count`，不能悄悄改变
  支持率分母。
- `observed_core_mask`、`observed_weak_support_mask` 与
  `completion_hypothesis_mask` 分开保存。宽框的推测区域不能回写为直接
  观测证据。
- 圆形不是硬规则。宽高比只用于判断可能缺失的方向和有界补全范围，
  `circle_completion=false`。

## 当前测试命令

在 `worldtrace` 项目根目录执行：

```powershell
..\.venv\Scripts\python.exe -m experiments.ui_anchor_gradient_lab `
  --video "tests\YuanShen 2026-07-24 11-22-03-133.mp4" `
  --window "clean-a=0:10.25" `
  --window "clean-b=14.45:31.05" `
  --probe-roi "293,151,315,172" `
  --episode-vote-score-minimum 0.06 `
  --weak-support-ratio 0.30 `
  --core-support-ratio 0.55 `
  --maximum-completion-px 12 `
  --output-dir "runtime_data\ui_anchor_gradient_lab\yuanshen-q-uncharged-20260724-run10"
```

`--output-dir` 必须是尚不存在的新目录，避免覆盖之前的实验事实。

四个新增 CLI 参数只用于本实验的可复现调参：

- `--episode-vote-score-minimum`：单个运动阶段内，一个像素是否投票的连续
  梯度下限；
- `--weak-support-ratio`：直接弱支持率，默认 `0.30`；
- `--core-support-ratio`：强核心支持率，默认 `0.55`；
- `--maximum-completion-px`：宽框最多补全的分析画布像素数，不是 30% 扩框。

## 输出

```text
<run>/
├── summary.json
├── agreement_heatmap.png
├── agreement_mask.png
├── agreement_maps.npz
├── window_comparison.png
├── probe_comparison.png
├── candidate_overlay.png
├── candidate_masks.png
├── candidate_maps.npz
├── probe_candidate_overlay.png
├── clean-a/
│   ├── reference.png
│   ├── generalized_gradient_gray.png
│   ├── generalized_gradient_heatmap.png
│   ├── generalized_gradient_overlay.png
│   ├── coherence_heatmap.png
│   ├── otsu_mask.png
│   ├── episode_support_count.png
│   ├── episode_eligible_count.png
│   ├── support_ratio.png
│   ├── orientation_consistency.png
│   ├── core_mask.png
│   ├── weak_support_mask.png
│   ├── support_overlay.png
│   └── maps.npz
└── clean-b/
    └── ...
```

- `generalized_gradient_heatmap.png`：单窗口的广义梯度证据。
- `generalized_gradient_overlay.png`：热力图叠加到一个采样帧，仅供定位。
- `coherence_heatmap.png`：累计梯度的方向一致性，不单独作为候选。
- `agreement_heatmap.png`：两个窗口分别归一化后取逐像素最小值。
- `agreement_mask.png`：只在多帧累计结束后执行的 Otsu 对照掩码。
- `probe_comparison.png`：提供探针 ROI 时，以最近邻放大两个参考帧、两个窗口
  热图和交集热图；只改善可查看性，不参与计算。
- `episode_support_count.png`：一个像素被多少个独立运动阶段直接支持；
  PNG 只供查看，原始 `uint16` 保存在 NPZ。
- `support_ratio.png`：`support_count / eligible_count`，不是图标概率。
- `orientation_consistency.png`：跨阶段的双角梯度方向一致度。
- `core_mask.png` 与 `weak_support_mask.png`：互斥的直接观测层。
- `candidate_masks.png`：洋红为强核心，暗青为未接收弱支持，亮青为被方向
  生长接收的弱支持，黄色为仅供提框的推测区域。
- `candidate_overlay.png`：全画布候选；绿色框是紧框，橙色框是宽框。
- `probe_candidate_overlay.png`：只使用探针内部强核心作为诊断种子，但不修改
  支持率或全局候选结果。它用于放大查看“框如何从局部证据生长”，不是人工
  标注真值。
- `candidate_maps.npz`：未经 PNG 量化的计数、支持率、强/弱/生长/推测掩码。
- 各窗口 `maps.npz`：保留第一轮梯度张量，同时增加本窗口逐阶段支持证据。
- `summary.json`：采样时间、运动门控统计、探针 ROI 指标和明确限制。
  同时记录源视频 SHA-256、修改时间、解码后端、OpenCV 版本、PTS/回退计数、
  每个运动阶段的入选帧、候选紧/宽框及来源、有效运动样本的实际时间覆盖，
  以及每个产物的 SHA-256。

`agreement_heatmap.png` 当前只是两个窗口分别归一化后逐像素取最小值，尚未
核验梯度方向，也没有消除动态背景。它只能称为“跨窗口共享响应”，不能称为
已确认的同一图标轮廓。

## 第一轮历史验收

1. 两个独立窗口都能在右下角显示终结技图标的外圆或内部图形证据。
2. `agreement_heatmap.png` 中的中心和尺寸基本一致。
3. 不能只抓到 `Q` 键位框或旁边的青色装饰。
4. 不能把右侧整列 HUD 合并成一个大候选。
5. 第一轮失败时先保留产物；第二轮只增加逐阶段支持率与有界框，不接入
   OCR、SAM 或背景运动补偿。

## 第二轮数据约束

- 相邻帧不是独立票；长运动阶段无论包含 1 帧还是 20 帧，每个像素最多贡献
  1 票。
- `episode_support_count <= episode_eligible_count`。
- `eligible_count > 0` 时，支持率必须严格等于两者相除；不可观测像素保持
  0，不产生 NaN。
- 强核心与弱支持互斥；方向生长只能接收直接弱支持，不能把空白桥接像素写入
  观测层。
- 紧框包含该提案的所有直接证据；宽框包含紧框，最大边不超过
  `maximum_proposal_extent_px`。
- 任意提案的推测掩码均不得覆盖任何直接观测掩码。
- 没有强核心时输出 `INSUFFICIENT_EVIDENCE`；不能输出 `ABSENT`。
- 即使成功提框，运行级 `interpretation_status` 仍为 `UNKNOWN`。

## 2026-07-24 实际回放

最终验证产物位于：

```text
runtime_data/ui_anchor_gradient_lab/yuanshen-q-uncharged-20260724-run10/
```

固定视频的两个窗口共得到 15 个独立运动阶段。全画布输出 43 个局部形状
候选；探针诊断只得到 1 个候选：

- 生长方向：`UP`；
- 强核心：58 像素；
- 接收的直接弱支持：6 像素；
- 紧框：`[294, 163, 313, 174)`；
- 宽框：`[294, 157, 313, 174)`；
- 补全方式：`VERTICAL_UP_FROM_WEAK_SUPPORT`。

这说明当前数据支持“下部直接证据较强、上方存在可连接的间歇弱证据，并可
提出向上补全的有界框”。它不证明该区域就是终结技图标，也不证明宽框内部
都是图标像素，因此结论仍为 `UNKNOWN`。

本轮未解决全画布候选的语义筛选：右侧 HUD、文字边缘、血条等也可能形成
屏幕固定局部形状。后续若继续实验，应先增加负例和框稳定性评分，而不是直接
让 OCR 或 SAM 把这些候选提升为图标。

## 第三轮：发现后的固定锚点跟踪

### 解决什么问题

发现候选后，图标可能因为技能动画、透明度、冷却表现或动态背景而暂时降低
视觉置信度。如果继续使用发现阶段的累计支持率重新判断，会把同一槽位误当成
消失或重复候选。

本轮新增 `anchor_tracking.py`，但不改变前两轮的 clean-window 发现算法和
`worldtrace.ui_anchor_gradient_experiment.v2` 输出：

```text
BOX_CANDIDATE + 已选 SAM 掩码
→ 固定 anchor_id
→ 独立 tracking_confidence
→ STABLE
   ├── 证据短暂降低 → TRANSITION_PENDING
   ├── 长期不可观测 → UNKNOWN_RETAINED
   └── 同点位恢复 → STABLE，继续使用原 anchor_id
```

- `discovery_confidence` 是不可变的发现事实；
- `tracking_confidence` 只描述当前固定位置的连续视觉支持；
- 置信度降低不会删除锚点；
- 同一 capture scope 内，新候选框与已有锚点达到 IoU 门槛时由
  `FixedAnchorRegistry` 复用原锚点，不新建 ID；
- 同点位持续出现不同外观时只增加 `appearance_revision` 候选；
- 所有状态都保持 `UNKNOWN`，不产生 UI 模式、图标语义或操作结论。

真实视频桥接位于 `tests/ui_anchor_tracking_replay.py`。它读取 run10 的候选框和
SAM run02 的已选宽框掩码，SAM 不会在每一跟踪帧重新运行。

```powershell
..\.venv\Scripts\python.exe tests\ui_anchor_tracking_replay.py `
  --gradient-summary "runtime_data\ui_anchor_gradient_lab\yuanshen-q-uncharged-20260724-run10\summary.json" `
  --sam-summary "runtime_data\ui_anchor_sam_box_replay\yuanshen-q-run02\summary.json" `
  --window "neighbor-skill-long-tail=8:31" `
  --sam-variant loose `
  --sample-fps 10 `
  --output-dir "runtime_data\ui_anchor_tracking_replay\yuanshen-q-neighbor-skill-run05"
```

主要产物：

- `tracking_timeline.jsonl`：不经过 latest-only GUI 队列的逐观测证据；
- `tracking_overlay.mp4`：固定框、状态、跟踪置信度和外观相似度；
- `tracking_confidence_plot.png`：三条独立曲线及过渡状态背景；
- `tracking_contact_sheet.png`：首帧、最低证据、状态转换和末帧；
- `identity_comparison.png`：过渡前后同槽位与移位负例对照；
- `summary.json`：输入哈希、状态转换、非恒真视觉验收和明确限制。

### 2026-07-24 实际结果

`8–31s` 共回放 230 个观测：

- `STABLE=213`，`TRANSITION_PENDING=17`，没有进入
  `UNKNOWN_RETAINED`；
- `11.20s` 进入过渡，`12.10s` 使用同一
  `anchor-2ff2e38b3df05a19` 恢复；
- `20.80s` 因半透明区域后的背景变化再次进入过渡，`21.60s` 恢复；
- 第二干净窗口内稳定观测占 `95.15%`，最大连续稳定 94 次，末帧为
  `STABLE`；
- frame 1068 与 frame 2412 的同槽位高通 NCC 为 `0.943`、梯度 NCC
  为 `0.931`，最佳偏移为 `[0, 0]`；
- 最强移位负例的组合 NCC 为 `0.262`，同槽位相对负例的差距为 `0.674`。

因此本轮支持“局部证据降低时保留固定槽位候选，并能在证据恢复后继续使用
同一锚点”。但该视频实际触发的是相邻的 E 技能，没有释放被跟踪的 Q 终结技；
它不能验收“Q 自身释放、消失、重现”的完整情况。稳定 ID 和“永不删除”也是
状态机契约，不能单独当作视觉证明，所以 summary 额外要求前后帧相关、位置
偏移、移位负例差距及恢复后的连续稳定证据。

当前状态机尚未接入实时 `minimal_trace_gui`；本轮只完成可离线复核的基础检测
模块和视频回放闭环。
