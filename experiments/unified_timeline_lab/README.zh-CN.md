# Unified Timeline Lab（统一时间线实验）

## 当前目标

`experiments.unified_timeline_lab` 用于验证 Trace Core（轨迹核心）正式契约之前的
最小事实主干：

```text
帧观测 ─┐
         ├─> 同一单调时钟与目标作用域 ─> 确定顺序 ─> 冻结回放快照
输入观测 ┘                                  │
                                             └─> EvidenceRef（证据引用）
```

第一版只回答：

1. 不同生产者的帧和键鼠观测能否在同一录制作用域中稳定排序；
2. 迟到观测能否保留原时间并明确标记，而不是重写时间戳；
3. 时间线能否只保存 `EvidenceRef`（证据引用），不复制帧字节；
4. 冻结后能否重复得到相同顺序，并查找某次输入前后的最近可用帧；
5. 缺失、跨存储或完整性不符的证据能否显式失败。

它不做 OCR、图标识别、页面分类、游戏状态确认、关系生成或键鼠输出。
`InputObservation`（输入观测）会保留输入监听器给出的 `source_status`（源捕获状态），
但它的投递状态和效果状态仍然都被固定为 `UNKNOWN`。源捕获成功只表示“监听器观察到
了事件”，不能据此推断 Windows 已投递、目标程序已消费或画面已产生预期变化。

## 模块边界

本目录仍属于实验代码，不是 `src/worldtrace` 下的正式公共契约，也不修改 Trace Core
已有模块。文件职责如下：

- `contracts.py`：不可变的作用域、观测绑定见证、帧、输入和证据引用契约；
- `evidence.py`：有界、内容寻址、永不静默淘汰的易失内存证据存储；
- `timeline.py`：有界组装、迟到标记、原子冻结和输入前后帧查询；
- `replay.py`：严格、版本化、只在内存中工作的 JSON 编解码与证据审计；
- `adapters.py`：显式适配现有 `FramePacket` 和 `InputCaptureEvent`；
- `__main__.py`：不接触真实窗口的内存冒烟入口。

包级 `__init__.py` 不自动导入 `adapters.py`，因此导入核心契约不会顺带加载截图、
输入监听、Qt、OpenCV 或模型模块。

## 录制作用域

`TimelineScope`（时间线作用域）由以下信息共同确定：

```text
recording_id
clock_domain
recording_started_at_monotonic_ns
application_id
window_instance_id
window_handle
process_id
process_started_at
target_generation
```

不能只按窗口标题、`HWND` 或 PID 匹配。PID 和窗口句柄都可能复用；
`process_started_at`、`window_instance_id` 和 `target_generation` 用于降低错误串线风险。

`recording_id` 是统一录制身份，不等于截图后端或输入监听器自己的 `session_id`。
每条观测另外保留：

- `producer_id`；
- `producer_session_id`；
- `source_sequence`。

因此截图和输入可以来自两个独立生产者会话，仍然进入同一录制时间线。

`focus_epoch`（焦点代次）不放进窗口身份。一个录制可以跨多个焦点代次，但每个
帧和输入仍保留发生时的焦点代次；后续模块不能把“仍是同一窗口”误当成“输入仍属
于同一焦点租约”。

### ObservationBindingWitness

`ObservationBindingWitness`（观测绑定见证）把一次生产者观测显式绑定到：

```text
完整 TimelineScope
clock_domain
producer_id
producer_session_id
focus_epoch
capture-time client_geometry
valid_from_monotonic_ns .. valid_through_monotonic_ns
```

第一版由调用方提供见证；正式接入后应由 `WindowRegistry`（窗口注册表）或采集协调器
在冻结目标身份、焦点租约和捕获时几何后签发。适配器只接受发生时间落在见证有效区间
内、生产者会话一致且目标身份与组装器作用域完全一致的观测。

见证是显式的信任边界，不是加密证明。它不能证明调用方没有伪造 HWND、进程启动时间、
几何或焦点代次，也不能替代未来跨进程签名、权限隔离或可信采集通道；它的作用是禁止
适配器依靠标题、单个 HWND 或调用时的当前窗口状态临时猜测观测归属。

## 时间语义

帧保留：

```text
capture_started_at_monotonic_ns
captured_at_monotonic_ns       # 统一排序时间
capture_completed_at_monotonic_ns
```

输入保留：

```text
observed_at_monotonic_ns       # 统一排序时间
received_at_monotonic_ns
```

现有 `InputCaptureEvent` 只有捕获时间，因此适配器要求调用方另外传入
`received_at_monotonic_ns`；不能把捕获时间复制成接收时间并伪造“零接收延迟”。
两者必须来自见证声明的同一 `clock_domain`。

所有可比较值必须属于同一个 `clock_domain`。墙钟和设备自己的时间不能参与排序。

组装器先分配 `ingest_sequence`（写入序号）。冻结时的唯一规范顺序为：

```text
(occurred_at_monotonic_ns, ingest_sequence)
```

当新观测的发生时间小于此前的时间高水位时，记录 `late_arrival=true`。相同时间戳
不算迟到，按写入序号稳定排序。这个顺序只是事实排序，不是因果证明。

相同 ID、相同内容的重复追加是幂等操作；相同 ID、不同内容会被拒绝，不能覆盖。
错误作用域、证据校验失败或容量用尽也会在修改时间线前失败。

冻结时间不能早于录制开始、观测完成或任何已引用证据的创建时间；严格回放会重新
检查这些边界，不能通过篡改 JSON 制造“证据尚未存在，时间线已经冻结”的快照。
`freeze()` 要求调用方显式传入属于 `TimelineScope.clock_domain` 的冻结时间，不使用
隐式的本进程时钟替调用方猜测时钟域。

除观测 ID 外，组装器还把以下三元组作为生产者源键：

```text
(producer_id, producer_session_id, source_sequence)
```

同一源键对应另一个观测 ID 时会被拒绝，避免生产者局部序号被静默复用。第一版尚未
生成“序号缺口”或“连续性中断”诊断；因此源键不冲突不等于生产者事件完整无丢失。

## 帧健康状态

`FrameHealth`（帧健康状态）包括：

- `FRESH`：新鲜帧；
- `STALE`：旧帧；
- `DUPLICATE`：重复帧；
- `UNKNOWN`：当前生产者无法判断；
- `CAPTURE_FAILED`：明确截图失败。

截图失败记录没有 `frame_ref`，必须带失败原因。它会保留在时间线中，但
`input_frame_window()` 不会把它当作输入前后可用图像。
可用帧的证据字节数至少必须覆盖 `frame_stride * frame_height`，防止时间线接受无法按
其行跨度和高度解码的帧引用。

## EvidenceRef 与易失存储

`EvidenceRef` 只包含：

```text
store_id
evidence_id
kind
storage_kind
media_type
byte_length
sha256
created_at_monotonic_ns
```

时间线 JSON 不包含帧字节、Base64 图像或临时文件路径。`VolatileEvidenceStore`
（易失证据存储）按 SHA-256 对内容去重；相同字节可以由不同 `kind` 或
`media_type` 引用，语义元数据不会因去重丢失。

第一版证据只存在于进程内：

- 有明确的 `max_items`、`max_references` 和 `max_bytes`；
- `max_references` 独立限制语义引用，避免相同字节签发无限多种引用；
- 不自动淘汰已引用内容；
- 容量满时抛出 `EvidenceCapacityExceeded`；
- 读取时复核存储身份、已签发语义元数据、长度和 SHA-256；
- 实验代码默认不写帧、证据或回放产物。

这里的“默认不写产物”不包含 Python 解释器可能生成的 `__pycache__`，也不代表未来
正式 `EvidenceStore` 不需要持久化。第一版先验证引用与排序契约，持久化策略后置。

成功帧通过 `append_frame_packet_to_timeline()` 在同一个时间线临界区内完成前置校验、
易失证据写入和观测追加。冻结、容量不足、作用域错误、源键冲突或内容冲突都会在证据
写入前失败，避免先写帧字节、后追加时间线失败而留下无人引用的孤儿证据。这里的
“原子”只覆盖当前进程内的时间线组装器与易失存储，不宣称具备跨进程或持久化事务语义。

## 输入前后帧查询

冻结后的 `FrozenTimeline.input_frame_window(input_id)` 返回该输入在规范时间线顺序中
最近的前帧和后帧。它只说明时间邻近关系：

- 不表示输入已投递；
- 不表示游戏消费了输入；
- 不表示后帧变化由该输入造成；
- 截图失败记录不会被选为可用帧。

## 现有实验适配器

`adapters.py` 当前提供：

- `append_frame_packet_to_timeline()`：验证 `FramePacket`、观测绑定见证和组装器作用域，
  再原子完成原始帧的易失证据写入与时间线追加；
- `input_capture_event_to_observation()`：保留现有 `InputCaptureEvent` 的生产者会话、
  局部序号、目标窗口、焦点代次、屏幕/客户区/归一化坐标、按压时长、键码和未知
  投递语义，并把监听器原始状态保存在 `source_status`；调用方必须提供独立的同域
  接收时间。

适配时必须由调用方提供有效的 `ObservationBindingWitness`。适配器不会尝试仅靠标题
或 HWND 猜测窗口实例，也不会自动把两个生产者的 `session_id` 当成同一会话；见证的
完整作用域必须与时间线组装器一致。

## 运行冒烟

在项目根目录运行：

```powershell
cd D:\Games\worldtrace_workspace\worldtrace
..\.venv\Scripts\python.exe -m experiments.unified_timeline_lab --smoke-test
```

冒烟使用纯内存构造一个输入、一个后帧和一个迟到的前帧，随后冻结、JSON 内存
往返、审计证据并查询输入前后帧。它不枚举窗口，不捕获桌面，也不发送输入。

## 验收命令

```powershell
..\.venv\Scripts\python.exe -m unittest discover `
  -s experiments\unified_timeline_lab\tests -t . -v

ruff check experiments\unified_timeline_lab
ruff format --check experiments\unified_timeline_lab

..\.venv\Scripts\python.exe -m compileall -q `
  experiments\unified_timeline_lab
```

## 当前未实现

- 正式 Trace Core 公共契约；
- 跨进程时钟归一化；
- 帧环形缓冲与最近数秒错误现场策略；
- 文件或数据库证据存储；
- 跨进程重启后的持久化离线回放；
- OCR、图标、布局或模型派生结果；
- 生产者序号缺口、乱序来源和连续性中断的诊断事件；
- 输入投递、消费确认、画面效果归因和游戏状态确认。

下一步是否进入正式模块，取决于本实验在真实连续帧与输入流中能否保持无丢失、
无串线、可审计且可重复的顺序。
