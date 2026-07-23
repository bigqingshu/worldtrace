# Model Nodes（模型节点）实验契约

`experiments.model_nodes` 是实验性的模型节点契约与隔离执行层。公共数据结构保持模型无关；当前运行适配器通过独立 Python 环境执行 ZipDepth、Depth Anything V2 Small、MoGe-2、Video Depth Anything（视频深度估计）、SAM 2.1 单图分割与视频跟踪、OpenCLIP Rank/Embed/Retrieve（排序/嵌入/检索）、YOLO、RapidOCR（轻量 OCR 后端），以及 PaddleOCR Stable/RTX50（稳定版/RTX 50 系显卡版 OCR 后端）两个部署变体，不把模型依赖合并进 Frame Inspector（帧检查器）环境，也不写入 `Trace Core`（轨迹核心）。

## 职责

- 用 `NodeDescriptor`（节点描述）声明节点身份、版本、参数分组和支持的设备。
- 用 `NodeParameterSpec`（节点参数规格）描述参数类型、默认值、选项和显式条件。
- 用 `NodeRequest`（节点请求）绑定 `FrameRef`（帧引用）或 `TemporalWindow`（时间窗），并记录 ROI、颜色空间、坐标空间、参数和请求设备。
- 用 `Observation`（结构化观测）保存模型输出的业务中立 payload，用 `ArtifactRef`（产物引用）指向调试图或数组文件。
- 用 `NodeExecutionContext`（节点执行上下文）固定运行 ID、帧/时间窗身份、窗口实例、模型版本和权重哈希。
- 用 `RuntimeReport`（运行报告）区分请求设备、实际设备、回退情况和耗时，用 `NodeResult`（节点结果）包装状态、原因码、上下文和输出。
- 用 `RuntimeAdapterRegistry`（运行适配器登记表）明确区分“模型已部署”和“节点执行适配器已实现”。
- 用版本化 JSONL 协议、`IsolatedWorkerProcess`（隔离工作进程）和 `ModelNodeExecutor`（模型节点执行器）传递共享帧描述符、可选文件路径与小型结构化数据。
- 首批适配器持久复用模型实例；默认返回易失的结构化观测和共享内存预览，显式持久模式才返回原始数组/JSON 引用与可视化产物。易失只承诺不写帧和模型产物，隔离 worker 的诊断日志仍可能落盘。CLIP Embed 显式填写 `index_path` 后的索引写入是用户主动选择的实验数据集副作用，不属于 worker 静默保存模型产物，即使输出保留模式为易失也会发生。

## 设备

`NodeDevice`（节点设备）只包含：

| 枚举 | 规范值 | 接受的常用别名 |
| --- | --- | --- |
| `CPU` | `cpu` | `host` |
| `GPU0` | `cuda:0` | `gpu0`、`gpu:0`、`cuda0` |
| `GPU1` | `cuda:1` | `gpu1`、`gpu:1`、`cuda1` |

`normalize_device()` 只做字符串语义归一化，不检查设备是否存在，也不会偷偷把 GPU 请求改成 CPU。设备发生变化时 `RuntimeReport.fallback_occurred` 必须为真；即使设备没有变化，运行时因精度、Provider 或其他原因发生回退时也可以显式记录为真。

`ModelRegistry.normalize_parameters()` 和 `validate_parameters()` 可选接收
`requested_device`。传入时会同时校验节点设备支持范围，并拒绝 CPU 与 `fp16`/`bf16`
的组合；省略该参数的旧调用继续只做参数规格校验。

## 参数条件

条件使用 `NodeParameterCondition`（节点参数条件）的显式运算符：`equals`、`not_equals`、`in` 和 `not_in`。条件只能引用同一描述中的其他参数，不执行 Python 表达式；多个 UI 条件的组合方式留给上层界面或调度器。

## 输入与输出边界

`FrameRef` 只有帧身份和可选时间信息，不携带像素缓冲。`TemporalWindow` 保存有序且不重复的帧引用。`ROI` 使用 `left/top/right/bottom`，实际单位由请求的 `CoordinateSpace`（坐标空间）声明；归一化坐标必须落在 `0..1`。

`NodeExecutionContext` 只能绑定单帧或时间窗之一，并通过 `frame_id`、`frame_ids` 和 `window_id` 属性提供统一读取方式。模型节点完成时应把它放入 `NodeResult.execution_context`，不要把帧身份、模型版本或权重哈希只写入任意 payload。

状态为 `SUCCEEDED` 的 `NodeResult` 必须同时携带 `NodeExecutionContext` 和成功的 `RuntimeReport`；两处都声明模型 ID 时必须一致。失败、未知、取消或阻断结果至少需要 `reason_code` 或错误说明。

`NodeRequest` 的单帧和时间窗输入互斥；`ROI_PIXEL`/`ROI_NORMALIZED` 坐标空间必须同时提供 ROI。`allow_fallback` 默认关闭，调度器不能因为目标 GPU 不可用而静默改用 CPU。

`Observation.value` 和 `NodeResult.payload`（原始模型载荷）保持为模型无关的结构化 payload，可以是数组、字典或标量；它们不会被强制转换为 `frame_processing.ImageData`。`ArtifactRef` 只保存可追溯引用，不负责创建或写入文件。

公共契约模块不负责加载模型。`executor.py`、`worker_process.py` 和 `workers/` 只实现实验性隔离调用；它们已支持显式 `TemporalWindow` 多帧请求，但不负责连续窗口生成、正式 GPU 全局调度、证据包治理、关系网写入、键鼠输入或 Trace Core 生命周期。连续帧积累与冻结由 Frame Inspector 私有的 `TemporalModelNodeExecutionSession`（时序模型节点执行会话）承担，不会反向扩张本模块或 `Trace Core` 的公共契约。

该 GUI 私有会话默认保留最多 `32` 个路由输入，从中按步长 `1` 冻结 `16` 帧，接受 `16` 个新输入后再生成下一窗，并把锚点绑定到最新采样帧。原始帧、独立 FIT（等比缩放）和完整线性处理结果三条路由均已接入；采集会话、目标 generation、输入来源、处理 revision、节点配置 revision、节点、输入几何/颜色或路由快照变化时都会清空历史并重新预热。

## 模型登记与路由

`ModelRegistry`（模型登记表）只保存部署参考中的静态元数据，不导入模型包、不检查文件是否存在、不探测 GPU，也不启动子进程。工作区根目录由调用方注入，登记表中的仓库、环境和权重始终是相对路径。

```python
from experiments.model_nodes import ModelRegistry, NodeDevice

registry = ModelRegistry(workspace_root=r"D:/workspaces/worldtrace")
node = registry.get("depth.zipdepth")
route = registry.resolve("depth.zipdepth", NodeDevice.GPU0)

assert node.environment_id == "zipdepth-py311"
assert route.python_executable == (
    registry.workspace_root
    / "environments/zipdepth-py311/Scripts/python.exe"
)
```

可用的查询和路由方法：

- `get(node_id)`：按节点 ID（或登记的别名）查询 `ModelRegistration`（模型登记）；
- `list_nodes()` / `list_node_ids()`：列出登记，不包含模型实例；
- `resolve_paths(node_id, weight_key=...)`：只解析路径，`BLOCKED`（阻断）和 `PROPOSED`（拟议）节点也可以查看；
- `resolve(node_id, requested_device=..., weight_key=...)`：校验状态和设备后返回可执行路由；
- `resolve_request(node_id, request, weight_key=...)`：使用请求设备，并以“登记允许且请求允许”的交集决定是否允许回退；
- `can_execute(...)`：只返回是否允许执行，不会触发回退或硬件探测。

当前默认登记包括：

| 类别 | 节点 | 状态/设备摘要 |
| --- | --- | --- |
| 深度 | `depth.zipdepth`、`depth.depth_anything_v2`、`depth.moge2`、`depth.video_depth_anything` | `VERIFIED`；`GPU0`、`GPU1`、`CPU` |
| SAM | `vision.sam.segment_image` | `VERIFIED`；`GPU0`、`GPU1`、`CPU` |
| SAM 视频 | `vision.sam.track_video` | `EXPERIMENTAL`；`GPU0`、`GPU1`，显式提示驱动 |
| SAM 3 | `vision.sam3.segment` | `BLOCKED`；不可执行 |
| CLIP | `vision.clip.rank`、`vision.clip.retrieve`、`vision.clip.embed` | `rank` 为 `VERIFIED`；`retrieve`/`embed` 为可执行的 `EXPERIMENTAL`；`GPU0`、`GPU1`、`CPU` |
| YOLO | `vision.yolo.detect`、`vision.yolo.track` | `detect` 已验证，`track` 拟议 |
| OCR | `vision.ocr.read`、`vision.ocr.read.paddle_stable`、`vision.ocr.read.paddle_rtx50` | RapidOCR 为 `CPU_ONLY`；Stable 为 `VERIFIED`/GPU0；RTX50 为可执行的 `EXPERIMENTAL`/GPU1 |

`ModelNodeStatus`（模型节点状态）只有 `BLOCKED` 和 `PROPOSED` 会被路由器拒绝；`CPU_ONLY` 和 `EXPERIMENTAL` 仍可在其声明设备上显式执行。请求不支持的设备会抛出 `DeviceSelectionError`（设备选择错误），不会静默改用 CPU。

没有登记专用资源键时，可执行路由会按实际选择解析为 `gpu:0`、`gpu:1` 或 `cpu:inference`；CPU OCR 使用独立的 `cpu:ocr`。这只是未来调度器的互斥键，不表示当前已经启动模型或占用对应设备。

路径解析只拼接并返回 `Path`，不调用 `exists()`、不展开通配符；Video Depth Anything 的 `relative`/`metric` 和 YOLO 的 `yolo26n`/`yolo26s` 通过 `weight_key` 选择。路由会同时固定选中的 `weight_key`、模型 ID、模型版本和 SHA-256，避免加载一种权重却记录成另一种模型。OpenCLIP、SAM、YOLO 和已有深度权重的本机哈希已登记；运行适配器在模型加载前再次检查路径边界、文件存在性和已登记 SHA-256。

### CLIP 嵌入与检索实验入口

`clip_index.py` 提供 `ClipEmbeddingRecord`（CLIP 嵌入记录）、`ClipEmbeddingIndex`（CLIP 嵌入索引）和 `ClipEmbeddingQuery`（CLIP 查询向量）等不可变纯逻辑类型。索引要求记录使用相同的 `model_id`、`model_revision`、权重 SHA-256、维度和存储归一化模式；查询时再次校验这些身份与维度，并以记录 ID 作为同分项的确定性排序键。

`vision.clip.embed` 已接入 `openclip.embed.v1` 适配器。当前 ViT-B/32 返回 `512` 维结构化向量，单个嵌入没有栅格可视化；`index_path` 默认留空，因此默认只返回易失结果、不写索引。只有调用方显式填写工作区内相对索引路径时，worker 才以单写者方式加载已有兼容索引、追加唯一记录，并通过同目录临时文件、刷新与原子替换写回。

`vision.clip.retrieve` 已接入 `openclip.retrieve.v1` 适配器。它要求 `index_path` 指向显式存在且兼容的索引，支持图像查询和文本查询，返回余弦 Top-K 结构化结果，并支持 `retrieval_contact_sheet`、`similarity_matrix`、`pair_comparison` 三种共享内存预览。`load_clip_embedding_index(...)` 会在解析记录前检查文件校验和、可选预期 SHA-256、模型身份、维度和归一化模式。

原子替换只保证读者不会看到半写文件，不提供多进程 compare-and-swap（比较并交换）或锁。两个 worker 同时读取旧索引再写回时仍可能发生最后写入覆盖，因此当前索引更新契约明确要求调用方保证单写者；不能把“原子文件替换”描述成“并发追加无丢更新”。

## 隔离执行链路

当前执行路径为：

```text
ModelNodeConfiguration（模型节点配置）
-> ModelNodeExecutionSession（单帧监督线程）或TemporalModelNodeExecutionSession（时序监督线程）
-> 请求出队后向SharedFramePool（共享帧池）发布一帧或有序多帧（默认），或显式编码为PNG
-> ModelNodeExecutor解析环境、权重、设备与适配器
-> IsolatedWorkerProcess启动/复用对应独立环境
-> JSONL只传共享内存描述符/路径、参数和身份
-> worker以只读视图推理并发布共享预览；可选保存原始结果与产物
-> NodeResult + FilterResult + DeduplicationResult
-> 父进程复制共享预览、发送WorkerRelease并交给GUI
```

工作进程不会通过 pickle 传递 `MappingProxyType`、NumPy 数组或 Torch Tensor。`SharedFrameDescriptor`（共享帧描述符）只携带共享内存名、布局、颜色模型、generation 和租约令牌；像素不进入 JSONL。stdout 只允许协议消息，第三方模型日志进入有界 stderr 日志；关闭、取消或超时会回收工作进程，在 Windows 上同时终止虚拟环境启动器的子进程树。

`FrameTransportKind.SHARED_MEMORY`（共享内存传输）和 `OutputRetention.VOLATILE`（易失输出）是 GUI 默认值。共享输入失败会返回结构化错误，不会静默改写临时文件。`FrameTransportKind.FILE_PATH`（文件路径传输）与 `OutputRetention.PERSISTENT`（持久输出）相互独立：前者只决定输入如何跨进程，后者决定输入临时文件和模型产物是否保留。单帧与时序窗口都可使用共享内存；Video Depth Anything 和 SAM 2 Video 均已在共享内存加易失输出下执行，不再要求文件窗口或持久输出。易失模式不写输入帧、原始数组或可视化产物，但 worker 诊断日志仍可能由隔离运行时写盘。

父进程的输入池采用固定双槽并按需升级容量 generation。worker 输出池也有固定槽数；执行器复制完预览后发送 `WorkerRelease`（worker输出释放消息），worker 按 `request_id`、`run_id` 和租约令牌校验所有权后复用槽位。协议不允许文件与共享内存载荷同时出现。

GPU0/GPU1 使用宿主逻辑设备：

| 用户选择 | 工作进程环境 | 上游模型看到的设备 | 运行报告 |
| --- | --- | --- | --- |
| `GPU0` | `CUDA_VISIBLE_DEVICES=0` | 本进程 `cuda:0` / ZipDepth `cuda` | `actual_device=cuda:0` |
| `GPU1` | `CUDA_VISIBLE_DEVICES=1` | 本进程 `cuda:0` / ZipDepth `cuda` | `actual_device=cuda:1` |
| `CPU` | `CUDA_VISIBLE_DEVICES=-1` | `cpu` | `actual_device=cpu` |

这样可兼容 ZipDepth 上游对 `device == "cuda"` 的精确判断，同时仍诚实记录宿主的 GPU0 或 GPU1。当前没有静默 CPU 回退。

当前类型化适配器：

| 节点 | 适配器 | 环境 | 当前设备 | 主要输出 |
| --- | --- | --- | --- | --- |
| `depth.zipdepth` | `zipdepth.image.v1` | `zipdepth-py311` | GPU0、GPU1、CPU | `float32 NPY`、相对逆深度观测、深度预览 |
| `depth.depth_anything_v2` | `depth_anything_v2.image.v1` | `depth-anything-v2-py311` | GPU0、GPU1、CPU | 相对逆深度观测、持久 `float32 NPY`、六种深度预览 |
| `depth.moge2` | `moge2.geometry.v1` | `moge2-py311` | GPU0、GPU1、CPU | 深度/点图/法线/有效区/内参 NPY 和几何预览 |
| `depth.video_depth_anything` | `video_depth_anything.temporal.v1` | `video-depth-anything-py311` | GPU0、GPU1、CPU | 时间窗深度观测、共享首帧预览；NPY/NPZ/视频为可选持久产物 |
| `vision.sam.segment_image` | `sam2.image.segment.v1` | `sam2-py312` | GPU0、GPU1、CPU | 掩码/分数/logits NPZ、提示分割观测和九种预览 |
| `vision.sam.track_video` | `sam2.video.track.v1` | `sam2-py312` | GPU0、GPU1 | 显式提示跟踪观测、共享掩码/轨迹预览；持久产物可选 |
| `vision.clip.rank` | `openclip.rank.v1` | `torch-vision-py312` | GPU0、GPU1、CPU | 排序 JSON、真实相似度/概率、可选嵌入和三种预览 |
| `vision.clip.embed` | `openclip.embed.v1` | `torch-vision-py312` | GPU0、GPU1、CPU | `512` 维结构化嵌入；无单样本预览，可选原子追加索引 |
| `vision.clip.retrieve` | `openclip.retrieve.v1` | `torch-vision-py312` | GPU0、GPU1、CPU | 图像/文本查询的余弦 Top-K 与三种共享预览 |
| `vision.yolo.detect` | `ultralytics.detect.v1` | `torch-vision-py312` | GPU0、GPU1、CPU | 检测 JSON、逐框真实置信度/ROI、八种预览 |
| `vision.ocr.read` | `rapidocr.read.v1` | `ocr-onnx-py312` | CPU only | transcript、原始 JSON、逐行文字/分数/几何和 OCR 预览 |
| `vision.ocr.read.paddle_stable` | `paddleocr.read.v1` | `paddleocr-py312` | GPU0 | 与 RapidOCR 对齐的逐行观测、词框、原始 JSON、共享 OCR 预览 |
| `vision.ocr.read.paddle_rtx50` | `paddleocr.read.v1` | `paddleocr-rtx50-py312` | GPU1 | 与 Stable 相同的 Small 权重、OCR 公共结果和共享预览 |

Depth Anything V2 Small 适配器固定当前已验证的 `vits` 结构和 FP32 精度，只开放请求级 `input_size`（默认 `518`，保持比例并按 14 对齐）与 `warmup_iters`（默认 `0`）。模型实例在相同设备、权重和模型身份下复用；修改这两个请求参数不会重新加载模型。它返回 `relative_depth_map` 相对逆深度观测，但没有模型置信度，深度值或预览亮度不得被伪装成 `Observation.confidence`。

该节点登记八种输出模式：`raw_npy`、`metrics_json`，以及六种栅格预览 `color_image`、`grayscale_image`、`comparison_color`、`color_only`、`comparison_gray`、`grayscale_only`；默认主预览为 `color_image`。共享内存加易失输出时只发布所选主预览，不创建输出目录或图片文件；显式持久输出会保存核心原始 NPY，并按选择保存指标 JSON 和栅格图像。只有登记状态不可执行或确实缺少运行适配器的节点才显示只读语义；当前 SAM 2 Video 与 CLIP Embed/Retrieve 已有运行适配器，不能再标成“待接入”。

PaddleOCR 两个部署变体必须同时使用登记的 `PP-OCRv6_small_det` / `PP-OCRv6_small_rec` 模型名和本地目录，避免 PaddleOCR 改选默认 Medium 模型或下载权重。Stable 的宿主 GPU0 与 RTX50 的宿主 GPU1 在各自隔离 worker 内都映射为本地 `gpu:0`；适配器不允许 CPU 回退，并把 Paddle 模块/分发包版本及 cuDNN 编译版与运行版差异写入结构化元数据和 warning。Stable GPU0 与 RTX50 GPU1 的统一执行链路均已完成真实运行验证；RTX50 仍因开发版 wheel 保持 `EXPERIMENTAL`，而不是因为无法执行。

过滤顺序固定为先生成全部原始 `Observation`，再执行置信度/质量过滤，只把 `ACCEPTED` 和 `UNSCORED` 观测送入时序去重。`REJECTED` 观测仍保留在结果表中用于审计，但不会污染稳定状态。

## 当前真实运行验收

- Video Depth Anything：GPU0 已通过执行器级时序验证；Frame Inspector 以 GPU1、共享内存和易失输出运行 `16` 张有序 `256×256` 帧，成功得到时序深度观测与内存 `preview_first_frame`，没有持久产物。
- SAM 2 Video：GPU0 与 GPU1 均通过 `3` 帧执行器级冒烟；Frame Inspector 以 GPU0 和默认 `16` 帧窗口通过 GUI 冒烟，返回 `16` 条观测、`0` 个持久产物，总耗时约 `4.62 s`、模型执行约 `2.10 s`。该节点为 `EXPERIMENTAL`，必须提供显式点或框提示，不支持无提示自动发现对象。
- SAM 2 的 Windows 环境缺少可选 `sam2._C` 原生后处理扩展时，worker 会把降级写成结构化 warning 并继续运行；这不是静默忽略，也不把成功结果伪装成完全无降级。
- PaddleOCR：Stable GPU0 与 RTX50 GPU1 均通过真实隔离执行；前者为 `VERIFIED`，后者因开发版 wheel 保持 `EXPERIMENTAL`，两者都不会回退到 CPU。
- OCR 手动同帧对比：Frame Inspector 使用一份 `720×820` 共享内存输入和易失输出，RapidOCR 得到 `43` 行、PaddleOCR Stable 得到 `48` 行，匹配 `34` 行，未匹配分别为 `9` 与 `14` 行，总耗时约 `10.55 s`，两侧均为 `0` 个持久产物；第一版禁用实时预览。
- CLIP：Embed 与 Retrieve 的执行器已在双 GPU 上验证；GUI 也完成 GPU0 Embed 写索引后由 GPU1 Retrieve 查询的链路。Embed 返回 `512` 维结构化向量且没有单样本预览；Retrieve 的图像/文本查询及三种内存预览均已进入可执行链路。
- SAM 3：继续保持 `BLOCKED`；权重不可用且 Windows Triton 导入条件未满足前，不注册可执行适配器，也不以 SAM 2 结果代替。

## 离线冒烟验证

`offline_smoke.py` 走与 GUI 相同的登记、权重校验、进程协议和结果装配：

```powershell
# ZipDepth / GPU1
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe `
  -m experiments.model_nodes.offline_smoke `
  --node depth.zipdepth `
  --input reference_repos/zipdepth/assets/examples/im0.jpg `
  --device cuda:1 `
  --parameters-json '{"precision":"fp32","input_size":384,"ensure_multiple_of":32,"warmup_iters":1}'

# Depth Anything V2 Small / GPU1；适配器固定vits和FP32
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe `
  -m experiments.model_nodes.offline_smoke `
  --node depth.depth_anything_v2 `
  --input reference_repos/depth-anything-v2/assets/examples/demo01.jpg `
  --device cuda:1 `
  --parameters-json '{"input_size":518,"warmup_iters":0}'

# YOLO / GPU0；GPU1只需改为cuda:1
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe `
  -m experiments.model_nodes.offline_smoke `
  --node vision.yolo.detect `
  --input reference_repos/ultralytics/ultralytics/assets/bus.jpg `
  --device cuda:0 --weight-key yolo26n `
  --parameters-json '{"imgsz":640,"conf":0.25,"iou":0.7,"classes":"","max_det":300,"agnostic_nms":false,"precision":"fp32"}'

# RapidOCR / CPU
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe `
  -m experiments.model_nodes.offline_smoke `
  --node vision.ocr.read `
  --input reference_repos/rapidocr/python/tests/test_files/ch_en_num.jpg `
  --device cpu

# Video Depth Anything / GPU1；时间窗按顺序重复 --input
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe `
  -m experiments.model_nodes.offline_smoke `
  --node depth.video_depth_anything `
  --input scratch/video_depth_temporal_input/frame_001.png `
  --input scratch/video_depth_temporal_input/frame_002.png `
  --device cuda:1 --weight-key relative `
  --parameters-json '{"precision":"fp16","input_size":384,"max_res":640,"target_fps":30.0}'
```

命令只打印小型 JSON 摘要。原始数组、掩码、排序、检测和 OCR 结果保存在 `runtime_data/vision_artifacts/<run_id>`，不会被展开到终端。单帧节点要求恰好一个 `--input`；时序节点要求每帧重复一次并保持顺序。

## 可视化契约

`visualization.py` 只负责配置和结果校验，不包含 OpenCV、Matplotlib、Qt 或模型绘图代码。调用方先用 `get_visualization_modes(registry, node_id)` 查询 `ModelRegistry` 中登记的模式，再用 `validate_visualization_request()` 拒绝节点未声明的模式。

`VisualizationRequest`（可视化请求）支持：

- `modes`：可同时选择多个登记模式；空元组表示关闭所有可视化；
- `primary_mode`：可选主预览模式，填写时必须在 `modes` 中；未填写表示只生成所选产物，不创建主预览；
- `output_directory`：工作区相对目录，拒绝绝对路径、盘符和 `..`；
- `image_format`：`png` 或 `jpeg`，输入 `jpg` 会规范为 `jpeg`；
- `alpha`：`0..1` 的叠加透明度；
- `line_width`：正整数线宽；
- `save_artifacts`：是否允许生成持久化 `ArtifactRef`；
- `depth_normalization`：`per_frame`、`fixed_range` 或 `percentile`。

`ModelRegistration.preview_visualization_modes`（栅格预览模式）是
`visualization_modes` 的显式子集。前者只包含能够返回单张 PNG/JPEG 预览的模式，
后者还可以包含 NPY、NPZ、视频、PLY、GLB 和其他仅产生产物的模式；未填写前者的
自定义登记继续按“全部输出模式均可预览”处理。`primary_mode` 必须属于栅格预览
子集，不能把数组、视频或三维文件当成 Qt 主图；只选择产物模式时允许其为 `None`。

SAM、OpenCLIP 和 YOLO 已实现的节点专属绘图选项使用
`visualization.*` 前缀登记为 `NodeParameterSpec`。这些值保留在冻结配置中，执行器
会在模型缓存键与 `WorkerRequest.parameters` 之外提取它们，再合并到
`WorkerRequest.visualization`；留空的可选字符串不会下发。

深度归一化范围采用以下语义：

| 策略 | `visual_min` / `visual_max` |
| --- | --- |
| `per_frame` | 不接受固定值，由未来渲染器逐帧确定 |
| `fixed_range` | 两者必填、有限且 `visual_min < visual_max` |
| `percentile` | 表示百分位，默认 `2` 和 `98`，范围必须在 `0..100` |

`VisualizationResult`（可视化结果）只能返回 `ArtifactRef` 和可选 `MemoryPreviewRef`（内存预览引用）。当 `modes=()` 时，两者都必须为空；当 `save_artifacts=False` 时，仍可返回主模式的内存预览，但不能返回文件产物。`VisualizationRenderer`（可视化渲染器）只是未来实现需要遵守的协议，输入的 `NodeResult.payload` 和 `Observation.value` 必须按只读数据处理。

运行适配器会先完成推理和结构化观测，再进入可视化阶段。易失模式不写核心 NPY/NPZ/JSON，主模式预览直接发布到 worker 持有的共享槽；持久模式才写原始输出和所选产物，此时核心写入失败仍是执行失败。可视化异常时节点结果保持 `SUCCEEDED`，预览和可视化产物为空，并返回稳定前缀 `VISUALIZATION_FAILED:` 的告警。`save_artifacts=False` 时，视频、EXR、PLY、GLB、多裁剪和整套 map 等仅产生产物的分支不会继续写盘。Video Depth Anything 和 SAM 2 Video 的主预览都支持共享内存时序输入与易失输出，文件路径和持久化只是显式兼容选项。

Video Depth Anything 在 `max_res` 缩放后会为每帧记录 `source_size`、`processed_size`、`scale_x`、`scale_y` 和 `processed_frame_pixel` 坐标空间。点云导出使用按缩放比例换算后的焦距，同时保留源焦距，避免把处理分辨率的深度坐标误投回原始截图。离线图片序列没有真实采集时间，因此 smoke 入口将时间戳保留为未知，而不会伪造帧间隔。

调用方应通过 `dispatch_visualization()` 进入渲染：当 `modes=()` 时它不会调用 renderer；其他情况只接受成功的节点结果，并把同一个 `NodeExecutionContext` 绑定到可视化结果，防止调试产物脱离帧和模型身份。

## 观测过滤

`filtering.py` 只对 `Observation` 做派生判定，不替换、删除或改写 `NodeResult.observations`。`FilterDecision`（过滤判定）始终引用原始观测对象，状态为：

| 状态 | 含义 |
| --- | --- |
| `ACCEPTED` | 所有启用规则均通过，或缺失项由配置明确允许 |
| `REJECTED` | 至少一个白名单、置信度或质量规则明确失败 |
| `UNSCORED` | 缺少规则所需的显式分数，且策略要求保留为未评分 |

`ConfidenceFilterConfig`（置信度过滤配置）只把 `Observation.confidence` 当作置信度。启用 `min_confidence` 后，无分数观测根据 `missing_score_policy` 进入 `ACCEPTED`、`REJECTED` 或 `UNSCORED`；默认策略是 `UNSCORED`。CLIP 的 `raw_logit`、深度数组、可视化亮度及任意 payload 数字都不会被自动压缩或转换成 `0..1` 置信度。

```python
from experiments.model_nodes import (
    ConfidenceFilterConfig,
    MissingScorePolicy,
    filter_observations,
)

filtered = filter_observations(
    node_result.observations,
    ConfidenceFilterConfig(
        min_confidence=0.5,
        missing_score_policy=MissingScorePolicy.UNSCORED,
        kind_allowlist=("detection", "ocr_text"),
    ),
)
```

标签白名单必须同时设置 `label_metadata_key`，过滤器只读取该明确的 `Observation.metadata` 键，不从 `Observation.value` 猜测标签。`QualityFilterRule`（质量过滤规则）固定从 `metadata["quality_metrics"]` 读取有限数值，例如有效像素比例、清晰度或覆盖率；顶层 metadata、confidence 和模型 payload 中同名字段都不会被当作质量指标。

`FilterResult`（过滤结果）按输入顺序保留全部 decisions，并分别暴露 `accepted`、`rejected` 和 `unscored`。确定性拒绝优先于未评分：同一观测既不在白名单又缺少分数时，最终为 `REJECTED`，同时 `reason_codes` 保留其他原因。

## 会话级去重

`deduplication.py` 只生成引用原始 `Observation` 的状态判定，不替换、删除或改写 `NodeResult.observations`。`ObservationDeduplicator.apply()` 每次显式接收：

- `scope_id`：由采集会话身份和配置版本组成；任何变化都会立即清空旧状态；
- `monotonic_ns`：同一 scope 内不得倒退的单调时钟；
- 当前帧或时间窗产生的原始观测序列。

`DeduplicationConfig`（去重配置）包含 `enabled`、identity/signature metadata 键、`confidence_delta`、`stable_frames`、`cooldown_ms`、`ttl_ms`、`emit_initial`、`emit_expired` 和 `max_entries`。状态只有：

| 状态 | 含义 |
| --- | --- |
| `NEW` | 首次看到显式 identity，或缺少显式键时按策略直通 |
| `UPDATED` | 稳定 signature 改变，或置信度变化达到阈值 |
| `UNCHANGED` | signature 相同、小分数抖动、稳定帧不足或仍处于冷却 |
| `EXPIRED` | TTL 到期或有界缓存淘汰 |

所有 raw decisions 始终保存在 `DeduplicationResult.decisions`。`only_changed` 排除 `UNCHANGED`，`visible` 排除已经 `EXPIRED` 的判定，`emitted` 才应用 `emit_initial`/`emit_expired` 开关；因此关闭发出不会删除可审计的 `NEW` 或 `EXPIRED` 原始决策。

固定 ROI OCR 的建议配置为：

```python
config = DeduplicationConfig(
    identity_metadata_key="slot_id",
    signature_metadata_key="normalized_text",
    confidence_delta=0.05,
    stable_frames=2,
    cooldown_ms=250,
    ttl_ms=2_000,
)
```

首次 signature 和后续新 signature 都必须连续达到 `stable_frames`：首次确认产生 `NEW`，后续确认产生 `UPDATED`。同一 identity 在同一 `monotonic_ns` 最多贡献一个稳定帧，同批重复观测或重复调用不能伪造稳定性。同一个 `slot_id` 和 `normalized_text` 下的小置信度波动只产生 `UNCHANGED`。去重器不从 OCR payload 猜文字，也不对深度数组、图像、任意字典或模型 payload 做隐式哈希。缺少或提供复杂 identity/signature 时，默认 `PASSTHROUGH` 为不缓存的 `NEW`；需要严格输入时可使用 `MissingMetadataPolicy.ERROR`。`ERROR` 会先校验整批 metadata，任一项非法时不会推进时钟、过期条目或提交前面的更新。

## 配置快照

`ModelNodeConfiguration`（模型节点配置）把一次 UI 应用结果冻结为带 revision 的快照，字段包括：

- `node_id`、`requested_device` 和可选 `weight_key`；
- 经登记表补默认值并校验后的只读 `parameters`；
- `VisualizationRequest`、`ConfidenceFilterConfig` 和 `DeduplicationConfig`；
- `only_changed`，表示消费去重结果时是否只显示发生变化的判定；
- `input_transport` 和 `output_retention`，默认分别为共享内存与易失输出。

快照本身不加载模型、不检查权重文件或 GPU，也不创建 `NodeRequest`。`visualization.node_id` 必须与配置节点一致，设备别名会规范为 `NodeDevice`，外部参数字典会被复制并冻结。revision 只由配置面板在成功点击“应用节点配置”后递增；继续编辑控件不会改变已经应用的旧快照。
