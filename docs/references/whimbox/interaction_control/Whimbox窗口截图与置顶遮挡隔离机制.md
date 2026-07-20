# Whimbox 窗口截图与置顶遮挡隔离机制

## 1. 文档范围

本文说明 Whimbox 2.5.4 在 Windows 上为什么能够获取游戏画面，而不把覆盖在游戏上方的 Whimbox 前端或其他独立置顶窗口截入图像。

重点区分：

1. 截取目标窗口内容；
2. 截取显示器最终画面；
3. 游戏前台焦点；
4. 系统级键鼠输入。

相关文档：

- [Whimbox鼠标拖拽与镜头旋转机制](Whimbox鼠标拖拽与镜头旋转机制.md)
- [平台抽象与交互核心](../analysis/07-平台抽象与交互核心.md)
- [Whimbox图像处理参考](../../../concepts/game_ai_planning/03-Whimbox图像处理参考.md)
- [Whimbox任务调度停止与资源互斥](../../../concepts/game_ai_planning/06-Whimbox任务调度停止与资源互斥.md)
- [Whimbox错误恢复与执行证据](../../../concepts/game_ai_planning/10-Whimbox错误恢复与执行证据.md)
- [多窗口输入归属与游戏视角异常恢复构想](../../../concepts/interface_discovery/多窗口输入归属与游戏视角异常恢复构想.md)
- [游戏瞬态故障判读与置信度隔离构想](../../../concepts/interface_discovery/游戏瞬态故障判读与置信度隔离构想.md)
- [AzurLaneAutoScript可借鉴机制与设计思想](../../../references/alas/AzurLaneAutoScript可借鉴机制与设计思想.md)

## 2. 核心结论

Whimbox 在 Windows 上不是截取显示器最终显示出来的桌面画面，而是：

> 根据游戏窗口句柄 `HWND`，要求游戏窗口把自己的客户区内容绘制到一张屏幕之外的内存位图中。

因此，Whimbox 前端即使设置为置顶并覆盖在游戏窗口上方，只要它是另一个独立窗口，通常也不会出现在游戏截图中。

可以概括为：

```text
桌面最终画面
├─ 游戏窗口
├─ Whimbox置顶前端窗口
├─ 通知窗口
└─ 其他应用窗口

Whimbox截图
└─ 只请求游戏窗口HWND对应的客户区内容
```

这里的关键不是“截图后把前端抠掉”，而是截图源一开始就不是桌面最终合成画面。

## 3. Windows 截图调用链

### 3.1 上层入口

业务代码调用：

```python
itt.capture()
```

[`InteractionBGD.capture()`](../../../../../reference_repos/whimbox/whimbox/interaction/interaction_core.py#L47) 将请求交给 `PrintWindowCapture`：

```text
itt.capture()
→ InteractionBGD.capture()
→ PrintWindowCapture.capture()
→ WindowsCaptureManager.capture_window()
```

平台截图管理器由 [`platform/factory.py`](../../../../../reference_repos/whimbox/whimbox/platform/factory.py#L27) 根据当前操作系统选择。

### 3.2 获取游戏窗口客户区尺寸

Windows 实现位于 [`platform/windows/capture.py`](../../../../../reference_repos/whimbox/whimbox/platform/windows/capture.py#L17)。首先使用：

```python
left, top, right, bottom = win32gui.GetClientRect(hwnd)
```

由此获得目标游戏窗口的客户区宽度和高度。

这里使用的是 `GetClientRect`，不是整个桌面尺寸，也不是 `GetWindowRect`。按代码意图，截图范围不包含：

- Windows 标题栏；
- 窗口边框；
- 游戏窗口外部区域；
- 覆盖在上方的独立应用窗口。

### 3.3 创建屏幕外内存位图

接下来创建窗口 DC、兼容 DC 和兼容位图：

```python
hdc_window = win32gui.GetWindowDC(hwnd)
hdc_mem = win32ui.CreateDCFromHandle(hdc_window)
hdc_compat = hdc_mem.CreateCompatibleDC()

bmp = win32ui.CreateBitmap()
bmp.CreateCompatibleBitmap(hdc_mem, width, height)
hdc_compat.SelectObject(bmp)
```

这张位图存在于内存中，不是显示器上的可见窗口。

可以理解为：

```text
游戏窗口
→ 按要求绘制
→ 内存DC
→ 内存Bitmap
→ NumPy数组
```

### 3.4 PrintWindow 请求目标窗口绘制

核心调用是：

```python
ctypes.windll.user32.PrintWindow(
    hwnd,
    hdc_compat.GetSafeHdc(),
    3,
)
```

对应源码见 [`platform/windows/capture.py`](../../../../../reference_repos/whimbox/whimbox/platform/windows/capture.py#L36)。

这里显式传入了游戏窗口的 `hwnd`，所以 Windows 请求的是这个目标窗口的内容，而不是当前屏幕上位于最上层的窗口。

参数 `3` 可理解为两个标志的组合：

```text
PW_CLIENTONLY        = 1
PW_RENDERFULLCONTENT = 2
```

含义分别是：

- 只请求客户区；
- 尝试获取完整窗口渲染内容。

### 3.5 转换为 NumPy 图像

绘制完成后读取位图字节：

```python
bmpstr = bmp.GetBitmapBits(True)
img = np.frombuffer(bmpstr, dtype=np.uint8)
img.shape = (height, width, 4)
```

平台层输出 BGRA 四通道数组。后续截图层会统一分辨率并缓存，业务入口默认删除 Alpha 通道，返回 BGR 三通道数组。

## 4. 为什么置顶窗口不会进入截图

### 4.1 桌面截屏读取最终合成画面

普通桌面截屏通常读取桌面最终合成结果：

```text
游戏窗口
+
置顶窗口
+
通知
+
系统光标
→ 显示器最终画面
→ 截图
```

在这种方式下，谁位于游戏上方，谁就会遮挡游戏并进入截图。

### 4.2 PrintWindow 指定目标 HWND

Whimbox 的 Windows 主链则是：

```text
指定游戏HWND
→ 让该窗口单独绘制客户区
→ 写入内存位图
```

Whimbox 前端窗口拥有自己的独立 `HWND`，不属于游戏客户区，也没有被传给 `PrintWindow`。因此前端窗口的置顶状态通常不会改变截图内容。

### 4.3 窗口Z序与截图目标不同

Windows 桌面上的窗口存在 Z 序：

```text
最上层：Whimbox前端
下方：游戏窗口
更下方：其他窗口
```

桌面截屏受到 Z 序影响，但 `PrintWindow(game_hwnd, ...)` 根据句柄选择目标。只要游戏窗口仍能提供自己的渲染内容，Z 序本身通常不决定截图结果。

所以准确表述是：

> Whimbox 不是绕过置顶窗口后读取屏幕，而是根本没有把置顶窗口所在的桌面合成层作为截图源。

## 5. 独立置顶窗口与游戏内覆盖层的区别

### 5.1 独立桌面窗口

以下内容通常不会进入 `PrintWindow` 获取的游戏客户区：

- Whimbox 自己的独立前端窗口；
- 记事本、浏览器等普通窗口；
- Windows 桌面通知；
- 其他独立置顶工具窗口。

前提是它们确实属于独立 `HWND`，而不是注入到游戏渲染流程中。

### 5.2 游戏自身渲染内容

以下内容可能被截入，因为它们属于游戏最终提交的客户区画面：

- 游戏内菜单；
- 游戏内聊天窗口；
- 游戏自身鼠标光标；
- 游戏内任务提示；
- 游戏自己的覆盖UI。

### 5.3 注入式覆盖层

Steam Overlay、显卡滤镜、性能监控叠加层等工具可能通过注入游戏渲染管线工作，而不是作为单独桌面窗口存在。

这类覆盖层是否进入截图，取决于：

- 它在游戏渲染链中的合成阶段；
- `PrintWindow` 获得的是哪一层表面；
- 游戏和显卡驱动如何实现窗口绘制；
- 覆盖层是否被合成进目标窗口客户区。

因此不能把“独立置顶窗口不会进入截图”推广为“所有覆盖层都不会进入截图”。

## 6. 截图与游戏前台焦点是两件事

### 6.1 截图按 HWND 选择目标

截图调用传入游戏窗口句柄，因此不以“当前前台窗口是谁”作为唯一选择依据。

理论上，即使游戏不是前台窗口，只要游戏仍能响应 `PrintWindow` 并继续渲染，也可能获得正确画面。

### 6.2 键鼠输入依赖前台状态

Whimbox 的 Windows 键鼠输入使用系统级 `mouse_event`、`SetCursorPos` 和键盘事件。这些事件不是发送到截图位图，而是进入真实桌面输入系统。

因此输入通常要求游戏能够接收事件。

[`before_operation()`](../../../../../reference_repos/whimbox/whimbox/interaction/interaction_core.py#L294) 会在执行操作前检查游戏窗口；若游戏失去焦点，会尝试调用窗口处理器恢复游戏前台状态。

所以两条链应分开理解：

```text
截图：按游戏HWND获取窗口内容
输入：依赖游戏处于可接收系统输入的状态
```

前端窗口可以保持“视觉上置顶”，但它不能持续抢占输入焦点或拦截本应发送给游戏的鼠标位置。

## 7. 与 Whimbox 前端解耦的实际关系

前端置顶不会直接污染截图，是因为：

1. 前端与游戏属于不同窗口；
2. 截图后端保存了游戏窗口句柄；
3. 截图调用显式指定游戏 HWND；
4. 图像写入内存位图，不经过桌面最终合成画面；
5. 图像处理只读取这张内存位图转换出的 NumPy 数组。

前端的职责是展示状态和接收用户命令；截图后端的职责是根据游戏窗口句柄获取画面。两者没有共用“截取当前桌面”这一输入源。

## 8. PrintWindow 的限制

### 8.1 目标窗口必须能够提供内容

`PrintWindow` 并不是直接读取所有 GPU 显存表面。它通常需要目标窗口或相关窗口管理机制提供可绘制内容。

对于 DirectX、Vulkan、OpenGL 或 Unreal Engine 游戏，结果取决于具体渲染实现。

可能出现：

- 正常画面；
- 全黑画面；
- 旧画面；
- 部分区域未刷新；
- 窗口最小化后无法获取；
- HDR 或特殊交换链颜色异常。

### 8.2 游戏失去焦点后可能停止渲染

即使外部置顶窗口不会直接进入截图，游戏自身可能在失去焦点后：

- 降低帧率；
- 暂停更新；
- 停止交换缓冲；
- 保持最后一帧。

此时获得的仍然可能是一张“没有被前端遮挡”的游戏图，但它是旧帧而不是当前实时画面。

### 8.3 最小化窗口不等于被覆盖

普通窗口覆盖和窗口最小化不是同一情况：

- 被其他窗口覆盖：目标窗口仍可能正常渲染；
- 最小化：目标窗口可能不再拥有可用绘制表面。

所以不能因为 `PrintWindow` 不受遮挡，就假设它一定支持最小化后台截图。

### 8.4 当前代码没有检查 PrintWindow 返回值

当前实现直接调用 `PrintWindow`，但没有检查它的布尔返回值：

```python
ctypes.windll.user32.PrintWindow(...)
```

随后仍会读取位图字节。因此若调用没有抛出异常，但实际绘制失败，仍可能生成正确尺寸的黑图或无效图。

### 8.5 截图缓存可能掩盖失败

截图管理层只在新图非空且宽高比有效时更新缓存，见 [`Capture._capture()`](../../../../../reference_repos/whimbox/whimbox/interaction/capture.py#L80)。

如果新截图返回 `None`：

- 首次失败时，调用方可能得到初始化全黑缓存；
- 成功过后再失败时，调用方通常继续得到上一张有效帧。

这会把“截图中断”表现成“游戏画面一直没有变化”。

如果 `PrintWindow` 返回的是尺寸正常的黑图，宽高比检查仍可能通过，并把黑图写入缓存。

## 9. macOS 与 Windows 的差异

上述“独立置顶窗口通常不会进入截图”的结论主要针对 Windows `PrintWindow` 主链。

macOS 实现位于 [`platform/macos/capture.py`](../../../../../reference_repos/whimbox/whimbox/platform/macos/capture.py#L34)，主要过程是：

```text
Quartz根据PID查找窗口边界
→ mss.grab抓取该屏幕矩形
```

这种方式更接近按屏幕坐标截取窗口所在区域。若其他窗口覆盖该区域，具体结果取决于 macOS 屏幕捕获和窗口合成行为，不能直接套用 Windows `PrintWindow` 的隔离结论。

如果找不到目标窗口，macOS 当前还会回退截取主显示器，这时置顶窗口显然可能进入截图。

## 10. 与其他截图方式的对比

| 截图方式 | 截图源 | 独立置顶窗口是否可能进入 | 主要特点 |
| --- | --- | --- | --- |
| 桌面 `mss.grab` | 显示器最终画面区域 | 是 | 快、兼容性较广，但受遮挡 |
| 桌面 BitBlt | 桌面DC或屏幕区域 | 是 | 传统屏幕截图 |
| `PrintWindow(HWND)` | 指定窗口请求的绘制内容 | 通常不会 | 不依赖桌面Z序，但依赖目标窗口支持 |
| Windows Graphics Capture | 指定窗口/显示器捕获项 | 通常不会 | 现代接口，适合持续帧流，但接入更复杂 |
| 游戏内截图API | 游戏自身渲染结果 | 不受外部窗口遮挡 | 需要游戏提供或注入能力 |

Whimbox 仓库中存在 [`winsdk_capture.py`](../../../../../reference_repos/whimbox/whimbox/interaction/winsdk_capture.py#L1)，尝试使用 Windows Graphics Capture，但文件已明确标注“暂时不使用”，当前生产主链仍是 `PrintWindow`。

## 11. 对后续系统设计的参考

Whimbox 已经证明了“业务识别不直接依赖桌面截图”的价值，但多游戏系统不能只在 `PrintWindow`外面再套一层接口。后续设计还必须管理窗口生命周期、捕获后端能力、多来源帧、输入所有权以及错误现场。

### 11.1 `WindowRegistry`管理窗口身份和代次

Whimbox 2.5.4 通过当前游戏 `HWND`完成截图，这一源码事实不变。但 `HWND`只是系统在当前时刻分配的句柄，不适合作为长期应用身份：游戏重启、渲染窗口重建或多开实例切换后，句柄会变化；旧句柄释放后还可能被其他窗口复用。

后续系统应由 `WindowRegistry`维护两层身份：

```text
ApplicationIdentity
├── application_id
├── executable_path
├── application_adapter
└── instance_policy

WindowInstance
├── window_id
├── window_generation
├── hwnd
├── process_id
├── executable_path
├── window_class
├── title_snapshot
├── client_rect / screen_rect
├── monitor_id / DPI
├── visible / minimized / occluded
├── discovered_at / last_verified_at
└── capture_capabilities / input_policy
```

`window_id`是系统内部的逻辑身份，`window_generation`表示这个窗口实例的生命周期代次。出现以下情况时应创建新代次，并拒绝旧异步任务继续使用原来的截图或输入请求：

- HWND变化或失效；
- HWND对应的进程ID变化；
- 游戏进程重启；
- 主渲染窗口被新的窗口替换；
- 窗口类、客户区或适配器验证结果发生不兼容变化。

`window_generation`与 `focus_epoch`含义不同：前者回答“还是不是同一个窗口实例”，后者回答“本轮输入租约是否仍然有效”。帧、动作和恢复证据应同时记录两者。

### 11.2 `CaptureRouter`按能力选择后端

建议继续使用平台抽象，但由 `CaptureRouter`根据应用配置和本次需求选择实际后端：

```text
CaptureRouter
├── PrintWindowBackend
├── WindowsGraphicsCaptureBackend
├── DesktopDuplicationBackend
├── DesktopRegionBackend
├── ApplicationSpecificBackend
└── ReplayFrameBackend
```

每个后端应声明能力，而不是只有一个“能否截图”的布尔值：

```text
supports_background_capture
supports_minimized_window
captures_desktop_composition
captures_injected_overlay
provides_frame_timestamp
provides_api_result
expected_pixel_formats
known_limitations
```

路由过程可以表示为：

```text
CaptureRequest
-> 核对application_id、window_id和window_generation
-> 根据请求选择候选后端
-> 捕获并验证结果
-> 必要时切换到允许的备用后端
-> 报告实际来源和降级原因
```

后端切换不能静默发生。例如从 `PrintWindow`切换到桌面区域截图后，图像开始受到遮挡和Z序影响，其证据语义已经改变。上层必须能看到实际后端和切换原因。

截图与识别原则上还应保持只读。`capture()`或 `observe()`不能为了关闭弹窗、聚焦窗口或翻页而夹带业务输入，否则后续无法判断画面变化由用户、计划还是观察模块造成。

### 11.3 多来源截图承担不同职责

多窗口环境建议保留三类观察源：

| 来源 | 主要回答的问题 |
| --- | --- |
| 目标窗口截图 | 游戏内部页面、HUD、菜单和角色状态是什么 |
| 桌面截图 | 用户实际看到什么、哪个窗口在顶部、是否有系统弹窗和遮挡 |
| 其他受管理窗口的低频见证帧 | 微量输入是否错误作用到其他游戏或应用 |

三类图像不能互相替代：

```text
PrintWindow取得正确游戏画面
!= 桌面前台一定是该游戏
!= 物理鼠标一定由该游戏消费
```

目标窗口帧适合持续运行页面识别；桌面帧适合在取得焦点、输入验证和异常恢复时采集；见证窗口通常只需在输入探测期或错误发生后低频采集。

多来源证据还要按单调时钟对齐。若目标窗口前帧、桌面后帧和见证窗口帧相隔过久，就不能把它们拼成同一次输入归属判断。证据对象应保存各自时间戳和允许的最大配对时间差。

### 11.4 帧对象不能只是一张 NumPy 数组

上层 OCR、YOLO、CLIP、SAM 和状态识别可以继续只依赖统一像素格式，但运行系统需要一个完整的 `FrameRecord`：

```json
{
  "frame_id": 18240,
  "frame_seq": 731,
  "capture_attempt_id": "capture_000918",
  "application_id": "infinity_nikki",
  "window_id": "infinity_nikki.main",
  "window_generation": 12,
  "window_handle": "0x000A032E",
  "process_id": 18432,
  "capture_backend": "window_printwindow",
  "capture_wall_time": "",
  "capture_monotonic_ns": 0,
  "api_success": true,
  "is_fresh": true,
  "capture_error": null,
  "cache_age_ms": 0,
  "original_resolution": [2560, 1440],
  "normalized_resolution": [1920, 1080],
  "pixel_format": "BGR",
  "client_rect": [0, 0, 2560, 1440],
  "dpi": 144,
  "image_hash": "",
  "content_valid": true,
  "validation_reasons": []
}
```

如果捕获失败后返回缓存旧帧，必须保留原帧的 `frame_id`和原始采集时间，另建 `capture_attempt_id`记录本次失败。不能仅因为调用方这次又取得了一个数组，就为旧像素生成看似更新的帧ID。

### 11.5 帧有效性需要分层判断

完整验证链可以分为：

```text
窗口身份有效
-> 捕获API成功
-> 尺寸、通道和内存布局有效
-> 内容不是黑帧、损坏帧或明显旧帧
-> 与目标应用的视觉锚点相容
-> 在本次验证要求下足够新鲜
```

可以组合以下信号：

- `PrintWindow`或其他后端的原始返回值；
- 全黑、近全黑比例和图像方差；
- 连续帧哈希、感知哈希和帧时间戳；
- 游戏进程、窗口代次和渲染窗口存活状态；
- 已知HUD、页面锚点或色彩分布；
- 桌面帧与目标窗口帧的局部一致性；
- 捕获后端是否刚刚切换。

连续帧相同不能单独证明截图失效，因为暂停菜单和静态设置页可能合法地长时间不变。只有结合本应变化的输入、游戏存活状态、后端时间戳和视觉锚点，才能判断是静态画面还是旧帧。

### 11.6 截图资源与物理输入仲裁分离

多个窗口的截图和视觉推理可以并行，但真实键鼠必须经过 `GlobalInputArbiter`取得桌面级独占资源 `desktop_physical_input`：

```text
后台CaptureRouter
-> 可并行获取多个窗口帧

GlobalInputArbiter
-> 取得desktop_physical_input
-> 建立绑定window_generation的FocusLease
-> 验证前台窗口并生成focus_epoch
-> 获取目标、桌面和必要见证窗口的前置帧
-> 发送安全探测或正式输入
-> InputOwnershipVerifier比较各来源后置响应
-> 验证Outcome后释放租约
```

`InputOwnershipVerifier`应同时检查窗口元数据和实际视觉响应。目标窗口成为前台只是必要条件；目标窗口对输入产生符合预期的变化，且其他见证窗口没有异常响应，才构成输入就绪证据。

若焦点变化、窗口代次变化或输入作用到其他窗口，系统应使当前 `focus_epoch`失效，清空未注入队列，释放所有已按下输入，并结束当前 StepAttempt。截图成功不能阻止这条安全路径。

### 11.7 截图结果与公共 `ExecutionOutcome`分开

捕获模块可以输出 `CaptureDecision`，例如成功、新鲜、旧帧、黑帧或后端降级；这些专业状态只能成为 `reason_code`或证据，不能自行宣布业务 Step 成功。公共终态统一使用：

```text
ExecutionOutcome = SUCCEEDED / FAILED / UNKNOWN / CANCELLED / BLOCKED
```

`BLOCKED`表示输入前已经明确知道当前条件不允许执行，并且没有发送业务输入。捕获配置静态不兼容、目标能力当前不可用或焦点租约门禁不满足时，应优先使用 `BLOCKED`；如果输入已经发出而后置画面不可用，仍必须使用 `UNKNOWN`。

一次统一结果使用 `reason_code + FailureAttribution + SampleDisposition`表达专业原因、最终归因和样本统计去向：

| 场景 | `ExecutionOutcome` | `reason_code` | `FailureAttribution` | `SampleDisposition` |
| --- | --- | --- | --- | --- |
| 新鲜且来源正确的后置帧确认目标状态 | `SUCCEEDED` | `POSTCONDITION_CONFIRMED` | - | `VALID_SUCCESS` |
| 输入前确定当前捕获后端不支持目标窗口 | `BLOCKED` | `CAPTURE_BACKEND_INCOMPATIBLE` | `EXPECTED_UNAVAILABLE` | `EXPECTED_UNAVAILABLE` |
| 捕获配置与当前游戏或窗口版本静态不兼容 | `BLOCKED` | `CAPTURE_PROFILE_INCOMPATIBLE` | `GAME_VERSION_INCOMPATIBLE` | `QUARANTINED` |
| 无法建立有效焦点租约且未发送业务输入 | `BLOCKED` | `FOCUS_EPOCH_EXPIRED` | `INPUT_ROUTING` | `INPUT_ROUTING_INVALID` |
| 前置帧可能是旧帧，无法确认前置状态 | `UNKNOWN` | `STALE_FRAME` | `AUTOMATION_ENVIRONMENT` | `QUARANTINED` |
| 输入已经发出，后置截图失败 | `UNKNOWN` | `CAPTURE_FAILED` | `AUTOMATION_ENVIRONMENT` | `TRANSIENT_ENVIRONMENT_FAULT` |
| 目标窗口未达到结果，且见证窗口明确响应了输入 | `FAILED` | `INPUT_OWNERSHIP_VIOLATION` | `INPUT_ROUTING` | `INPUT_ROUTING_INVALID` |
| 用户取消且捕获worker和输入均已清理 | `CANCELLED` | `USER_CANCELLED` | - | `USER_CANCELLED` |

专业捕获模块可以立即给出 `reason_code`，但 `FailureAttribution`必须在统一证据验证后生成，`SampleDisposition`再决定更新捕获健康、输入路由、版本兼容或业务能力指标。只有新鲜、来源正确且命中预期后置状态的证据才能支持 `SUCCEEDED`。黑帧、缓存帧和来源不明的帧都不能用于降低关系正确性，也不能支持盲目重试。

`RecoveryRun`始终独立保存。切换截图后端、重新发现窗口或恢复焦点成功，只证明恢复动作本身成功；原 StepAttempt 的 `BLOCKED`、`FAILED`或 `UNKNOWN`不能被改写，恢复后需要创建新的 Attempt。

### 11.8 建立 ALAS 式错误证据包

ALAS 值得参考的是保留最近截图和日志的取证思想。新系统可以为每个受管理应用维护短时环形缓冲，并在异常时冻结为不可变证据包：

```text
CaptureEvidenceBundle
├── application_id / window_id / window_generation
├── WindowRegistry生命周期事件
├── 最近目标窗口帧
├── 最近桌面帧和见证窗口帧
├── 捕获后端选择与切换记录
├── API结果、帧哈希和有效性判定
├── FocusLease和focus_epoch变化
├── 输入动作与InputOwnershipDecision
├── 状态识别原始分数和版本
├── StepAttempt及公共ExecutionOutcome
├── reason_code / FailureAttribution / SampleDisposition
├── RecoveryRun
└── CleanupRecord
```

证据包应能回答：是 `PrintWindow`失败、游戏停止渲染、缓存掩盖失败、窗口已经重建、焦点被抢走、输入被其他窗口消费，还是状态识别器本身出错。恢复运行必须另存，不能覆盖原 Attempt 的失败或未知结果。

### 11.9 推荐的模块边界

```text
WindowRegistry
-> 提供经过代次验证的窗口身份

CaptureRouter
-> 选择后端并生成FrameRecord

FrameValidator
-> 生成帧新鲜度和内容有效性证据

PerceptionPipeline
-> 对有效帧运行OCR、YOLO、CLIP、SAM或传统视觉

GlobalInputArbiter
-> 串行管理桌面物理输入

InputOwnershipVerifier
-> 用多来源前后帧验证真实输入消费者

EvidenceStore
-> 保存帧、动作、ExecutionOutcome、归因、样本处置、恢复和清理记录
```

这样更换截图技术不会影响识别器，增加多窗口仲裁也不要求把焦点逻辑塞进每个 OCR 或任务函数。

## 12. 最终总结

Whimbox 前端置顶而不污染游戏截图的根本原因是：

```text
它截的是目标游戏窗口内容
而不是显示器最终合成画面
```

Windows 主链通过游戏 `HWND`、客户区尺寸、内存 DC、兼容位图和 `PrintWindow` 直接取得目标窗口图像。独立前端窗口属于另一个 `HWND`，不会因为位于游戏上方就自动进入这张内存位图。

但这种隔离并不代表截图永远可靠。游戏停止渲染、窗口最小化、硬件交换链不兼容、注入式覆盖层和未检查的 `PrintWindow` 失败都可能产生黑帧或旧帧。

后续系统还必须坚持四个边界：

```text
HWND只是当前捕获句柄，不是长期应用身份
窗口截图只证明看到了什么，不证明输入由谁消费
缓存旧帧不能伪装成新的FrameRecord
捕获故障、输入路由故障和业务关系失败必须分别归因
```

因此完整方案应由 `WindowRegistry + CaptureRouter + FrameValidator + GlobalInputArbiter + InputOwnershipVerifier`共同组成，并通过多来源帧、窗口代次、`focus_epoch`和 ALAS 式证据包保存可回放的执行现场。
